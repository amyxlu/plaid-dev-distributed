from pathlib import Path
import json
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.multiprocessing import Process, Queue, Lock, Value, Array
from torch.multiprocessing import Process, Queue, Lock, Value, set_start_method
import os
import k_diffusion as K
import safetensors
import k_diffusion as K
from k_diffusion.models.esmfold import ESMFold


from evo.dataset import FastaDataset
import time
import dataclasses


@dataclasses.dataclass
class ShardConfig:
    fasta_file: str = "/shared/amyxlu/data/uniref90/uniref90.fasta"
    output_dir: str = "/shared/amyxlu/data/uniref90/shards/uniref90"
    batch_size: int = 256 
    max_seq_len: int = 512
    min_seq_len: int = 30
    num_workers: int = 4
    num_batches_per_shard: int = 500


def make_fasta_dataloader(fasta_file, batch_size, num_workers=4):
    # for loading batches into ESMFold and embedding
    ds = FastaDataset(fasta_file, cache_indices=True)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers
    )


def embed_batch(esmfold, sequences, max_len=512, min_len=30):
    with torch.no_grad():
        sequences = K.utils.get_random_sequence_crop_batch(
            sequences, max_len=max_len, min_len=min_len
        )
        seq_lens = torch.tensor([len(seq) for seq in sequences], dtype=torch.int16, device="cpu")
        embed_results = esmfold.infer_embedding(sequences)
        embs = embed_results["s"].detach().cpu()  # (N, L, 1024)
        embs = embs.to(dtype=torch.bfloat16)
    return embs, seq_lens


def worker(model, num_batches_per_shard, input_queue, output_queue, gpu_lock, counter, counter_lock, progress_counter):
    # each worker processes the embeddings for one shard.
    # grabs batches from the input queue of batches, embeds them, aggregates until it has enough batches to make a shard,
    # then saves the shard to disk.
    while True:
        sequence = input_queue.get()
        if sequence is None:  # Poison pill means shutdown
            break

        # Process data
        all_embs = []
        all_seq_lens = []
        with gpu_lock:  # Ensure only one process accesses the GPU at a time
            for _ in range(num_batches_per_shard):
                embs, seq_lens = embed_batch(model, sequence)
                embs, seq_lens = embs.cpu(), seq_lens.cpu()
                all_embs.append(embs)
                all_seq_lens.append(seq_lens)
        all_embs = torch.cat(all_embs, dim=0)
        all_seq_lens = torch.cat(all_seq_lens, dim=0)
        
        # Increment shard counter
        with counter_lock:
            shard_number = counter.value
            counter.value += 1
        
        output_queue.put((all_embs, all_seq_lens, shard_number))

        # Increment progress counter
        with progress_counter.get_lock():
            progress_counter.value += 1


def save_embeddings(output_queue, outdir):
    while True:
        item = output_queue.get()
        if item is None:  # Poison pill means shutdown
            break

        embs, seq_lens, shard_number = item
        safetensors.torch.save_file(
            {
                "embeddings": embs,
                "seq_len": seq_lens,
            }, outdir / f"shard{shard_number:04}.pt"
        )
        print(f"saved {embs.shape[0]} sequences to shard {shard_number}")

    del embs, seq_lens


def make_esmfold():
    start = time.time()
    print("making esmfold...")
    esmfold = ESMFold(make_trunk=False)
    end = time.time()
    print(f"done making esmfold in {end - start:.2f} seconds.")

    device = torch.device("cuda")
    esmfold.to(device)
    esmfold.eval()
    esmfold.requires_grad_(False)
    for param in esmfold.parameters():
        param.requires_grad = False
    # esmfold.set_chunk_size(128)
    return esmfold


def main(cfg: ShardConfig):
    start = time.time()
    try:
        set_start_method('spawn')  # Set the start method to 'spawn'
    except RuntimeError:
        pass  # Ignore if the start method has already been set

    esmfold = make_esmfold()

    print("creating queues...")
    input_queue = Queue()
    output_queue = Queue()
    gpu_lock = Lock()
    counter = Value('i', 0)  # Shared counter, initialized to 0
    counter_lock = Lock()
    progress_counter = Value('i', 0)  # Shared progress counter

    dataloader = make_fasta_dataloader(cfg.fasta_file, cfg.batch_size, cfg.num_workers)
    num_batches = len(dataloader)
    outdir = Path(cfg.output_dir) / f"max_len_{cfg.max_seq_len}"
    if not outdir.exists():
        outdir.mkdir(parents=True)

    # Start the tqdm progress bar
    pbar = tqdm(total=num_batches, desc="Embedding sequences")

    # Start worker processes
    processes = []
    for i in range(cfg.num_workers):
        p = Process(target=worker, args=(esmfold, cfg.num_batches_per_shard, input_queue, output_queue, gpu_lock, counter, counter_lock, progress_counter))
        p.start()
        processes.append(p)

    # Start a process for saving embeddings
    save_process = Process(target=save_embeddings, args=(output_queue, outdir))
    save_process.start()

    # Feed data to the input queue
    for (sequence, _) in dataloader:  
        input_queue.put(sequence)
    
    # Update tqdm in the main process
    while progress_counter.value < num_batches:
        pbar.update(progress_counter.value - pbar.n)
        time.sleep(0.1)

    # Send poison pills to shut down workers
    for i in range(cfg.num_workers):
        input_queue.put(None)

    for p in processes:
        p.join()

    # Shut down the saving process
    output_queue.put(None)
    save_process.join()

    # Close the queues
    input_queue.close()
    output_queue.close()

    # Release locks
    gpu_lock.release()
    counter_lock.release()

    # Close the progress bar
    pbar.close()
    argdict = K.config.dataclass_to_dict(cfg)
    with open(outdir / "config.json", "w") as f:
        json.dump(argdict, f, indent=2)
    
    end = time.time()
    print(f"done in {end - start:.2f} seconds.")

if __name__ == "__main__":
    cfg = ShardConfig()
    main(cfg)
