# import torch
# from torch.utils.data import IterableDataset, DataLoader
# import safetensors

# class ShardedTensorDataset(IterableDataset):
#     def __init__(self, shard_files):
#         self.shard_files = shard_files

#     def __iter__(self):
#         for file in self.shard_files:
#             tensor_dict = safetensors.torch.load_file(file)
#             for item in tensor_dict["features"]:
#                 yield item

# # List of shard files
# shard_files = [f"shard_{i}.pt" for i in range(num_shards)]

# # Create dataset
# dataset = ShardedTensorDataset(shard_files)

# # Create dataloader
# dataloader = DataLoader(dataset, batch_size=64)

import torch
import safetensors
from tqdm import tqdm

from evo.dataset import FastaDataset
from torch.utils.data import random_split
import torch
from pathlib import Path

import dataclasses

import safetensors
import k_diffusion as K
from k_diffusion.models.esmfold import ESMFold
import einops
import time

start = time.time()
print("making esmfold...")
esmfold = ESMFold(make_trunk=False)
end = time.time()
print(f"done making esmfold in {end - start:.2f} seconds.")

device = torch.device("cuda")
esmfold.to(device)
esmfold.eval()
esmfold.requires_grad_(False)
# esmfold.set_chunk_size(128)


@dataclasses.dataclass
class ShardConfig:
    fasta_file: str = "/shared/amyxlu/data/uniref90/partial.fasta"
    output_dir: str = "/shared/amyxlu/data/uniref90/shards/partial"
    batch_size: int = 128
    max_seq_len: int = 512
    min_seq_len: int = 30
    num_workers: int = 4
    n_batches_per_shard: int = 100


def make_fasta_dataloader(fasta_file, batch_size, num_workers=4):
    # for loading batches into ESMFold and embedding
    ds = FastaDataset(fasta_file, cache_indices=True)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers
    )


def embed_batch(sequences, max_len=512, min_len=30):
    with torch.no_grad():
        sequences = K.utils.get_random_sequence_crop_batch(
            sequences, max_len=max_len, min_len=min_len
        )
        seq_lens = [len(seq) for seq in sequences]
        embed_results = esmfold.infer_embedding(sequences)
        feats = embed_results["s"].detach().cpu()  # (N, L, 1024)
        feats = feats.to(dtype=torch.float16)
    return feats, seq_lens


def make_shard(dataloader, n_batches_per_shard):
    cur_shard_tensors = []
    seq_lens = []

    for i, batch in tqdm(
        enumerate(dataloader),
        desc="Batches within shard",
        total=n_batches_per_shard,
        leave=False,
    ):
        sequences = batch[1]
        feats, seq_lens = embed_batch(sequences, cfg.max_seq_len, cfg.min_seq_len)
        cur_shard_tensors.append(feats)
        seq_lens.extend(seq_lens)
        # n_seq_in_shard = len(seq_lens)

        # if n_seq_in_shard >= num_seq_per_shard:
        if i % cfg.batches_per_shard == 0:
            yield torch.cat(cur_shard_tensors, dim=0), torch.tensor(
                seq_lens, dtype=torch.int32
            )
            cur_shard_tensors = []
            seq_lens = []
            # n_seq_in_shard = 0


def main(cfg: ShardConfig):
    print("making dataloader")
    dataloader = make_fasta_dataloader(cfg.fasta_file, cfg.batch_size, cfg.num_workers)
    num_samples = len(dataloader.dataset)
    num_shards = len(dataloader) // cfg.batches_per_shard + 1

    shard_generator = make_shard(dataloader, cfg.n_batches_per_shard)
    outdir = Path(cfg.output_dir) / f"max_len_{cfg.max_seq_len}"
    print(f"making shards, expected {num_shards} shards in {str(outdir)}")
    if not outdir.exists():
        outdir.mkdir(parents=True)


    cur_shard_tensors = []
    seq_lens = []

    for batch_idx, batch in tqdm(
        enumerate(dataloader),
    ):
        sequences = batch[1]
        feats, seq_lens = embed_batch(sequences, cfg.max_seq_len, cfg.min_seq_len)
        cur_shard_tensors.append(feats)
        seq_lens.extend(seq_lens)
        # n_seq_in_shard = len(seq_lens)

        # if n_seq_in_shard >= num_seq_per_shard:
        if i % cfg.batches_per_shard == 0:
            yield torch.cat(cur_shard_tensors, dim=0), torch.tensor(
                seq_lens, dtype=torch.int32
            )
            cur_shard_tensors = []
            seq_lens = []
            # n_seq_in_shard = 0




    for shard, seq_lens in tqdm(
        shard_generator,
        desc="Shards",
        total=num_shards
    ):
        safetensors.torch.save_file(
            {
                "features": shard,
                "seq_len": seq_lens,
            }, outdir / "shard{i:03}.pt"
        )


if __name__ == "__main__":
    cfg = ShardConfig()
    main(cfg)

# esmfold.set_chunk_size(128)

# # Define shard size

# # Save tensors into shards
# for i in range(0, len(tensors), shard_size):
#     shard = tensors[i:i+shard_size]
#     safetensors.torch.save_file({"features": shard}, f"shard_{i//shard_size}.pt")




""""
Alternative implementation, to be tested
"""
# import torch
# import torch.nn as nn
# from torch.multiprocessing import Process, Queue, Lock
# import os

# # Example Model
# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         # Define model layers here

#     def forward(self, x):
#         # Define forward pass here
#         return x

# def worker(input_queue, output_queue, gpu_lock):
#     while True:
#         data = input_queue.get()
#         if data is None:  # Poison pill means shutdown
#             break

#         # Process data
#         with gpu_lock:  # Ensure only one process accesses the GPU at a time
#             embeddings = model(data.to('cuda')).cpu()

#         output_queue.put(embeddings)

# def save_embeddings(output_queue, file_path):
#     while True:
#         embeddings = output_queue.get()
#         if embeddings is None:  # Poison pill means shutdown
#             break

#         # Save embeddings to disk
#         # Ensure thread-safe file operations

# if __name__ == '__main__':
#     num_workers = 4

#     input_queue = Queue()
#     output_queue = Queue()
#     gpu_lock = Lock()

#     # Load model and freeze weights
#     model = MyModel().to('cuda')
#     model.eval()
#     for param in model.parameters():
#         param.requires_grad = False

#     # Start worker processes
#     processes = []
#     for i in range(num_workers):
#         p = Process(target=worker, args=(input_queue, output_queue, gpu_lock))
#         p.start()
#         processes.append(p)

#     # Start a process for saving embeddings
#     save_process = Process(target=save_embeddings, args=(output_queue, 'path/to/save/embeddings'))
#     save_process.start()

#     # Feed data to the input queue
#     for data in data_loader:  # Assuming data_loader is your data source
#         input_queue.put(data)

#     # Send poison pills to shut down workers
#     for i in range(num_workers):
#         input_queue.put(None)

#     for p in processes:
#         p.join()

#     # Shut down the saving process
#     output_queue.put(None)
#     save_process.join()
