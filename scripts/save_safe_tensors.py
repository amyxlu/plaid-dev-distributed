import torch
import safetensors
from tqdm import tqdm, trange
import json

from evo.dataset import FastaDataset
import torch
from pathlib import Path
import h5py

import dataclasses

import safetensors
import plaid as K
from plaid.esmfold import ESMFold
import time


@dataclasses.dataclass
class ShardConfig:
    fasta_file: str = "/shared/amyxlu/data/cath/cath-dataset-nonredundant-S40.atom.fa"
    train_output_dir: str = "/shared/amyxlu/data/cath/shards/train"
    val_output_dir: str = "/shared/amyxlu/data/cath/shards/val"
    batch_size: int = 256 
    max_seq_len: int = 64
    min_seq_len: int = 16
    num_workers: int = 4
    num_batches_per_shard: int = 20
    compression: str = "safetensors"  # "safetensors", "hdf5"
    shard: bool = False
    train_frac: float = 0.8


def make_fasta_dataloaders(fasta_file, batch_size, num_workers=4):
    # for loading batches into ESMFold and embedding
    ds = FastaDataset(fasta_file, cache_indices=True)
    train_ds, val_ds = torch.utils.data.random_split(ds, [0.8, 0.2])
    train_dataloader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
    )
    return train_dataloader, val_dataloader


def embed_batch(esmfold, sequences, max_len=512, min_len=30):
    with torch.no_grad():
        sequences = K.utils.get_random_sequence_crop_batch(
            sequences, max_len=max_len, min_len=min_len
        )
        seq_lens = [len(seq) for seq in sequences]
        embed_results = esmfold.infer_embedding(sequences)
        feats = embed_results["s"].detach()  # (N, L, 1024)
        seq_lens = torch.tensor(seq_lens, device="cpu", dtype=torch.int16)
    return feats, seq_lens


def make_shard(esmfold, dataloader, n_batches_per_shard, max_seq_len, min_seq_len):
    # TODO: also save aatype / sequence
    cur_shard_tensors = []
    cur_shard_lens = []

    for _ in trange(n_batches_per_shard, leave=False):
        batch = next(iter(dataloader))
        sequences = batch[1]
        feats, seq_lens = embed_batch(esmfold, sequences, max_seq_len, min_seq_len)
        feats, seq_lens = feats.cpu(), seq_lens.cpu()
        cur_shard_tensors.append(feats)
        cur_shard_lens.append(seq_lens)

    cur_shard_tensors = torch.cat(cur_shard_tensors, dim=0)
    cur_seq_lens = torch.cat(cur_shard_lens, dim=0)
    return cur_shard_tensors, cur_seq_lens


def save_safetensor_embeddings(embs, seq_lens, shard_number, outdir):
    embs = embs.to(dtype=torch.bfloat16)
    seq_lens = seq_lens.to(dtype=torch.int16)
    safetensors.torch.save_file(
        {
            "embeddings": embs,
            "seq_len": seq_lens,
        }, outdir / f"shard{shard_number:04}.pt"
    )
    print(f"saved {embs.shape[0]} sequences to shard {shard_number}")


def save_h5_embeddings(embs, seq_lens, shard_number, outdir):
    # doesn't work with bfloat16
    embs = embs.to(dtype=torch.float32)
    seq_lens = seq_lens.to(dtype=torch.int16)
    with h5py.File(str(outdir / f"shard{shard_number:04}.h5"), "w") as f:
        f.create_dataset("embeddings", data=embs.numpy())
        f.create_dataset("seq_len", data=seq_lens.numpy())
        print(f"saved {embs.shape[0]} sequences to shard {shard_number} as h5 file")
    del embs, seq_lens


def run(dataloader, esmfold, output_dir, cfg: ShardConfig):
    if cfg.shard:
        num_shards = len(dataloader) // cfg.num_batches_per_shard + 1
    else:
        num_shards = 1

    outdir = Path(output_dir) / f"seqlen_{cfg.max_seq_len}"
    if not outdir.exists():
        outdir.mkdir(parents=True)
    
    argsdict = dataclasses.asdict(cfg)
    with open(outdir / "config.json", "w") as f:
        json.dump(argsdict, f, indent=2)

    for shard_number in tqdm(range(num_shards), desc="Shards"):
        emb_shard, seq_lens = make_shard(esmfold, dataloader, cfg.num_batches_per_shard, cfg.max_seq_len, cfg.min_seq_len)
        if cfg.compression == "safetensors":
            save_safetensor_embeddings(emb_shard, seq_lens, shard_number, outdir)
        elif cfg.compression == "hdf5":
            save_h5_embeddings(emb_shard, seq_lens, shard_number, outdir)
        else:
            raise ValueError(f"invalid compression type {cfg.compression}")


def main(cfg: ShardConfig):
    start = time.time()
    print("making esmfold...")
    esmfold = ESMFold(make_trunk=False)
    end = time.time()
    print(f"done making esmfold in {end - start:.2f} seconds.")

    device = torch.device("cuda")
    esmfold.to(device)
    esmfold.eval()
    esmfold.requires_grad_(False)

    print("making dataloader")
    train_dataloader, val_dataloader = make_fasta_dataloaders(cfg.fasta_file, cfg.batch_size, cfg.num_workers)

    print("creating train dataset")
    run(train_dataloader, esmfold, cfg.train_output_dir, cfg)

    print("creating val dataset")
    run(val_dataloader, esmfold, cfg.val_output_dir, cfg)


if __name__ == "__main__":
    import tyro
    cfg = tyro.cli(ShardConfig) 
    main(cfg)
