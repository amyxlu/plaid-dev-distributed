import torch
import safetensors
from tqdm import tqdm, trange
import numpy as np
import json

import h5py
from evo.dataset import FastaDataset
import torch
from pathlib import Path
# import h5py

import dataclasses

from safetensors.torch import save_file, load_file
from plaid import utils
from plaid.esmfold import esmfold_v1
from plaid.transforms import get_random_sequence_crop_batch
import time

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Precompute embeddings")
    parser.add_argument('--fasta_file', type=str, default="/homefs/home/lux70/storage/data/cath/cath-dataset-nonredundant-S40.atom.fa", help='Path to the fasta file')
    parser.add_argument('--train_output_dir', type=str, default="/homefs/home/lux70/storage/data/cath/shards/train", help='Directory for training output shards')
    parser.add_argument('--val_output_dir', type=str, default="/homefs/home/lux70/storage/data/cath/shards/val", help='Directory for validation output shards')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--max_seq_len', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--compression', type=str, default="hdf5", choices=["safetensors", "hdf5"], help='Compression type')
    parser.add_argument('--train_frac', type=float, default=0.8, help='Training fraction')
    parser.add_argument('--dtype', type=str, default="fp32", choices=["bf16", "fp32", "fp64"], help='Data type')

    return parser.parse_args()

args = parse_args()

def _get_dtype(dtype):
    if dtype == "bf16":
        return torch.bfloat16
    elif dtype == "fp32":
        return torch.float32
    elif dtype == "fp64":
        return torch.float64
    else:
        raise ValueError(f"invalid dtype {dtype}")


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


def embed_batch(esmfold, sequences, max_len=512):
    with torch.no_grad():
        # don't disgard short sequences since we're also saving headers
        sequences = get_random_sequence_crop_batch(
            sequences, max_len=max_len, min_len=0
        )
        seq_lens = [len(seq) for seq in sequences]
        embed_results = esmfold.infer_embedding(sequences)
        feats = embed_results["s"].detach()  # (N, L, 1024)
        seq_lens = torch.tensor(seq_lens, device="cpu", dtype=torch.int16)
    return feats, seq_lens, sequences


def make_shard(esmfold, dataloader, max_seq_len):
    cur_shard_tensors = []
    cur_shard_lens = []
    cur_headers = []
    cur_sequences = []

    for batch in tqdm(dataloader, desc="Loop through batches"):
        headers, sequences = batch

        # for CATH:
        # headers = [s.split("|")[2].split("/")[0] for s in headers]
        feats, seq_lens, sequences = embed_batch(esmfold, sequences, max_seq_len)
        feats, seq_lens = feats.cpu(), seq_lens.cpu()

        if not len(headers) == len(sequences) == len(feats) == len(seq_lens):
            import pdb;pdb.set_trace()

        cur_headers.extend(headers)
        cur_sequences.extend(sequences)
        cur_shard_tensors.append(feats)
        cur_shard_lens.append(seq_lens)

    cur_shard_tensors = torch.cat(cur_shard_tensors, dim=0)
    cur_seq_lens = torch.cat(cur_shard_lens, dim=0)
    return cur_shard_tensors, cur_seq_lens, cur_headers, cur_sequences


def save_safetensor_embeddings(embs, seq_lens, shard_number, outdir, dtype):
    # doesn't work with strings, but does work with bfloat16
    assert isinstance(dtype, str)
    outdir = Path(outdir) / dtype
    if not outdir.exists():
        outdir.mkdir(parents=True)
    
    dtype = _get_dtype(dtype)
    embs = embs.to(dtype=dtype)
    seq_lens = seq_lens.to(dtype=torch.int16)
    save_file(
        {
            "embeddings": embs,
            "seq_len": seq_lens,
        }, outdir / f"shard{shard_number:04}.pt"
    )
    print(f"saved {embs.shape[0]} sequences to shard {shard_number} as safetensor file")


def save_h5_embeddings(embs, sequences, pdb_id, shard_number, outdir, dtype: str):
    # doesn't work with bfloat16, but does work with strings
    assert dtype in ("fp32", "fp64")
    outdir = outdir / dtype
    if not outdir.exists():
        outdir.mkdir(parents=True)
    dtype = _get_dtype(dtype)
    embs = embs.to(dtype=dtype)
    with h5py.File(str(outdir / f"shard{shard_number:04}.h5"), "w") as f:
        f.create_dataset("embeddings", data=embs.numpy())
        f.create_dataset("sequences", data=sequences)
        f.create_dataset("pdb_id", data=pdb_id)
        print(f"saved {embs.shape[0]} sequences to shard {shard_number} as h5 file")
        print("num unique proteins,", len(np.unique(pdb_id)))
        print("num unique sequences,", len(np.unique(sequences)))
    del embs


def run(dataloader, esmfold, output_dir, cfg):
    outdir = Path(output_dir) / f"seqlen_{cfg.max_seq_len}"
    if not outdir.exists():
        outdir.mkdir(parents=True)
        
    emb_shard, seq_lens, headers, sequences, = make_shard(esmfold, dataloader, cfg.max_seq_len)
    if cfg.compression == "safetensors":
        save_safetensor_embeddings(emb_shard, seq_lens, 0, outdir, cfg.dtype)
    elif cfg.compression == "hdf5":
        save_h5_embeddings(emb_shard, sequences, headers, 0, outdir, cfg.dtype)
    else:
        raise ValueError(f"invalid compression type {cfg.compression}")


def main(cfg):
    start = time.time()
    print("making esmfold...")
    esmfold = esmfold_v1()
    end = time.time()
    print(f"done making esmfold in {end - start:.2f} seconds.")

    device = torch.device("cuda:0")
    esmfold.to(device)
    esmfold.eval()
    esmfold.requires_grad_(False)

    print("making dataloader")
    train_dataloader, val_dataloader = make_fasta_dataloaders(cfg.fasta_file, cfg.batch_size, cfg.num_workers)

    print("creating val dataset")
    run(val_dataloader, esmfold, cfg.val_output_dir, cfg)

    print("creating train dataset")
    run(train_dataloader, esmfold, cfg.train_output_dir, cfg)



if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)