"""
Hacky but working script that saves embeddings to disk along with its
sequence and header as a H5 file. Handles different:

* LM types (e.g. ESMFold vs. smaller ESM models)
* datasets (just swap out the FASTA file)
* header parsing schemes (For CATH, will parse according to the pattern, otherwise saves the header as is)
* sequence lengths
* dtypes (currently only fp32, fp16 in the .h5 file version. saving bf16 to disk is only supported by safetensors) 

note: loads model twice, once for training and once for loading. Is slow and can be trivially speeded up but
retained inefficiency to minimize introducing new bugs.
"""

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
from plaid.utils import embed_batch_esmfold, embed_batch_esm, make_embedder
from plaid.transforms import get_random_sequence_crop_batch
from plaid.constants import ACCEPTED_LM_EMBEDDER_TYPES
import time

import argparse


def check_model_type(lm_embedder_type):
    assert lm_embedder_type in ACCEPTED_LM_EMBEDDER_TYPES


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute embeddings")
    parser.add_argument(
        "--fasta_file",
        type=str,
        default="/data/lux70/data/cath/cath-dataset-nonredundant-S40.atom.fa",
        help="Path to the fasta file",
    )
    parser.add_argument(
        "--train_output_dir",
        type=str,
        default="/data/lux70/data/cath/shards/train",
        help="Directory for training output shards",
    )
    parser.add_argument(
        "--val_output_dir",
        type=str,
        default="/data/lux70/data/cath/shards/val",
        help="Directory for validation output shards",
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--max_seq_len", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers")
    parser.add_argument(
        "--compression", type=str, default="hdf5", choices=["safetensors", "hdf5"], help="Compression type"
    )
    parser.add_argument("--train_frac", type=float, default=0.8, help="Training fraction")
    parser.add_argument(
        "--dtype", type=str, default="fp32", choices=["bf16", "fp32", "fp64"], help="Data type"
    )
    parser.add_argument("--lm_embedder_type", type=str, default="esmfold", choices=ACCEPTED_LM_EMBEDDER_TYPES)
    parser.add_argument(
        "--max_num_samples",
        type=int,
        default=-1,
        help="If specified, uses up to this many samples before splitting into train/val",
    )
    return parser.parse_args()


def _get_dtype(dtype):
    if dtype == "bf16":
        return torch.bfloat16
    elif dtype == "fp32":
        return torch.float32
    elif dtype == "fp64":
        return torch.float64
    else:
        raise ValueError(f"invalid dtype {dtype}")


# def make_embedder(lm_embedder_type):
#     start = time.time()
#     print(f"making {lm_embedder_type}...")

#     if "esmfold" in lm_embedder_type:
#         # from plaid.denoisers.esmfold import ESMFold
#         from plaid.esmfold import esmfold_v1
#         embedder = esmfold_v1()
#         alphabet = None
#     else:
#         print('loading LM from torch hub')
#         embedder, alphabet = torch.hub.load("facebookresearch/esm:main", lm_embedder_type)

#     embedder = embedder.eval().to("cuda")

#     for param in embedder.parameters():
#         param.requires_grad = False

#     end = time.time()
#     print(f"done loading model in {end - start:.2f} seconds.")

#     return embedder, alphabet


# def embed_batch_esmfold(esmfold, sequences, max_len=512, embed_result_key="s"):
#     with torch.no_grad():
#         # don't disgard short sequences since we're also saving headers
#         sequences = get_random_sequence_crop_batch(
#             sequences, max_len=max_len, min_len=0
#         )
#         seq_lens = [len(seq) for seq in sequences]
#         embed_results = esmfold.infer_embedding(sequences, return_intermediates=True)
#         feats = embed_results[embed_result_key].detach()
#         seq_lens = torch.tensor(seq_lens, device="cpu", dtype=torch.int16)
#     return feats, seq_lens, sequences


# def embed_batch_esm(embedder, sequences, batch_converter, repr_layer, max_len=512):
#     sequences = get_random_sequence_crop_batch(
#         sequences, max_len=max_len, min_len=0
#     )
#     seq_lens = [len(seq) for seq in sequences]
#     seq_lens = torch.tensor(seq_lens, device="cpu", dtype=torch.int16)

#     batch = [("", seq) for seq in sequences]
#     _, _, tokens = batch_converter(batch)
#     device = get_model_device(embedder)
#     tokens = tokens.to(device)

#     with torch.no_grad():
#         results = embedder(tokens, repr_layers=[repr_layer], return_contacts=False)
#         feats = results["representations"][repr_layer]

#     return feats, seq_lens, sequences


def make_fasta_dataloaders(fasta_file, batch_size, num_workers=4, max_num_samples=-1):
    # for loading batches into ESMFold and embedding
    ds = FastaDataset(fasta_file, cache_indices=True)
    if max_num_samples != -1:
        print(f"Orignal size of total dataset {len(ds):,}")
        idxs = torch.randperm(len(ds))[:max_num_samples]
        ds = torch.utils.data.Subset(ds, idxs)
        print(f"Subsampled size of total dataset {len(ds):,}")

    train_ds, val_ds = torch.utils.data.random_split(ds, [0.8, 0.2])
    train_dataloader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
    )
    return train_dataloader, val_dataloader


def make_shard(
    embedder,
    dataloader,
    max_seq_len,
    batch_converter=None,
    repr_layer=None,
    lm_embedder_type="esmfold",
    split_header=True,
):
    if "esmfold" in lm_embedder_type:
        if lm_embedder_type == "esmfold":
            embed_result_key = "s"
        elif lm_embedder_type == "esmfold_pre_mlp":
            embed_result_key = "s_post_softmax"
        else:
            raise ValueError(f"lm embedder type {lm_embedder_type} not understood.")
    else:
        embed_result_key = None
        assert not batch_converter is None
        assert not repr_layer is None

    cur_shard_tensors = []
    cur_shard_lens = []
    cur_headers = []
    cur_sequences = []

    for batch in tqdm(dataloader, desc="Loop through batches"):
        headers, sequences = batch

        if split_header:
            # for CATH:
            headers = [s.split("|")[2].split("/")[0] for s in headers]

        if "esmfold" in lm_embedder_type:
            feats, seq_lens, sequences = embed_batch_esmfold(
                embedder, sequences, max_seq_len, embed_result_key
            )
        else:
            feats, seq_lens, sequences = embed_batch_esm(
                embedder, sequences, batch_converter, repr_layer, max_seq_len
            )

        feats, seq_lens = feats.cpu(), seq_lens.cpu()
        from plaid.transforms import trim_or_pad_batch_first

        # to avoid errors in cases where whole batch is less than the target
        feats = trim_or_pad_batch_first(feats, max_seq_len)

        if not len(headers) == len(sequences) == len(feats) == len(seq_lens):
            import pdb

            pdb.set_trace()

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
        },
        outdir / f"shard{shard_number:04}.pt",
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
        print(f"saved {embs.shape[0]} sequences to shard {shard_number} at {str(outdir)} as h5 file")
        print("num unique proteins,", len(np.unique(pdb_id)))
        print("num unique sequences,", len(np.unique(sequences)))
    del embs


def run(dataloader, output_dir, cfg):
    print(cfg)
    """
    Set up: ESMFold vs. other embedder
    """
    lm_embedder_type = cfg.lm_embedder_type
    embedder, alphabet = make_embedder(lm_embedder_type)

    if "esmfold" in lm_embedder_type:
        batch_converter = None
        repr_layer = None
    else:
        batch_converter = alphabet.get_batch_converter()
        repr_layer = int(lm_embedder_type.split("_")[1][1:])

    outdir = Path(output_dir) / lm_embedder_type / f"seqlen_{cfg.max_seq_len}"
    if not outdir.exists():
        outdir.mkdir(parents=True)

    """
    Make shards: wrapper fn
    """
    emb_shard, seq_lens, headers, sequences = make_shard(
        embedder,
        dataloader,
        cfg.max_seq_len,
        batch_converter,
        repr_layer,
        lm_embedder_type,
        split_header="cath-dataset" in cfg.fasta_file,
    )

    if cfg.compression == "safetensors":
        save_safetensor_embeddings(emb_shard, seq_lens, 0, outdir, cfg.dtype)
    elif cfg.compression == "hdf5":
        save_h5_embeddings(emb_shard, sequences, headers, 0, outdir, cfg.dtype)
    else:
        raise ValueError(f"invalid compression type {cfg.compression}")


def main(cfg):
    print(f"making dataloader from {cfg.fasta_file}")
    train_dataloader, val_dataloader = make_fasta_dataloaders(
        cfg.fasta_file, cfg.batch_size, cfg.num_workers, cfg.max_num_samples
    )

    print("creating val dataset")
    run(val_dataloader, cfg.val_output_dir, cfg)

    print("creating train dataset")
    run(train_dataloader, cfg.train_output_dir, cfg)


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
