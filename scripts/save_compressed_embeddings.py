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
from plaid.utils import embed_batch_esmfold, embed_batch_esm, make_embedder, LatentScaler
from plaid.transforms import get_random_sequence_crop_batch
from plaid.constants import ACCEPTED_LM_EMBEDDER_TYPES
from plaid.compression.hourglass_lightning import HourglassTransformerLightningModule

import argparse


def check_model_type(lm_embedder_type):
    assert lm_embedder_type in ACCEPTED_LM_EMBEDDER_TYPES


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute embeddings")
    parser.add_argument("--compressor_model_id", type=str, default="2024-02-23T05-10-53")
    parser.add_argument("--compressor_ckpt_dir", type=str, default="/homefs/home/lux70/storage/plaid/checkpoints/hourglass/")
    parser.add_argument('--fasta_file', type=str, default="/homefs/home/lux70/storage/data/cath/cath-dataset-nonredundant-S40.atom.fa", help='Path to the fasta file')
    parser.add_argument('--train_output_dir', type=str, default="/homefs/home/lux70/storage/data/cath/shards/train", help='Directory for training output shards')
    parser.add_argument('--val_output_dir', type=str, default="/homefs/home/lux70/storage/data/cath/shards/val", help='Directory for validation output shards')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--max_seq_len', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--compression', type=str, default="hdf5", choices=["safetensors", "hdf5"], help='Compression type')
    parser.add_argument('--train_frac', type=float, default=0.8, help='Training fraction')
    parser.add_argument('--dtype', type=str, default="fp32", choices=["bf16", "fp32", "fp64"], help='Data type')
    parser.add_argument("--lm_embedder_type", type=str, default="esmfold", choices=ACCEPTED_LM_EMBEDDER_TYPES)
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


def make_hourglass(ckpt_dir, ckpt_name):
    ckpt_path = Path(ckpt_dir) / str(ckpt_name) / "last.ckpt"
    print("Loading hourglass from", str(ckpt_path))
    model = HourglassTransformerLightningModule.load_from_checkpoint(ckpt_path)
    model = model.eval()
    return model


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
 

def make_shard(embedder, compressor, dataloader, max_seq_len, batch_converter=None, repr_layer=None, lm_embedder_type="esmfold", split_header=True):
    """Set up variables depending on base LM embedder type
    TODO: this will be a sharding function in the future for large, large datasets 
    """
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
    
    """base loop"""
    cur_compressed = []
    cur_sequences = []
    cur_headers = []
    cur_compression_errors = []

    for batch in tqdm(dataloader, desc="Loop through batches"):
        headers, sequences = batch
        if split_header:
            # for CATH:
            headers = [s.split("|")[2].split("/")[0] for s in headers]

        """
        Round 1: make LM embeddings
        """    
        if "esmfold" in lm_embedder_type:
            feats, mask, sequences = embed_batch_esmfold(embedder, sequences, max_seq_len, embed_result_key, return_seq_lens=False)
        else:
            raise NotImplementedError("can only use ESMFold rn")
            # feats, seq_lens, sequences = embed_batch_esm(embedder, sequences, batch_converter, repr_layer, max_seq_len)
        
        """
        Round 2: make hourglass compression
        """
        latent_scaler = LatentScaler()
        feats_norm = latent_scaler.scale(feats) 
        del feats

        with torch.no_grad():
            x_recons, compressed = compressor(feats_norm, mask.bool())

        recons_error = torch.mean((feats_norm - x_recons) ** 2)
        del x_recons, feats_norm

        cur_compression_errors.append(recons_error.item())
        cur_compressed.append(compressed.cpu())
        cur_headers.extend(headers)
        cur_sequences.extend(sequences)
        print(recons_error)

    cur_compressed = torch.cat(cur_compressed)
    return cur_compressed, cur_sequences, cur_headers


def save_h5_embeddings(embs, sequences, pdb_id, shard_number, outdir, dtype: str):
    """
    2024/02/27: This function is exactly the same as the older script to ensure that
    the hourglass & non-hourglass compressed datasets look the same, but still setting up
    two functions for future flexibility

    h5 doesn't work with bfloat16, but does work with strings
    """
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

    dirname = f"{lm_embedder_type}_hourglass_{cfg.compressor_model_id}"
    
    outdir = Path(output_dir) / dirname / f"seqlen_{cfg.max_seq_len}"
    if not outdir.exists():
        outdir.mkdir(parents=True)

    """
    Setup: hourglass
    """
    compressor = make_hourglass(cfg.compressor_ckpt_dir, cfg.compressor_model_id)
    
    """
    Make shards: wrapper fn
    TODO: for larger datasets, actually shard; here it's just all saved in one file
    """ 
    emb_shard, seq_lens, headers, sequences = make_shard(
        embedder,
        compressor,
        dataloader,
        cfg.max_seq_len,
        batch_converter,
        repr_layer,
        lm_embedder_type,
        split_header="cath-dataset" in cfg.fasta_file
    )
    
    assert cfg.compression == "hdf5", "not yet support safetensor b/c sequences as strings is more versatile"
    save_h5_embeddings(emb_shard, sequences, headers, 0, outdir, cfg.dtype)


def main(cfg):
    print(f"making dataloader from {cfg.fasta_file}")
    train_dataloader, val_dataloader = make_fasta_dataloaders(cfg.fasta_file, cfg.batch_size, cfg.num_workers)

    print("creating val dataset")
    run(val_dataloader, cfg.val_output_dir, cfg)

    print("creating train dataset")
    run(train_dataloader, cfg.train_output_dir, cfg)


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
