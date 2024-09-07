from pathlib import Path
import argparse

from evo.dataset import FastaDataset
from tqdm import tqdm
from torch.utils.data import random_split
import torch
import einops
import safetensors

from plaid.transformers import get_random_sequence_crop_batch
from cheap.pretrained import CHEAP_pfam_shorten_2_dim_32


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta_file", type=str, default="/data/lux70/data/pfam/val.fasta")
    parser.add_argument("--max_num_batches", type=int, default=200)
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--outdir", type=str, default="/data/lux70/data/pfam/")
    return parser.parse_args()


args = parse_args()
device = torch.device("cuda")

"""
Dataset
"""
ds = FastaDataset(args.fasta_file, cache_indices=True)
dataloader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

"""
CHEAP (ESMFold + autoencoder)
"""
cheap_pipeline = CHEAP_pfam_shorten_2_dim_32()
cheap_pipeline.to(device)


"""
Make features and save
"""
feats_all = []

for i, batch in tqdm(enumerate(dataloader)):
    if i >= args.max_num_batches:
        break

    sequences = batch[1]
    sequences = get_random_sequence_crop_batch(sequences, 512, min_len=30)
    with torch.no_grad():
        feats, mask = cheap_pipeline.encode(sequences)

    feats = feats.detach().cpu()  # (N, L, 32)
    masks = mask.detach().cpu()  # (N, L)
    
    masks = einops.repeat(masks, "N L -> N L C", C=32)  # (N, L, 32)
    feats = feats * masks  # (N, L, 32)
    feats = feats.sum(dim=1) / masks.sum(dim=1)  # (N, 32)
    feats_all.append(feats)

feats = torch.cat(feats_all, dim=0)

# safetensors
outdir = Path(args.outdir)
safetensors.torch.save_file({"features": feats}, "holdout_mean_pool.pt")
