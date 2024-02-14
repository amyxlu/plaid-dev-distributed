from evo.dataset import FastaDataset
from tqdm import tqdm, trange
import os
from pathlib import Path
import numpy as np
import einops
import safetensors
from torch.utils.data import random_split
import torch
import math
import argparse

from plaid.utils import get_model_device
from plaid.transforms import get_random_sequence_crop_batch

ACCEPTED_LM_EMBEDDER_TYPES = [
    "esmfold",  # 1024 -- i.e. t36_3B with projection layers, used for final model
    "esm2_t48_15B_UR50D",  # 5120 
    "esm2_t36_3B_UR50D",  # 2560
    "esm2_t33_650M_UR50D",  # 1280
    "esm2_t30_150M_UR50D",  # 64e $EMBED
    "esm2_t12_35M_UR50D",  # dim=480
    "esm2_t6_8M_UR50D"  # dim=320
]

# pcluster
DATASET_TO_FASTA_FILE = {
    "uniref": "",
    "cath": "/homefs/home/lux70/storage/data/cath/cath-dataset-nonredundant-S40.atom.fa",
    "pfam": "",
}

# # dgx5
# DATASET_TO_FASTA_FILE = {
#     "uniref": "/shared/amyxlu/data/uniref90/uniref90.fasta",
#     "cath": "/shared/amyxlu/data/cath/cath-dataset-nonredundant-S40.atom.fa",
#     "pfam": "/shared/amyxlu/data/pfam/Pfam-A.fasta",
# }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm_embedder_type", type=str, default="esmfold")
    parser.add_argument("--dataset", type=str, default="cath")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_val", type=int, default=5000)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--min_len", type=int, default=16)
    parser.add_argument("--suffix", type=str, default="nov28")
    return parser.parse_args()


def check_model_type(lm_embedder_type):
    assert lm_embedder_type in ACCEPTED_LM_EMBEDDER_TYPES


def get_dataloader(dataset, batch_size=64, n_val=5000):
    print("Making dataloader")
    fasta_file = DATASET_TO_FASTA_FILE[dataset]

    ds = FastaDataset(fasta_file, cache_indices=True)
    n_train = len(ds) - n_val  # 153,726,820
    train_set, val_set = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)
    return dataloader


def make_embedder(lm_embedder_type):
    if lm_embedder_type == "esmfold":
        print("making esmfold model")
        # from plaid.denoisers.esmfold import ESMFold
        from plaid.esmfold import esmfold_v1
        embedder = esmfold_v1()
        alphabet = None
    else:
        print('loading LM from torch hub')
        embedder, alphabet = torch.hub.load("facebookresearch/esm:main", lm_embedder_type)
    
    embedder = embedder.eval().to("cuda")
    for param in embedder.parameters():
        param.requires_grad = False
    return embedder, alphabet


def calc_stats(x, mask):
    mask = einops.repeat(mask, "N L -> N L C", C=x.shape[-1]).long()
    x *= mask
    print("calc max"); channel_max = x.cpu().numpy().max(axis=(0,1))
    print("calc min"); channel_min = x.cpu().numpy().min(axis=(0,1))

    print("calc means")
    channel_means = x.sum(dim=(0,1)) / mask.sum(dim=(0,1))

    print("calc stds") 
    _chan_means = einops.repeat(channel_means, "C -> N L C", N=x.shape[0], L=x.shape[1])
    channel_stds = (x - _chan_means).pow(2).sum(dim=(0,1)) / mask.sum(dim=(0,1))
    channel_means, channel_stds = channel_means.cpu().numpy(), channel_stds.cpu().numpy()
    return channel_max, channel_min, channel_means, channel_stds


def save_npy_pkl(outdir, channel_means, channel_stds, channel_max, channel_min):
    outdir = Path(outdir)
    print("save means"); np.save(outdir / "channel_mean.pkl.npy", channel_means, allow_pickle=True)
    print("save std"); np.save(outdir / "channel_std.pkl.npy", channel_stds, allow_pickle=True)
    print("save max"); np.save(outdir / "channel_max.pkl.npy", channel_max, allow_pickle=True)
    print("save min"); np.save(outdir / "channel_min.pkl.npy", channel_min, allow_pickle=True)


def main():
    args = parse_args()
    check_model_type(args.lm_embedder_type)
    if not args.lm_embedder_type == "esmfold":
        repr_layer = int(args.lm_embedder_type.split("_")[1][1:])
    else:
        repr_layer = None
    embedder, alphabet = make_embedder(args.lm_embedder_type)

    dataloader = get_dataloader(args.dataset, args.batch_size, args.n_val)
    # outdir = Path(os.path.dirname(__file__)) / f"../cached_tensors/{args.dataset}/{args.lm_embedder_type}/subset_{args.n_val}_{args.suffix}"
    outdir = Path(os.environ['HOME']) / f"plaid_cached_tensors/{args.dataset}/{args.lm_embedder_type}/subset_{args.n_val}_{args.suffix}" 
    if not outdir.exists():
        outdir.mkdir(parents=True)

    def embed_batch(sequences, batch_converter):
        batch = [("", seq) for seq in sequences]
        _, _, tokens = batch_converter(batch)
        device = get_model_device(embedder)
        tokens = tokens.to(device)
        mask = (tokens != alphabet.padding_idx)
        with torch.no_grad():
            results = embedder(tokens, repr_layers=[repr_layer], return_contacts=False)
        return results["representations"][repr_layer], mask
    
    def embed_batch_esmfold(sequences):
        with torch.no_grad():
            embed_results = embedder.infer_embedding(sequences)
            feats = embed_results["s"].detach().cpu()  # (N, L, 1024)
            masks = embed_results["mask"].detach().cpu()  # (N, L)
        return feats, masks
    
    #### loop through batches and begin collecting embeddings ####
    xs, masks = [], [] 

    for batch in tqdm(dataloader):
        _, sequences = batch
        sequences = get_random_sequence_crop_batch(sequences, args.seq_len, args.min_len)

        if args.lm_embedder_type == "esmfold":
            x, mask = embed_batch_esmfold(sequences)
        else:
            batch_converter = alphabet.get_batch_converter()
            x, mask = embed_batch(sequences, batch_converter)

        xs.append(x.detach().cpu())
        masks.append(mask.detach().cpu())
    xs, masks = torch.cat(xs), torch.cat(masks)

    print("Saving stats to", outdir)
    channel_max, channel_min, channel_means, channel_stds = calc_stats(xs, masks)
    for arr in channel_max, channel_min, channel_means, channel_stds:
        assert arr.shape == (xs.shape[-1],)  

    save_npy_pkl(outdir, channel_means, channel_stds, channel_max, channel_min)


if __name__ == "__main__":
    main()