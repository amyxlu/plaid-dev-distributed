from evo.dataset import FastaDataset
from tqdm import tqdm
from torch.utils.data import random_split
import torch
import einops
import safetensors
import argparse

from plaid.esmfold import esmfold_v1
from plaid.transformers import get_random_sequence_crop_batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta_file", type=str, default="/shared/amyxlu/data/rocklin/rocklin_stable.fasta")
    parser.add_argument("--n_val", type=int, default=50000)
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--outdir", type=int, default=64)
    return parser.parse_args()


args = parse_args()

"""
Dataset
"""
ds = FastaDataset(args.fasta_file, cache_indices=True)
n_train = len(ds) - args.n_val  # 153,726,820
train_set, val_set = random_split(ds, [n_train, args.n_val], generator=torch.Generator().manual_seed(42))
dataloader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size)

"""
ESMFold latent maker
"""
esmfold = esmfold_v1
device = torch.device("cuda")
esmfold.to(device)
esmfold.eval()
esmfold.set_chunk_size(128)

"""
Make features and save
"""
feats_all = []

for batch in tqdm(dataloader):
    sequences = batch[1]
    print(max([len(seq) for seq in sequences]))
    with torch.no_grad():
        sequences = get_random_sequence_crop_batch(sequences, 512, min_len=30)
        embed_results = esmfold.infer_embedding(sequences)
        feats = embed_results["s"].detach().cpu()  # (N, L, 1024)
        masks = embed_results["mask"].detach().cpu()  # (N, L)
    masks = einops.repeat(masks, "N L -> N L C", C=1024)  # (N, L, 1024)
    feats = feats * masks  # (N, L, 1024)
    feats = feats.sum(dim=1) / masks.sum(dim=1)  # (N, 1024)
    feats_all.append(feats)

feats = torch.cat(feats_all, dim=0)

# safetensors
safetensors.torch.save_file({"features": feats}, "holdout_esmfold_feats.pt")
x = safetensors.torch.load_file("esmfold_feats.pt")
