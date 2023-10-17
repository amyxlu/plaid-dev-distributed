from evo.dataset import FastaDataset
from torch.utils.data import random_split
import torch
import math

fasta_file = "/shared/amyxlu/data/uniref90/uniref90.fasta"

ds = FastaDataset(fasta_file, cache_indices=True)
n_val = 50000 
n_train = len(ds) - n_val  # 153,726,820
train_set, val_set = random_split(
    ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
)
batch_size=64
dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)

import safetensors
import k_diffusion as K
from k_diffusion.models.esmfold import ESMFold
import einops

esmfold = ESMFold()
device = torch.device("cuda:1")
esmfold.to(device)
esmfold.eval()
# esmfold.set_chunk_size(128)

feats_all = []
from tqdm import tqdm
for batch in tqdm(dataloader): 
    sequences = batch[1]
    print(max([len(seq) for seq in sequences]))
    with torch.no_grad():
        sequences = K.utils.get_random_sequence_crop_batch(sequences, 512, min_len=30) 
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