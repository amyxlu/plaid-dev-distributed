from evo.dataset import FastaDataset
from tqdm import trange

cath_fasta = "/shared/amyxlu/data/cath/cath-dataset-nonredundant-S40.atom.fa"
ds = FastaDataset(cath_fasta)

d = {}
for i in trange(len(ds)):
    header, seq = ds[i]
    d[header] = seq

import pickle
with open("/shared/amyxlu/data/cath/full.pkl", "wb") as f:
    pickle.dump(d, f)

with open("/shared/amyxlu/data/cath/full.pkl", "rb") as f:
    loaded = pickle.load(f)

from plaid.datasets import CATHShardedDataModule

dm = CATHShardedDataModule()
dm.setup("fit")
for i in trange(len(dm.train_dataset)):
    header, _ = dm.train_dataset[i]
    s = loaded[header]