from plaid.compression.hourglass_vq import HourglassVQLightningModule
from plaid.datasets import CATHShardedDataModule
import torch

device = torch.device("cuda")

dm = CATHShardedDataModule(
    storage_type="hdf5",
    shard_dir="/homefs/home/lux70/storage/data/cath/shards",
    seq_len=128,
)
dm.setup("fit")
dl = dm.train_dataloader()

from plaid.esmfold import esmfold_v1

esmfold = esmfold_v1()
model = HourglassVQLightningModule(dim=1024, esmfold=esmfold)
model = model.to(device)
import IPython

IPython.embed()

batch = next(iter(dl))
x = batch[0]
x = x.to(device)
out = model.forward(x, log_wandb=False)