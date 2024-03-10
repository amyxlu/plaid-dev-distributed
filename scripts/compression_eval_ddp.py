# mostly for saving structure
import torch
from pathlib import Path
from plaid.compression.hourglass_vq import HourglassVQLightningModule
from plaid.datasets import CATHStructureDataModule
from plaid.transforms import trim_or_pad_batch_first
from plaid.esmfold.misc import batch_encode_sequences
from plaid.utils import LatentScaler
from plaid.proteins import LatentToStructure
import os

model_hash = "2024-03-05T06-20-52"
shard_dir = "/homefs/home/lux70/storage/data/cath/shards/"
pdb_dir = "/data/bucket/lux70/data/cath/dompdb"
max_seq_len=256
batch_size=16

device = torch.device("cuda")

def load_model(model_hash):
    dirpath = Path(f"/homefs/home/lux70/storage/plaid/checkpoints/hourglass_vq/{model_hash}")
    try:
        model = HourglassVQLightningModule.load_from_checkpoint(dirpath / "last.ckpt")
    except Exception as e:
        print(e)
        print(list(dirpath.glob("*ckpt")))
    return model

def load_dataloader(shard_dir, pdb_dir, max_seq_len, batch_size):
    dm = CATHStructureDataModule(
        shard_dir,
        pdb_dir,
        seq_len=max_seq_len,
        batch_size=batch_size,
        max_num_samples=batch_size
    ) 
        
    dm.setup()
    val_dataloader = dm.val_dataloader()
    print(len(val_dataloader.dataset))
    return val_dataloader

def mask_from_sequence(sequences, max_seq_len):
    from plaid.esmfold.misc import batch_encode_sequences
    _, mask, _, _, _ = batch_encode_sequences(sequences)
    mask = trim_or_pad_batch_first(mask, pad_to=max_seq_len, pad_idx=0)
    return mask


# model forward pass!!
def get_compression(model, x_norm, mask):
    model.to(device)
    x_norm = x_norm.to(device)
    mask = mask.to(device)

    recons_norm, loss, log_dict, quant_out = model(x_norm, mask.bool(), log_wandb=False)# , return_vq_output=True)
    print(torch.mean((recons_norm - x_norm) ** 2))
    print(log_dict)
    return recons_norm, quant_out



def main(): # args
    model = load_model(model_hash)
    dataloader = load_dataloader(
        shard_dir=shard_dir,
        pdb_dir=pdb_dir,
        max_seq_len=max_seq_len,
        batch_size=batch_size
    )
    # todo: for batch in dataloader
    batch = next(iter(dataloader))
    x, sequences, orig_struct = batch 
    mask = mask_from_sequence(sequences, max_seq_len)

    latent_scaler = LatentScaler()
    x_norm = latent_scaler.scale(x)

    recons_norm, quant_out = get_compression(model, x_norm, mask)

    structure_constructor = LatentToStructure()
    structure_constructor.to(device)

    recons = latent_scaler.unscale(recons_norm)
    recons_struct = structure_constructor.to_structure(recons, sequences, return_raw_features=True, batch_size=4)
    orig_struct = structure_constructor.to_structure(x, sequences, return_raw_features=True, batch_size=4)

    # TODO: finish script, check for correctness, use PDB util written from before
    # TODO: DDP