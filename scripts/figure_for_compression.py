"""
Save compression and reconstructions for a batch of models
"""
import torch
from pathlib import Path
from plaid.compression.hourglass_vq import HourglassVQLightningModule
import matplotlib.pyplot as plt
import os

from plaid.datasets import CATHStructureDataModule

colors = ["184E77", "1E6091", "1A759F", "168AAD", "34A0A4", "52B69A", "76C893", "99D98C", "B5E48C", "D9ED92"]
# models = ["g8e83omk", "7str7fhl", "ich20c3q", "uhg29zk4", "13lltqha", "fbbrfqzk", "kyytc8i9", "mm9fe6x9", "8ebs7j9h"]
models = ["ich20c3q", "uhg29zk4", "13lltqha", "fbbrfqzk", "kyytc8i9", "mm9fe6x9", "8ebs7j9h"]
device = torch.device("cuda")


def load_batch():
    ########################################################################################################
    # Load batch of latents 
    ########################################################################################################

    shard_dir = "/homefs/home/lux70/storage/data/cath/shards/"
    pdb_dir = "/data/bucket/lux70/data/cath/dompdb"
    # shard_dir = "/homefs/home/lux70/storage/data/rocklin/shards/"
    # pdb_dir = "/data/bucket/lux70/data/rocklin/structures/"

    max_seq_len=512
    dm = CATHStructureDataModule(
        shard_dir,
        pdb_dir,
        seq_len=max_seq_len,
        batch_size=32,
        max_num_samples=32,
        shuffle_val_dataset=False,
        num_workers=2
    ) 
        
    dm.setup()
    val_dataloader = dm.val_dataloader()
    batch = next(iter(val_dataloader))
    print(len(val_dataloader.dataset))


    # grab saved embedding
    import torch
    from plaid.esmfold.misc import batch_encode_sequences
    from plaid.transforms import trim_or_pad_batch_first
    from plaid.utils import LatentScaler


    x = batch[0].to(device)
    sequences = batch[1]
    x = x.to(device)

    # make mask
    _, mask, _, _, _ = batch_encode_sequences(sequences)

    mask = trim_or_pad_batch_first(mask, pad_to=max_seq_len, pad_idx=0)
    mask = mask.to(device)

    # scale
    latent_scaler = LatentScaler()
    x_norm = latent_scaler.scale(x)

    print(x_norm.shape)
    print(x_norm.max())

    del dm, val_dataloader
    return x, x_norm, mask, latent_scaler, sequences


def compress_and_save(model_id, x, x_norm, mask, latent_scaler, sequences):
    print("Now running:", model_id)

    ########################################################################################################
    # Compression forward pass 
    ########################################################################################################

    root_dir = Path("/homefs/home/lux70/storage/plaid/checkpoints/hourglass_vq/")
    dirpath = root_dir / model_id
    print(os.listdir(str(dirpath)))

    model = HourglassVQLightningModule.load_from_checkpoint(dirpath / "last.ckpt")
    model = model.to(device)

    print("Downprojection factor:", model.enc.downproj_factor)
    print("Shorten factor:", model.enc.shorten_factor)


    # model forward pass!!
    recons_norm, _, _, compressed = model(x_norm, mask.bool(), log_wandb=False)
    print(compressed.shape)

    from plaid.proteins import LatentToStructure

    del model  # save some GPU space

    structure_constructor = LatentToStructure()
    structure_constructor.to(device)

    recons = latent_scaler.unscale(recons_norm)
    recons_struct = structure_constructor.to_structure(recons, sequences, return_raw_features=True, batch_size=4, num_recycles=1)
    orig_struct = structure_constructor.to_structure(x, sequences, return_raw_features=True, batch_size=4, num_recycles=1)


    ########################################################################################################
    # Write to disk 
    ########################################################################################################

    from plaid.utils import write_pdb_to_disk

    for i, pdbstr in enumerate(recons_struct[0]):
        write_pdb_to_disk(pdbstr, f"/homefs/home/lux70/cache/{model_id}/recons_pred_{i}.pdb")

    for i, pdbstr in enumerate(orig_struct[0]):
        write_pdb_to_disk(pdbstr, f"/homefs/home/lux70/cache/{model_id}/orig_pred_{i}.pdb")

    print("Finished writing to disk for", model_id)


if __name__ == "__main__":
    x, x_norm, mask, latent_scaler, sequences = load_batch()
    for model_id in models:
        compress_and_save(model_id, x, x_norm, mask, latent_scaler, sequences)
