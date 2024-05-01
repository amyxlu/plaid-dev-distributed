"""
Embed with ESMFold, compress, and save to h5py with each header getting its own index
"""
import random
import typing as T
from pathlib import Path

import pickle
import h5py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import numpy as np
from evo.dataset import FastaDataset

from plaid.utils import embed_batch_esmfold, LatentScaler, npy
from plaid.esmfold import esmfold_v1
from plaid.transforms import trim_or_pad_batch_first
from plaid.compression.hourglass_vq import HourglassVQLightningModule


PathLike = T.Union[Path, str]


class FastaToH5:
    """Class that deals with writeable H5 file creation."""
    def __init__(
        self,
        compression_model_id: str,
        hourglass_ckpt_dir: PathLike,
        fasta_file: PathLike,
        output_dir: PathLike,
        batch_size: int,
        max_dataset_size: T.Optional[int] = None,
        num_workers: int = 8,
        max_seq_len: int = 512,
        esmfold: T.Optional[torch.nn.Module] = None,
        pad_to_even_number: T.Optional[int] = None,
        train_split_frac: float = 0.8,
        latent_scaler_mode: T.Optional[str] = "channel_minmaxnorm",
        split_header: bool = False,
        device_mode: str = "cuda",
    ):
        # basic attributes
        self.compression_model_id = compression_model_id
        self.hourglass_ckpt_dir = hourglass_ckpt_dir
        self.fasta_file = fasta_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_seq_len = max_seq_len  
        self.pad_to_even_number = pad_to_even_number
        self.train_split_frac = train_split_frac
        self.val_split_frac = 1. - train_split_frac
        self.split_header = split_header
        self.max_dataset_size = max_dataset_size
        self.device = torch.device(device_mode)

        self.latent_scaler = LatentScaler(latent_scaler_mode)
        self.train_dataloader, self.val_dataloader = self._make_fasta_dataloaders()

        # set up esmfold
        if esmfold is None:
            esmfold = esmfold_v1()
        self.esmfold = esmfold
        self.esmfold.to(self.device)

        # set up processing
        self.hourglass_model = self._make_hourglass()
        self.hourglass_model.to(self.device)

        # set up output path
        dirname = f"hourglass_{compression_model_id}"
        outdir = Path(output_dir) / dirname / f"seqlen_{max_seq_len}"
        self.train_h5_path = outdir / "train.h5"
        self.val_h5_path = outdir / "val.h5"
        self._maybe_make_dir(self.train_h5_path)
        self._maybe_make_dir(self.val_h5_path)

        # metadata to be saved
        self.hid_dim = 1024 // self.hourglass_model.enc.downproj_factor
        self.shorten_factor = self.hourglass_model.enc.shorten_factor

    def _maybe_make_dir(self, outpath):
        if not Path(outpath).parent.exists():
            Path(outpath).parent.mkdir(parents=True)

    def _make_hourglass(self) -> HourglassVQLightningModule: 
        ckpt_path = Path(self.hourglass_ckpt_dir) / str(self.compression_model_id) / "last.ckpt"
        print("Loading hourglass from", str(ckpt_path))
        model = HourglassVQLightningModule.load_from_checkpoint(ckpt_path)
        model = model.eval()
        return model

    def _make_fasta_dataloaders(self) -> T.Tuple[DataLoader, DataLoader]:
        # potentially subset dataset
        ds = FastaDataset(self.fasta_file, cache_indices=True)
        if self.max_dataset_size is not None:
            indices = random.sample(range(len(ds)), self.max_dataset_size)
            ds = torch.utils.data.Subset(ds, indices)
            print(f"Subsetting dataset to {len(ds)} data points.")
        
        train_ds, val_ds = torch.utils.data.random_split(ds, [self.train_split_frac, self.val_split_frac], generator=torch.Generator().manual_seed(42))
        train_dataloader = torch.utils.data.DataLoader(
            train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=False
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=False
        )
        return train_dataloader, val_dataloader

    def process_and_compress(self, feats, mask):
        device = self.hourglass_model.device
        x_norm = self.latent_scaler.scale(feats) 
        del feats

        x_norm, mask = x_norm.to(device), mask.to(device)

        # for now, this is only for the Rocklin dataset, thus hardcard the target length
        if self.pad_to_even_number:
            self.max_seq_len = 64 
            x_norm = trim_or_pad_batch_first(x_norm, pad_to=64)
            mask = trim_or_pad_batch_first(mask, pad_to=64)

        # compressed_representation manipulated in the Hourglass compression module forward pass
        # to return the detached and numpy-ified representation based on the quantization mode.
        with torch.no_grad():
            _, _, _, compressed_representation = self.hourglass_model(x_norm, mask.bool(), log_wandb=False)
        return compressed_representation
    
    def run_batch(self, fh: h5py._hl.files.File, batch: T.Tuple[str, str], cur_idx: int) -> T.Tuple[np.ndarray, T.List[str], T.List[str]]:
        headers, sequences = batch

        if self.split_header:
            # for CATH:
            headers = [s.split("|")[2].split("/")[0] for s in headers]

        """
        1. make LM embeddings
        """    
        feats, mask, sequences = embed_batch_esmfold(self.esmfold, sequences, self.max_seq_len, embed_result_key="s", return_seq_lens=False)
        
        """
        2. make hourglass compression
        """
        compressed = self.process_and_compress(feats, mask)
        assert compressed.shape[-1] == self.hid_dim

        compressed = npy(compressed)
        del feats, mask

        """
        3. Write to h5 
        """
        for i in range(compressed.shape[0]):
            sequence = sequences[i] 
            data = compressed[i, :len(sequence), :].astype(np.float32)
            # ds = fh.create_dataset(str(cur_idx), data=data, dtype="f", compression="gzip")
            # ds = fh.create_dataset(str(cur_idx), data=data, dtype="f", compression="lzf")
            ds = fh.create_dataset(str(cur_idx), data=data, dtype="f")
            ds.attrs['sequence'] = sequence
            cur_idx += 1
        
        return cur_idx

        
    def make_h5_database(self, h5_path, dataloader):
        # set global index per database
        cur_idx = 0
        print("Making h5 database at", h5_path)

        # open handle
        fh = h5py.File(h5_path, "w")

        # store metadata
        fh.attrs['max_seq_len'] = self.max_seq_len
        fh.attrs['shorten_factor'] = self.shorten_factor
        fh.attrs['compressed_hid_dim'] = self.hid_dim
        fh.attrs['dataset_size'] = len(dataloader.dataset)

        # writes each data point as its own dataset within the run batch method
        for batch in tqdm(dataloader, desc=f"Running through batches for for {len(dataloader.dataset):,} samples"):
            cur_idx = self.run_batch(fh, batch, cur_idx)

        # close handle
        fh.close()
    
    def run(self):
        self.make_h5_database(self.val_h5_path, self.val_dataloader)
        self.make_h5_database(self.train_h5_path, self.train_dataloader)
    

def main():
    fasta_to_h5 = FastaToH5(
        compression_model_id="jzlv54wl",
        hourglass_ckpt_dir="/homefs/home/lux70/storage/plaid/checkpoints/hourglass_vq",
        fasta_file="/homefs/home/lux70/storage/data/pfam/Pfam-A.fasta",
        output_dir=f"/homefs/home/lux70/storage/data/pfam/compressed/subset_5000_redo",
        batch_size=128,
        max_dataset_size=5000,
    )
    fasta_to_h5.run()


if __name__ == "__main__":
    main()