"""
Embed with ESMFold, compress, and save to h5py with each header getting its own index
"""
import random
import typing as T
from pathlib import Path

import pickle
import pandas as pd
import h5py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import numpy as np
from einops import reduce
from evo.dataset import FastaDataset

from plaid.utils import embed_batch_esmfold, LatentScaler, npy
from plaid.esmfold import esmfold_v1
from plaid.transforms import trim_or_pad_batch_first
from plaid.compression.hourglass_vq import HourglassVQLightningModule


PathLike = T.Union[Path, str]


def argument_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compression_model_id", type=str, default="jzlv54wl")
    parser.add_argument("--compression_model_name", type=str, default="last.ckpt")
    parser.add_argument("--fasta_file", type=str, default="/homefs/home/lux70/storage/data/pfam/Pfam-A.fasta")
    parser.add_argument("--accession_to_clan_file", type=str, default="/homefs/home/lux70/storage/data/pfam/Pfam-A.clans.tsv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--max_dataset_size", type=int, default=30000)
    parser.add_argument("--output_dir", type=str, default="/homefs/home/lux70/storage/data/pfam/compressed/subset_30K_with_clan")
    parser.add_argument("--train_split_frac", type=float, default=0.9)
    parser.add_argument("--float_type", type=str, choices=["fp16", "fp32", "fp64"], default="fp16")
    return parser.parse_args()


def get_dtype(dtype: str):
    if "fp16":
        return np.float16
    elif "fp32":
        return np.float32
    elif "fp64":
        return np.float64
    else:
        raise ValueError(f"dtype {dtype} not recognized; must be fp16, fp32, or fp64.")


class _ToH5:
    """Class that deals with writeable H5 file creation."""
    def __init__(
        self,
        compression_model_id: str,
        hourglass_ckpt_dir: PathLike,
        fasta_file: PathLike,
        output_dir: PathLike,
        batch_size: int,
        compression_model_name: T.Optional[str] = None,
        max_dataset_size: T.Optional[int] = None,
        num_workers: int = 8,
        max_seq_len: int = 512,
        esmfold: T.Optional[torch.nn.Module] = None,
        pad_to_even_number: T.Optional[int] = None,
        train_split_frac: float = 0.8,
        latent_scaler_mode: T.Optional[str] = "channel_minmaxnorm",
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
        self.max_dataset_size = max_dataset_size
        self.device = torch.device(device_mode)
        if compression_model_name is None:
            compression_model_name = "last.ckpt"
        self.compression_model_name = compression_model_name

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
        ckpt_path = Path(self.hourglass_ckpt_dir) / str(self.compression_model_id) / self.compression_model_name
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

        # if shortened and using a Fasta loader, the latent might not be a multiple of 2
        s = self.shorten_factor 
        extra = x_norm.shape[1] % s
        if extra != 0:
            needed = s - extra
            x_norm = trim_or_pad_batch_first(x_norm, pad_to=x_norm.shape[1] + needed, pad_idx=0)

        if mask.shape[1] != x_norm.shape[1]:
            # pad with False
            mask = trim_or_pad_batch_first(mask, x_norm.shape[1], pad_idx=0)

        x_norm, mask = x_norm.to(device), mask.to(device)

        # compressed_representation manipulated in the Hourglass compression module forward pass
        # to return the detached and numpy-ified representation based on the quantization mode.
        with torch.no_grad():
            _, _, _, compressed_representation = self.hourglass_model(x_norm, mask.bool(), log_wandb=False)

        downsampled_mask = reduce(mask, 'b (n s) -> b n', 'sum', s = s) > 0
        return compressed_representation, downsampled_mask
    
    def run_batch(self, *args, **kwargs):
        raise NotImplementedError
        
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


class FastaToH5(_ToH5):
    def __init__(
        self,
        compression_model_id: str,
        hourglass_ckpt_dir: PathLike,
        fasta_file: PathLike,
        output_dir: PathLike,
        batch_size: int,
        compression_model_name: T.Optional[str] = None,
        max_dataset_size: T.Optional[int] = None,
        num_workers: int = 8,
        max_seq_len: int = 512,
        esmfold: T.Optional[torch.nn.Module] = None,
        pad_to_even_number: T.Optional[int] = None,
        train_split_frac: float = 0.8,
        latent_scaler_mode: T.Optional[str] = "channel_minmaxnorm",
        split_header: bool = False,
        device_mode: str = "cuda",
        float_type: str = "fp32"
    ):
        super().__init__(
            compression_model_id=compression_model_id,
            hourglass_ckpt_dir=hourglass_ckpt_dir,
            fasta_file=fasta_file,
            output_dir=output_dir,
            batch_size=batch_size,
            compression_model_name=compression_model_name,
            max_dataset_size=max_dataset_size,
            num_workers=num_workers,
            max_seq_len=max_seq_len,
            esmfold=esmfold,
            pad_to_even_number=pad_to_even_number,
            train_split_frac=train_split_frac,
            latent_scaler_mode=latent_scaler_mode,
            device_mode=device_mode,
        )
        self.split_header = split_header
        self.dtype = get_dtype(float_type)

    def run_batch(self, fh: h5py._hl.files.File, batch: T.Tuple[str, str], cur_idx: int) -> T.Tuple[np.ndarray, T.List[str], T.List[str]]:
        headers, sequences = batch

        if self.split_header:
            # for CATH:
            headers = [s.split("|")[2].split("/")[0] for s in headers]

        """
        1. make LM embeddings
        """    
        with torch.no_grad():
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
            data = compressed[i, :len(sequence), :].astype(self.dtype)
            # ds = fh.create_dataset(str(cur_idx), data=data, dtype="f", compression="gzip")
            # ds = fh.create_dataset(str(cur_idx), data=data, dtype="f", compression="lzf")
            ds = fh.create_dataset(str(cur_idx), data=data, dtype="f")
            ds.attrs['sequence'] = sequence
            cur_idx += 1
        
        return cur_idx


class FastaToH5Clans(_ToH5):
    def __init__(
        self,
        compression_model_id: str,
        hourglass_ckpt_dir: PathLike,
        fasta_file: PathLike,
        accession_to_clan_file: PathLike,
        output_dir: PathLike,
        batch_size: int,
        compression_model_name: T.Optional[str] = None,
        max_dataset_size: T.Optional[int] = None,
        num_workers: int = 8,
        max_seq_len: int = 512,
        esmfold: T.Optional[torch.nn.Module] = None,
        pad_to_even_number: T.Optional[int] = None,
        train_split_frac: float = 0.8,
        latent_scaler_mode: T.Optional[str] = "channel_minmaxnorm",
        device_mode: str = "cuda",
        float_type: str = "fp16"
    ):
        super().__init__(
            compression_model_id=compression_model_id,
            hourglass_ckpt_dir=hourglass_ckpt_dir,
            fasta_file=fasta_file,
            output_dir=output_dir,
            batch_size=batch_size,
            compression_model_name=compression_model_name,
            max_dataset_size=max_dataset_size,
            num_workers=num_workers,
            max_seq_len=max_seq_len,
            esmfold=esmfold,
            pad_to_even_number=pad_to_even_number,
            train_split_frac=train_split_frac,
            latent_scaler_mode=latent_scaler_mode,
            device_mode=device_mode,
        )
        self.accession_to_clan_file = accession_to_clan_file
        self.dtype = get_dtype(float_type)
        self._make_accession_to_clan_data_structures()

    def _make_accession_to_clan_data_structures(self):
        fam_to_clan_df = pd.read_csv(
            self.accession_to_clan_file,
            sep="\t",
            header=None
    )
        # read accession to clan dataframe and grab the first clan for each pfam family accession
        header = ["accession", "clan", "short_name", "gene_name", "description"]
        fam_to_clan_df.columns = header
        accession_to_clan = fam_to_clan_df.groupby("accession").first().filter(['accession','clan'], axis=1)
        accession_to_clan = accession_to_clan.to_dict()['clan']

        # create an unique index representer for each clan
        clans = fam_to_clan_df.clan.dropna().unique()
        clans.sort()

        # store mapping
        self.clans_to_idx = dict(zip(clans, np.arange(len(clans))))
        self.accession_to_clan = accession_to_clan
        self.clans = clans
    
    def _header_to_clan_idx(self, header):
        subid = header.split(" ")[-1]
        accession = subid.split(".")[0]
        clan_id = self.accession_to_clan[accession]
        if clan_id is None:
            return len(self.clans)  # dummy idx for unknown clan
        else:
            return self.clans_to_idx[clan_id]
    
    def run_batch(self, fh: h5py._hl.files.File, batch: T.Tuple[str, str], cur_idx: int) -> T.Tuple[np.ndarray, T.List[str], T.List[str]]:
        headers, sequences = batch

        """
        1. make LM embeddings
        """    
        with torch.no_grad():
            feats, mask, sequences = embed_batch_esmfold(self.esmfold, sequences, self.max_seq_len, embed_result_key="s", return_seq_lens=False)
        
        """
        2. make hourglass compression
        """
        compressed, downsampled_mask = self.process_and_compress(feats, mask)
        compressed_lens = downsampled_mask.sum(dim=-1)
        assert compressed.shape[-1] == self.hid_dim

        compressed = npy(compressed)
        del feats, mask

        clan_idxs = list(map(lambda header: self._header_to_clan_idx(header), headers))

        """
        3. Write to h5 
        """
        for i in range(compressed.shape[0]):
            data = compressed[i, :compressed_lens[i], :].astype(self.dtype)
            clan_idx = clan_idxs[i] 
            # ds = fh.create_dataset(str(cur_idx), data=data, dtype="f", compression="gzip")
            # ds = fh.create_dataset(str(cur_idx), data=data, dtype="f", compression="lzf")
            ds = fh.create_dataset(str(cur_idx), data=data, dtype="f")
            ds.attrs['clan'] = np.array([clan_idx], dtype=np.int16)
            ds.attrs['len'] = np.array([len(sequences[i])], dtype=np.int16)
            cur_idx += 1
        
        return cur_idx


def main():
    args = argument_parser()
    runner = FastaToH5Clans(
        compression_model_id=args.compression_model_id,
        compression_model_name=args.compression_model_name,
        hourglass_ckpt_dir="/homefs/home/lux70/storage/plaid/checkpoints/hourglass_vq",
        fasta_file=args.fasta_file,
        accession_to_clan_file=args.accession_to_clan_file,
        batch_size=args.batch_size,
        max_dataset_size=args.max_dataset_size,
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
        train_split_frac=args.train_split_frac,
        float_type=args.float_type
    )
    runner.run()


if __name__ == "__main__":
    main() 