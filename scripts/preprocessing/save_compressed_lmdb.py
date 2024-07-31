"""
Uses LMDB to embed with ESMFold, then compress.
"""

import random
import typing as T
from pathlib import Path
import pickle

import lmdb
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


class FastaToLMDB:
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
        lmdb_map_size: int = int(1e9),
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
        self.val_split_frac = 1.0 - train_split_frac
        self.split_header = split_header
        self.max_dataset_size = max_dataset_size
        self.device = torch.device(device_mode)
        self.lmdb_map_size = int(lmdb_map_size)

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
        self.train_lmdb_path = outdir / "train.lmdb"
        self.val_lmdb_path = outdir / "val.lmdb"
        self._maybe_make_dir(self.train_lmdb_path)
        self._maybe_make_dir(self.val_lmdb_path)

        # metadata to be saved
        self.all_headers = []
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

        train_ds, val_ds = torch.utils.data.random_split(
            ds, [self.train_split_frac, self.val_split_frac], generator=torch.Generator().manual_seed(42)
        )
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
            _, _, _, compressed_representation, downsampled_mask = self.hourglass_model(x_norm, mask.bool(), log_wandb=False)
        return compressed_representation

    def run_batch(self, env, batch) -> T.Tuple[np.ndarray, T.List[str], T.List[str]]:
        headers, sequences = batch

        if self.split_header:
            # for CATH:
            headers = [s.split("|")[2].split("/")[0] for s in headers]

        """
        1. make LM embeddings
        """
        feats, mask, sequences = embed_batch_esmfold(
            self.esmfold, sequences, self.max_seq_len, embed_result_key="s", return_seq_lens=False
        )

        """
        2. make hourglass compression
        """
        compressed = self.process_and_compress(feats, mask)
        sequence_lengths = mask.sum(dim=1).squeeze().detach().cpu().numpy()
        assert compressed.shape[-1] == self.hid_dim

        compressed = npy(compressed)
        del feats, mask, sequences

        """
        2. Write to LMDB transaction
        """
        with env.begin(write=True) as txn:
            for i in range(compressed.shape[0]):
                # save each key/value pair for each sample, trimming only to the actual sequence length
                key = headers[i].encode("utf-8")
                value = compressed[i, : sequence_lengths[i], :].tobytes()
                self.all_headers.append(key)
                txn.put(key, value)

    def make_lmdb_database(self, lmdb_path, dataloader):
        env = lmdb.open(str(lmdb_path), map_size=self.lmdb_map_size, create=True)
        print("Making LMDB database at", lmdb_path)
        for batch in tqdm(
            dataloader, desc=f"Running through batches for for {len(dataloader.dataset):,} samples"
        ):
            self.run_batch(env, batch)

        # add some metadata
        print("Adding final metadata to LMDB database...")
        with env.begin(write=True) as txn:
            txn.put(b"max_seq_len", self.max_seq_len.to_bytes(length=2, signed=False))
            txn.put(b"shorten_factor", self.shorten_factor.to_bytes(length=1, signed=False))
            txn.put(b"hid_dim", self.hid_dim.to_bytes(length=2, signed=False))
            txn.put(b"all_headers", pickle.dumps(self.all_headers))
        print("Finished making dataset at", lmdb_path)
        env.close()

    def run(self):
        self.make_lmdb_database(self.val_lmdb_path, self.val_dataloader)
        self.make_lmdb_database(self.train_lmdb_path, self.train_dataloader)


def main():
    fasta_to_lmdb = FastaToLMDB(
        compression_model_id="jzlv54wl",
        hourglass_ckpt_dir="/data/lux70/plaid/checkpoints/hourglass_vq",
        fasta_file="/data/lux70/data/pfam/Pfam-A.fasta",
        output_dir=f"/data/lux70/data/pfam/compressed/subset_2M",
        batch_size=128,
        max_dataset_size=2_000_000,
    )
    fasta_to_lmdb.run()


if __name__ == "__main__":
    main()
