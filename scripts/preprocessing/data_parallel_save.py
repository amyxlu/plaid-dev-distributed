from typing import List, Tuple
from pathlib import Path
import os
import sys
import dataclasses
import logging

from tqdm import tqdm
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from plaid.datasets import FastaDataset
from plaid.esmfold import esmfold_v1, ESMFold
from plaid.compression.hourglass_vq import HourglassVQLightningModule
from plaid.utils import LatentScaler, npy


logger = logging.getLogger(__name__)


class DictDataParallel(nn.DataParallel):
    # Custom override if function returns dictionary of tensors
    def gather(self, outputs, output_device):
        if isinstance(outputs[0], dict):
            return {
                key: torch.cat([output[key].to(output_device) for output in outputs])
                for key in outputs[0]
            }
        else:
            return super().gather(outputs, output_device)


@dataclasses.dataclass
class DistributedInferenceConfig:
    # fasta_file: str = "/data/lux70/data/pfam/Pfam-A.fasta"
    fasta_file: str = "/data/lux70/data/uniref90/partial.fasta"
    output_dir: str = "/data/lux70/data/uniref90/compressed/"

    compression_model_id: str = "j1v1wv6w"
    compression_ckpt_dir: str = (
        "/data/lux70/cheap/checkpoints/"
    )

    batch_size: int = 64
    max_seq_len: int = 512
    train_split_frac: float = 0.9


def load_hourglass(model_id, model_ckpt_dir):
    model_ckpt_fpath = Path(model_ckpt_dir) / model_id / "last.ckpt"
    return HourglassVQLightningModule.load_from_checkpoint(
        model_ckpt_fpath, force_infer=True
    )


class EmbeddingRunner:
    """
    With GPUs placed manually on the correct devices, take FASTA dataloader
    and runs them through the ESMFold -> norm -> compression pipeline
    and returns the embedding and mask on CPU.
    """

    def __init__(
        self,
        esmfold: ESMFold,
        hourglass_model: HourglassVQLightningModule,
        dataloader: torch.utils.data.DataLoader,
        args: DistributedInferenceConfig,
    ):
        self.esmfold = esmfold
        self.hourglass_model = hourglass_model
        self.dataloader = dataloader
        self.latent_scaler = LatentScaler()
        self.args = args

        self.max_seq_len = self.args.max_seq_len
        self.hid_dim = 1024 // self.hourglass_model.enc.downproj_factor
        self.shorten_factor = self.hourglass_model.enc.shorten_factor

        # TODO: make these updated
        self.cur_idx = 0

    def _setup_data_parallel(self):
        world_size = torch.cuda.device_count()
        device_ids = list(range(world_size))
        print("Visible devices:", device_ids)

        # Custom wrapper around data parallel to handle dictionaries
        self.esmfold = DictDataParallel(self.esmfold, device_ids=device_ids)
        self.hourglass_model = DictDataParallel(
            self.hourglass_model, device_ids=device_ids
        )

    def _open_h5_file_handle(self):
        h5_path = (
            Path(self.args.output_dir) / self.args.compression_model_id / "data.h5"
        )

        if not h5_path.parent.exists():
            h5_path.parent.mkdir(parents=True)
        
        if h5_path.exists():
            logger.info("Appending to h5 database at", h5_path)
        else:
            logger.info("Making new database at", h5_path)

        self.fh = h5py.File(h5_path, "a")

        if len(self.fh.keys()) > 0:
            self.cur_idx = len(self.fh.keys())  # index update happens after writing
            print("Resuming from entry", self.cur_idx)

    def _write_metadata(self):
        """Dataset level metadata, written only once."""
        self.fh.attrs["max_seq_len"] = self.max_seq_len
        self.fh.attrs["shorten_factor"] = self.shorten_factor
        self.fh.attrs["compressed_hid_dim"] = self.hid_dim
        self.fh.attrs["compression_model_id"] = self.args.compression_model_id
        self.fh.attrs["total_dataset_size"] = len(self.dataloader.dataset)

    def write_example_to_disk(
        self, data: np.ndarray, seq_len: int, header: str
    ) -> None:
        """Writes a single protein embedding to its own numpy array without padding in fp16.

        Each entry has a unique numerical index. Additionally, a text file of parsed headers
        is updated.
        """
        ds = self.fh.create_dataset(str(self.cur_idx), data=data, dtype="f")
        ds.attrs["len"] = np.array([seq_len], dtype=np.int16)
        ds.attrs["header"] = header
        
        # update global index
        self.cur_idx += 1

    def write_batch_to_disk(self, padded_embedding, sequence_lengths, headers):
        """Loops through a batch of calculated tasks and writes to H5"""
        for i in range(len(padded_embedding)):
            data = padded_embedding[i, : sequence_lengths[i], :].astype(np.float16)
            header = headers[i]
            seq_len = sequence_lengths[i]
            self.write_example_to_disk(data, seq_len, header)

    def esmfold_emb_fasta(
        self, sequences: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Takes sequence, runs through ESMFold, returns embedding and mask."""
        # use .module since ESM should be wrapped in dataparallel at this point
        res = self.esmfold.module.infer_embedding(sequences)
        return res["s"], res["mask"]

    def apply_channel_norm(self, emb: torch.Tensor) -> torch.Tensor:
        return self.latent_scaler.scale(emb)

    def compress_normed_embedding(
        self, norm_emb: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        compressed_embedding, downsampled_mask = self.hourglass_model(
            norm_emb, mask.bool(), log_wandb=False, infer_only=True
        )
        x, downsampled_mask = npy(compressed_embedding), npy(downsampled_mask)
        sequence_lengths = downsampled_mask.sum(axis=-1)
        return x, sequence_lengths

    def batch_embed(
        self, batch: Tuple[List[str], List[str]]
    ) -> Tuple[np.ndarray, List[int]]:
        headers, sequences = batch
        sequences = [s[:self.args.max_seq_len] for s in sequences]

        with torch.no_grad():
            emb, mask = self.esmfold_emb_fasta(sequences)  # (N, L, C)
            emb = self.apply_channel_norm(emb)  # (N, L, C)
            emb, sequence_lengths = self.compress_normed_embedding(
                emb, mask
            )  # (N, L // s, C')
        return emb, sequence_lengths, headers

    def run(self):
        # model parallel
        self._setup_data_parallel()
        
        # open the h5py filehandle and write metadata
        self._open_h5_file_handle()
        self._write_metadata()

        # loop through batches
        pbar = tqdm(total=len(self.dataloader.dataset))
        for batch_idx, batch in enumerate(self.dataloader):
            # skip at least cur_idx number
            if (batch_idx * self.args.batch_size) < self.cur_idx:
                pbar.update(self.args.batch_size)
                pass
            else:
                with torch.no_grad():
                    emb, sequence_lengths, headers = self.batch_embed(batch)
                self.write_batch_to_disk(emb, sequence_lengths, headers)
                pbar.update(len(emb))


def main(args):
    dataset = FastaDataset(args.fasta_file, cache_indices=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Create ESMFold module
    esmfold = esmfold_v1()
    esmfold.eval().cuda()
    for param in esmfold.parameters():
        param.requires_grad_(False)

    # Create hourglass module
    hourglass_model = load_hourglass(
        args.compression_model_id, args.compression_ckpt_dir
    )
    hourglass_model.eval().cuda()
    for param in hourglass_model.parameters():
        param.requires_grad_(False)


    runner = EmbeddingRunner(
        esmfold=esmfold,
        hourglass_model=hourglass_model,
        dataloader=dataloader,
        args=args,
    )
    runner.run()


if __name__ == "__main__":
    args = DistributedInferenceConfig()
    main(args)
