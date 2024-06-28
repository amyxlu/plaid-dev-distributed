from typing import List, Tuple
from pathlib import Path
import os
import dataclasses
import logging

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from plaid.datasets import FastaDataset
from plaid.esmfold import esmfold_v1, ESMFold
from plaid.compression.hourglass_vq import HourglassVQLightningModule
from plaid.utils import LatentScaler, npy


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DistributedInferenceConfig:
    compression_model_id: str = "jzlv54wl"
    compression_ckpt_dir: str = (
        "/homefs/home/lux70/storage/plaid/checkpoints/hourglass_vq/"
    )
    # fasta_file: str = "/homefs/home/lux70/storage/data/pfam/Pfam-A.fasta"
    fasta_file: str = "/homefs/home/lux70/storage/data/uniref90/partial.fasta"
    accession_to_clan_file: str = (
        "/homefs/home/lux70/storage/data/pfam/Pfam-A.clans.tsv"
    )
    batch_size: int = 64
    max_seq_len: int = 512
    max_dataset_size: int = -1
    output_dir: str = (
        "/homefs/home/lux70/storage/data/pfam/compressed/subset_30K_with_clan"
    )
    train_split_frac: float = 0.9
    float_type: str = "fp16"


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def load_hourglass(model_id, model_ckpt_dir):
    model_ckpt_fpath = Path(model_ckpt_dir) / model_id / "last.ckpt"
    return HourglassVQLightningModule.load_from_checkpoint(model_ckpt_fpath, force_infer=True)


class ShardRunner:
    """
    With GPUs placed manually on the correct devices, take FASTA dataloader
    and runs them through the ESMFold -> norm -> compression pipeline
    and returns the embedding and mask on CPU.
    """

    def __init__(
        self,
        rank: int,
        esmfold: ESMFold,
        hourglass_model: HourglassVQLightningModule,
        dataloader: torch.utils.data.DataLoader,
        args: DistributedInferenceConfig,
    ):
        self.rank = rank
        self.esmfold = esmfold  # should already be DDP and placed on rank
        self.hourglass_model = (
            hourglass_model  # should already be DDP and placed on rank
        )
        self.dataloader = dataloader  # should already use distributed sampler
        self.latent_scaler = LatentScaler()
        self.args = args

        self.max_seq_len = self.args.max_seq_len
        self.hid_dim = 1024 // self.hourglass_model.enc.downproj_factor
        self.shorten_factor = self.hourglass_model.enc.shorten_factor
        self.shard_dataset_size = len(self.dataloader.dataset)

    def open_h5_file_handle(self):
        h5_path = Path(self.args.output_dir) / f"shard{self.rank:02d}.h5"
        self.cur_shard_idx = 0
        self.fh = h5py.File(h5_path, "w")
        logger.info("Making h5 database at", h5_path)

    def write_metadata(self):
        """Dataset level metadata, written only once."""
        self.fh.attrs["max_seq_len"] = self.max_seq_len
        self.fh.attrs["shorten_factor"] = self.shorten_factor
        self.fh.attrs["compressed_hid_dim"] = self.hid_dim

        # TODO
        # self.fh.attrs["shard_dataset_size"] = len(self.dataloader.dataset)
        # self.fh.attrs["full_dataset_size"] = None
        # TODO: write config dataclass to disk

    def write_result_to_disk(
        self,
        cur_shard_idx: int,
        padded_embedding: np.ndarray,
        sequence_lengths: List[int],
        headers: List[str],
    ) -> None:
        """Takes the open h5py handle and writes the unpadded embedding to disk, each individually.
        Uses fp16 precision.
        """
        for i in range(len(padded_embedding)):
            # takes given embedding and saves it without padding
            data = padded_embedding[i, : sequence_lengths[i], :].astype(np.float16)
            header = headers[i]

            ds = self.fh.create_dataset(str(cur_shard_idx), data=data, dtype="f")
            ds.attrs["len"] = np.array([sequence_lengths[i]], dtype=np.int16)
            ds.attrs["header"] = header
            cur_shard_idx += 1

        return cur_shard_idx

    def esmfold_emb_fasta(
        self, sequences: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Takes sequence, runs through ESMFold, returns embedding and mask."""
        res = self.esmfold.infer_embedding(sequences)
        return res["s"], res["mask"]

    def apply_channel_norm(self, emb: torch.Tensor) -> torch.Tensor:
        return self.latent_scaler.scale(emb)

    def compress_normed_embedding(
        self, norm_emb: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        compressed_embedding, downsampled_mask = self.hourglass_model.forward(
            norm_emb, mask, log_wandb=False, infer_only=True
        )
        x, downsampled_mask = npy(compressed_embedding), npy(downsampled_mask)
        sequence_lengths = downsampled_mask.sum(dim=-1)
        return x, sequence_lengths

    def batch_embed(
        self, batch: Tuple[List[str], List[str]]
    ) -> Tuple[np.ndarray, List[int]]:
        headers, sequences = batch
        with torch.no_grad():
            emb, mask = self.esmfold_emb_fasta(sequences)  # (N, L, C)
            emb = self.apply_channel_norm(emb)  # (N, L, C)
            emb, sequence_lengths = self.compress_normed_embedding(
                emb, mask
            )  # (N, L // s, C')
        return emb, sequence_lengths, headers

    def run(self):
        import IPython;IPython.embed()
        # open the h5py filehandle and write metadata
        self.open_h5_file_handle()
        self.write_metadata()

        # loop through batches
        for batch in self.dataloader:
            emb, sequence_lengths, headers = self.batch_embed(batch)
            self.write_result_to_disk(emb, sequence_lengths, headers)


def process_shard_on_rank(
    rank: int,
    world_size: int,
    dataset: torch.utils.data.Dataset,
    args: DistributedInferenceConfig,
):
    """Create models on a given rank, wrap them in the ShardRunner class,
    and run the pipeline.
    """
    logger.info(f"Running on rank {rank}...")
    setup(rank, world_size)

    # Create ESMFold module
    esmfold = esmfold_v1().to(rank)
    ddp_esmfold = DDP(esmfold, device_ids=[rank])
    ddp_esmfold.eval()

    # Create hourglass module
    hourglass_model = load_hourglass(
        args.compression_model_id, args.compression_ckpt_dir
    )
    ddp_hourglass = DDP(hourglass_model, device_ids=[rank])
    ddp_hourglass.eval()

    # create distributed sampler...
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    # create shard runner
    runner = ShardRunner(
        rank=rank,
        esmfold=esmfold,
        hourglass_model=hourglass_model,
        dataloader=dataloader,
        args=args,
    )
    runner.run()


def main(args):
    world_size = torch.cuda.device_count()
    dataset = FastaDataset(args.fasta_file, cache_indices=True)
    mp.spawn(
        process_shard_on_rank,
        args=(world_size, dataset, args,),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    args = DistributedInferenceConfig
    main(args)
