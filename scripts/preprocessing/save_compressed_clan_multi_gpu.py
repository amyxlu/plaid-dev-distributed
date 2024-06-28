from typing import T
from pathlib import Path
import os
import dataclasses

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from plaid.esmfold import esmfold_v1, ESMFold
from plaid.compression.hourglass_vq import HourglassVQLightningModule
from plaid.utils import LatentScaler, npy


@dataclasses.dataclass
class DistributedInferenceConfig:
    world_size: int = 1
    compression_model_id: str = "jzlv54wl"
    compression_ckpt_dir: str = (
        "/homefs/home/lux70/storage/plaid/checkpoints/hourglass_vq/",
    )
    fasta_file: str = "/homefs/home/lux70/storage/data/pfam/Pfam-A.fasta"
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
    return HourglassVQLightningModule.load_from_checkpoint(model_ckpt_fpath)


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
        self.hourglass_model = hourglass_model  # should already be DDP and placed on rank
        self.dataloader = dataloader  # should already use distributed sampler
        self.latent_scaler = LatentScaler()
        self.args = args
    
    def open_h5_file_handle(self):
        fpath = Path(self.args.outdir) / f"shard{self.rank:02d}.h5"
        # ...
    
    def write_metadata(self):
        pass

    def write_result_to_disk(
        self, padded_embedding: np.ndarray, sequence_lengths: T.List[int], headers: T.List[str]
    ) -> None:
        """Takes the open h5py handle and writes the unpadded embedding to disk, each individually.
        Uses fp16 precision.
        """
        pass

    def esmfold_emb_fasta(
        self, sequences: T.List[str]
    ) -> T.Tuple[torch.Tensor, torch.Tensor]:
        """Takes sequence, runs through ESMFold, returns embedding and mask."""
        res = self.esmfold.infer_embedding(sequences)
        return res["s"], res["mask"]

    def apply_channel_norm(self, emb: torch.Tensor) -> torch.Tensor:
        return self.latent_scaler.scale(emb)

    def compress_normed_embedding(
        self, norm_emb: torch.Tensor, mask: torch.Tensor
    ) -> T.Tuple[np.ndarray, np.ndarray]:
        compressed_embedding, downsampled_mask = self.hourglass_model.forward(
            norm_emb, mask, log_wandb=False, infer_only=True
        )
        x, downsampled_mask = npy(compressed_embedding), npy(downsampled_mask)
        sequence_lengths = downsampled_mask.sum(dim=-1)
        return x, sequence_lengths

    def batch_embed(
        self, batch: T.Tuple[T.List[str], T.List[str]]
    ) -> T.Tuple[np.ndarray, T.List[int]]:
        headers, sequences = batch 
        emb, mask = self.esmfold_emb_fasta(sequences)  # (N, L, C)
        emb = self.apply_channel_norm(emb)  # (N, L, C)
        emb, sequence_lengths = self.compress_normed_embedding(emb, mask)  # (N, L // s, C')
        return emb, sequence_lengths, headers


    def run(self):
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
    dataset: torch.utils.Dataset,
    args: DistributedInferenceConfig,
):
    """Create models on a given rank, wrap them in the ShardRunner class,
    and run the pipeline.
    """
    print(f"Running on rank {rank}...")
    setup(rank, world_size)

    # Create ESMFold module
    esmfold = esmfold_v1().to(rank)
    ddp_esmfold = DDP(esmfold, device_ids=[rank])
    ddp_esmfold.eval()

    # Create hourglass module
    hourglass_model = load_hourglass(
        args.compression_model_id, args.compression_model_ckpt
    )
    ddp_hourglass = DDP(hourglass_model, device_ids=[rank])
    ddp_hourglass.eval()

    # create distributed sampler...
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    # create shard runner
    runner = ShardRunner()
    # create H5 file and write metadata headers

    # DDP to rank...
    # create shard runner class...
    # runner.run() --> Each result is written to the file as it's processed


def main(args):
    world_size = torch.cuda.device_count()
    mp.spawn(process_shard_on_rank, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
