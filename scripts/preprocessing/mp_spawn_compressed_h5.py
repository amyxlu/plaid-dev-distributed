"""
Embeds and compresses FASTA sequence into num_visible_devices number of shards.
"""

from typing import List, Tuple
from pathlib import Path
import os
from dataclasses import dataclass, asdict
import json
import logging

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from plaid.datasets import FastaDataset
from plaid.esmfold import esmfold_v1, ESMFold
from plaid.compression.hourglass_vq import HourglassVQLightningModule
from plaid.utils import LatentScaler, npy, print_cuda_info


logger = logging.getLogger(__name__)


@dataclass
class DistributedInferenceConfig:
    compression_model_id: str = "jzlv54wl"
    compression_ckpt_dir: str = (
        "/homefs/home/lux70/storage/plaid/checkpoints/hourglass_vq/"
    )
    # fasta_file: str = "/homefs/home/lux70/storage/data/pfam/Pfam-A.fasta"
    fasta_file: str = "/homefs/home/lux70/storage/data/uniref90/partial.fasta"
    output_dir: str = "/homefs/home/lux70/storage/data/uniref90/compressed/partial/"
    batch_size: int = 64
    max_seq_len: int = 512
    max_dataset_size: int = -1
    # train_split_frac: float = 0.9


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
    Given that models have already been placed onto individual ranks and dataset subsampled,
    run FASTA dataloader through the ESMFold -> norm -> compression pipeline,
    and writes results as its own shard as a H5 file.
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
        self.esmfold = esmfold
        self.hourglass_model = hourglass_model
        self.dataloader = dataloader  # should already use distributed sampler
        self.latent_scaler = LatentScaler()
        self.args = args

        self.max_seq_len = self.args.max_seq_len
        self.hid_dim = 1024 // self.hourglass_model.enc.downproj_factor
        self.shorten_factor = self.hourglass_model.enc.shorten_factor
        self.shard_dataset_size = len(self.dataloader.dataset)

    def open_h5_file_handle(self):
        # open h5 file handle
        h5_path = Path(self.args.output_dir) / f"shard{self.rank:04d}.h5"
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
        cur_sample_idx: int,
        padded_embedding: np.ndarray,
        sequence_lengths: List[int],
        headers: List[str],
    ) -> None:
        """Takes the open h5py handle and writes the unpadded embedding to disk, each individually.
        Uses fp16 precision.
        """
        print(padded_embedding.shape)
        for i in range(len(padded_embedding)):
            # takes given embedding and saves it without padding
            print(padded_embedding.shape)
            print(sequence_lengths[i])
            data = padded_embedding[i, : sequence_lengths[i], :].astype(np.float16)
            header = headers[i]

            ds = self.fh.create_dataset(str(cur_sample_idx), data=data, dtype="f")

            ds.attrs["header"] = header

            # refers to the sequence length, after compression
            ds.attrs["n_tokens"] = np.array([sequence_lengths[i]], dtype=np.int16)

            cur_sample_idx += 1

        return cur_sample_idx

    def esmfold_emb_fasta(
        self, sequences: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Takes sequence, runs through ESMFold, returns embedding and mask."""
        with torch.no_grad():
            res = self.esmfold.infer_embedding(sequences)
        return res["s"], res["mask"]

    def apply_channel_norm(self, emb: torch.Tensor) -> torch.Tensor:
        return self.latent_scaler.scale(emb)

    def compress_normed_embedding(
        self, norm_emb: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            compressed_embedding, downsampled_mask = self.hourglass_model.forward(
                norm_emb, mask.bool(), log_wandb=False, infer_only=True
            )
        x, downsampled_mask = npy(compressed_embedding), npy(downsampled_mask)
        sequence_lengths = downsampled_mask.sum(axis=-1)
        return x, sequence_lengths

    def batch_embed(
        self, batch: Tuple[List[str], List[str]]
    ) -> Tuple[np.ndarray, List[int]]:
        headers, sequences = batch
        sequences = [s[:self.max_seq_len] for s in sequences]

        with torch.no_grad():
            emb, mask = self.esmfold_emb_fasta(sequences)  # (N, L, C)
            emb = self.apply_channel_norm(emb)  # (N, L, C)
            emb, sequence_lengths = self.compress_normed_embedding(
                emb, mask
            )  # (N, L // s, C')
        return emb, sequence_lengths, headers

    def run(self):
        # open the h5py filehandle and write metadata
        self.open_h5_file_handle()
        self.write_metadata()

        print_cuda_info()

        cur_sample_idx = 0

        # loop through batches
        for batch in self.dataloader:
            print("batch",batch)
            emb, sequence_lengths, headers = self.batch_embed(batch)
            print("embshape", emb.shape)
            cur_sample_idx = self.write_result_to_disk(emb, sequence_lengths, headers, cur_sample_idx)


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

    # Create ESMFold module on the given rank
    esmfold = esmfold_v1().to(rank)
    esmfold.eval()
    for param in esmfold.parameters():
        param.requires_grad_(False)

    # Create hourglass module on teh given rank
    hourglass_model = load_hourglass(
        args.compression_model_id, args.compression_ckpt_dir
    )
    hourglass_model.to(rank)
    hourglass_model.eval()
    for param in hourglass_model.parameters():
        param.requires_grad_(False)
    
    # Create distributed sampler that grabs indices specific to this rank
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    sampler.set_epoch(0)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
    
    # create shard runner
    logger.info(f"Setting up runner for rank/shard {rank}...")

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

    # Make output dir and save config
    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True)

    config_dict = asdict(args)
    with open(Path(args.output_dir) / "config.json", "w") as json_file:
        json.dump(config_dict, json_file, indent=4)

    # Launch process on each rank
    mp.spawn(
        process_shard_on_rank,
        args=(world_size, dataset, args,),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    args = DistributedInferenceConfig()
    main(args)
