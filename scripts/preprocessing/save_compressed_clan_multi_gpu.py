from typing import T
import os
import dataclasses

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from plaid.esmfold import esmfold_v1


@dataclasses.dataclass
class DistributedInferenceConfig:
    world_size: int = 1
    compression_model_id: str = "jzlv54wl"
    compression_model_name: str = "last.ckpt"
    fasta_file: str = "/homefs/home/lux70/storage/data/pfam/Pfam-A.fasta"
    accession_to_clan_file: str = "/homefs/home/lux70/storage/data/pfam/Pfam-A.clans.tsv"
    batch_size: int = 64
    max_seq_len: int = 512
    max_dataset_size: int = -1
    output_dir: str = "/homefs/home/lux70/storage/data/pfam/compressed/subset_30K_with_clan"
    train_split_frac: float = 0.9
    float_type: str = "fp16"


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


class ShardRunner:
    """
    With GPUs placed manually on the correct devices, take FASTA dataloader
    and runs them through the ESMFold -> norm -> compression pipeline
    and returns the embedding and mask on CPU.
    """
    def __init__(self):
        pass

    def esmfold_emb_fasta(self, sequence: T.List[str]) -> T.Tuple[torch.Tensor, torch.Tensor]:
        """Takes sequence, runs through ESMFold, returns embedding and mask."""
        pass

    def apply_channel_norm(self, emb: torch.Tensor) -> torch.Tensor:
        pass

    def compress_normed_embedding(self, emb: torch.Tensor, mask: torch.Tensor) -> T.Tuple[np.ndarray, np.ndarray]:
        pass

    def run(self):
        pass


def process_shard_on_rank(rank: int, world_size: int, dataset: torch.utils.Dataset, batch_size: int):
    """Create models on a given rank, wrap them in the ShardRunner class,
    and run the pipeline.
    """
    print(f"Running on rank {rank}...")
    setup(rank, world_size)

    esmfold = esmfold_v1().to(rank)
    ddp_esmfold = DDP(esmfold, device_ids=[rank])
    ddp_esmfold.eval()

    hourglass_model = ...
    ddp_hourglass = DDP(hourglass_model, device_ids=[rank])
    ddp_hourglass.eval()

    # create distributed sampler...
    # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    # dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    all_embeddings = []

    # create H5 file and write metadata headers

    # DDP to rank...
    # create shard runner class...
    # runner.run() --> Each result is written to the file as it's processed




def main(args):
    world_size = torch.cuda.device_count()
    mp.spawn(process_shard_on_rank, args=(world_size,), nprocs=world_size, join=True)



if __name__ == '__main__':
    main()