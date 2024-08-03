import io
from pathlib import Path
import typing as T
from multiprocessing import Value

import webdataset as wds
import numpy as np
from lightning import LightningDataModule

from ._metadata_helpers import MetadataParser
from ..typed import PathLike


def pad_or_trim(arr, max_L):
    if arr.shape[0] < max_L:
        padding = max_L - arr.shape[0]
        arr = np.pad(arr, ((0, padding), (0, 0)), mode='constant')
    elif arr.shape[0] > max_L:
        arr = arr[:max_L]
    return arr


def _decode_header(header):
    return header.decode()


def _decode_numpy(npy, max_length):
    x = np.load(io.BytesIO(npy)).astype(np.float32)
    x = pad_or_trim(x, max_length)
    return x


class FunctionOrganismDataModule:
    def __init__(
            # TODO: proper PathLike typing
            self,
            train_shards: str,  # should follow brace expansion format
            val_shards: str,
            go_metadata_fpath: PathLike,
            organism_metadata_fpath: PathLike,
            train_epoch_num_samples: int = 1_000_000,
            val_epoch_num_samples: int = 100_000,
            max_length: int = 512,
            cache_dir: T.Optional[PathLike] = None,
            batch_size: int = 64,
            shuffle_buffer=1000,
            shuffle_initial=1000,
            num_workers: int = 1,
            # pin_memory: bool = False,
        ):
        super().__init__()
        self.train_shards = train_shards
        self.val_shards = val_shards
        self.max_length = max_length
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_buffer = shuffle_buffer
        self.shuffle_initial = shuffle_initial

        self.train_dataset_size = self.get_dataset_size(train_shards)
        self.val_dataset_size = self.get_dataset_size(val_shards)
        self.train_epoch_num_samples = train_epoch_num_samples
        self.val_epoch_num_samples = val_epoch_num_samples
        self.train_epoch_size = train_epoch_num_samples // batch_size
        self.val_epoch_size = val_epoch_num_samples // batch_size

        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True)
        
        self.metadata_parser = MetadataParser(
            go_metadata_fpath=go_metadata_fpath,
            organism_metadata_fpath=organism_metadata_fpath
        )
    
    def get_dataset_size(self, shards_brace_expand: str):
        # TODO:
        return -1

    def make_sample(self, sample, max_length):
        """From a loaded sample from a webdataset, extract the relevant conditioning fields."""
        embedding = _decode_numpy(sample["embedding.npy"], max_length)
        header = _decode_header(sample["header.txt"])
        sample_id = sample["__key__"]
        local_path = sample["__local_path__"]

        pfam_id = self.metadata_parser.header_to_pfam_id(header)
        go_idx = [self.metadata_parser.header_to_go_idx(h) for h in header]
        organism_idx = [self.metadata_parser.header_to_organism_idx(h) for h in header]
        return embedding, go_idx, organism_idx, pfam_id, sample_id, local_path
    
    def make_dataloader(self, split, epoch_size):
        assert split in ["train", "val"]
        if split == "train":
            path = self.train_shards
            shuffle_buffer = self.shuffle_buffer
            shuffle_initial = self.shuffle_initial
        else:
            path = self.val_shards
            shuffle_buffer = 0 
            shuffle_initial = 0

        # batching for improving worker-to-loader speed
        dataset = (
            wds.WebDataset(path, resampled=True, shardshuffle=(split == "train"), cache_dir=self.cache_dir)
            .shuffle(shuffle_buffer, initial=shuffle_initial)
            .map(lambda x: self.make_sample(x, self.max_length))
            .batched(self.batch_size)
        )

        # unbatch, shuffle, and form final batch for SGD (if training)
        dataloader = (
            wds.WebLoader(dataset, batch_size=None, shuffle=False, pin_memory=True, num_workers=self.num_workers)
            .unbatched()
            .shuffle(shuffle_buffer, initial=shuffle_initial)
            .with_epoch(epoch_size)
            .with_length(epoch_size)
            .batched(self.batch_size)
        )
        return dataset, dataloader

    def setup(self, stage=None):
        self.train_ds, self.train_dl = self.make_dataloader("train", self.train_epoch_size)
        self.val_ds, self.val_ds = self.make_dataloader("val", self.val_epoch_size)
        
    def train_dataloader(self):
        return self.train_dl 
    
    def val_dataloader(self):
        return self.val_dl 

