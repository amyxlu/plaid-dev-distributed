import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import h5py

from evo.dataset import FastaDataset

from safetensors.torch import load_file
import glob
from pathlib import Path
import pickle
import typing as T
import lightning as L

from plaid.transforms import mask_from_seq_lens


# class ShardedTensorDataset(IterableDataset):
#     def __init__(self, shard_dir, split=None):
#         shard_dir = Path(shard_dir)
#         shard_files = glob.glob(str(shard_dir / "*.pt"))
#         num_shards = len(shard_files)
#         assert num_shards > 0, f"no shards found in {shard_dir}"

#         if split is not None:
#             if split == "train":
#                 shard_files = shard_files[: int(0.8 * num_shards)]  # 5 shards for CATH
#             elif split == "val":
#                 shard_files = shard_files[int(0.8 * num_shards) :]  # 1 shard for CATH
#             else:
#                 raise ValueError(f"split must be one of ['train', 'val'], got {split}")
#         self.shard_files = shard_files

#     def __iter__(self):
#         for file in self.shard_files:
#             tensor_dict = load_file(file)
#             assert set(tensor_dict.keys()) == set(("embeddings", "seq_len"))
#             embs, seqlens = tensor_dict["embeddings"], tensor_dict["seq_len"]
#             mask = mask_from_seq_lens(embs, seqlens)
#             shuffled_idxs = torch.randperm(embs.size(0))
#             for i in shuffled_idxs:
#                 yield embs[i, ...], mask[i, ...]


class TensorShardDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split: T.Optional[str] = None,
        shard_dir: str = "/shared/amyxlu/data/cath/shards",
        header_to_sequence_file: str = "/shared/amyxlu/data/cath/sequences.pkl",
        seq_len: int = 64,
        dtype: str = "bf16",
    ):
        super().__init__()
        self.dtype = dtype
        self.seq_len = seq_len
        self.shard_dir = Path(shard_dir)
        self.header_to_seq = pickle.load(open(header_to_sequence_file, "rb"))
        self.embs, self.masks, self.ordered_headers = self.load_partition(split)

    def load_partition(self, split: T.Optional[str] = None):
        if not split is None:
            assert split in ("train", "val")
            datadir = self.shard_dir / split
        else:
            datadir = self.shard_dir
        datadir = datadir / f"seqlen_{self.seq_len}" / self.dtype
        data = load_file(datadir / "shard0000.pt")
        assert data.keys() == set(("embeddings", "seq_len"))
        emb, seqlen = data["embeddings"], data["seq_len"]
        mask = mask_from_seq_lens(emb, seqlen)

        ordered_headers = open(datadir / "../shard0000.txt").readlines()
        ordered_headers = [h.rstrip("\n") for h in ordered_headers]
        return emb, mask, ordered_headers

    def __len__(self):
        return self.embs.size(0)

    def __getitem__(self, idx: int) -> T.Tuple[torch.Tensor, torch.Tensor, str]:
        header = self.ordered_headers[idx]
        if "|" in header:
            header = header.split("|")[-1].split("/")[0]
        return (
            self.embs[idx, ...],
            self.masks[idx, ...],
            self.header_to_seq[header],
        )


class H5ShardDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split: T.Optional[str] = None,
        shard_dir: str = "/shared/amyxlu/data/cath/shards",
        header_to_sequence_file: str = "/shared/amyxlu/data/cath/sequences.pkl",
        max_seq_len: int = 64,
        dtype: str = "fp32",
    ):
        super().__init__()
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.shard_dir = Path(shard_dir)
        self.header_to_seq = pickle.load(open(header_to_sequence_file, "rb"))
        self.seq_to_header = {v: k for k, v in self.header_to_seq.items()}
        self.embs, self.sequences = self.load_partition(split)

    def load_partition(self, split: T.Optional[str] = None):
        if not split is None:
            assert split in ("train", "val")
            datadir = self.shard_dir / split
        else:
            datadir = self.shard_dir
        datadir = datadir / f"seqlen_{self.max_seq_len}" / self.dtype
        
        with h5py.File(datadir / "shard0000.h5", "r") as f:
            emb = torch.from_numpy(np.array(f["embeddings"]))
            sequence = list(f["sequences"])
        
        sequence = [s.decode()[:self.max_seq_len] for s in sequence]
        return emb, sequence

    def __len__(self):
        return self.embs.size(0)

    def __getitem__(self, idx: int) -> T.Tuple[torch.Tensor, torch.Tensor, str]:
        return (
            self.embs[idx, ...],
            self.sequences[idx]
        )


class CATHShardedDataModule(L.LightningDataModule):
    def __init__(
        self,
        storage_type: str = "safetensors",
        shard_dir: str = "/shared/amyxlu/data/cath/shards",
        header_to_sequence_file: str = "/shared/amyxlu/data/cath/sequences.pkl",
        seq_len: int = 64,
        batch_size: int = 32,
        num_workers: int = 0,
        dtype: str = "bf16",
    ):
        super().__init__()
        self.shard_dir = shard_dir
        self.dtype = dtype
        self.header_to_sequence_file = header_to_sequence_file
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers

        assert storage_type in ("safetensors", "hdf5")
        if storage_type == "safetensors":
            self.dataset_fn = TensorShardDataset
        elif storage_type == "hdf5":
            self.dataset_fn = H5ShardDataset

    def setup(self, stage: str = "fit"):
        if stage == "fit":
            self.train_dataset = self.dataset_fn(
                "train",
                self.shard_dir,
                self.header_to_sequence_file,
                self.seq_len,
                dtype=self.dtype,
            )
            self.val_dataset = self.dataset_fn(
                "val",
                self.shard_dir,
                self.header_to_sequence_file,
                self.seq_len,
                dtype=self.dtype,
            )
        elif stage == "predict":
            self.test_dataset = self.dataset_fn(
                "val",
                self.shard_dir,
                self.header_to_sequence_file,
                self.seq_len,
                dtype=self.dtype,
            )
        else:
            raise ValueError(f"stage must be one of ['fit', 'predict'], got {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return self.test_dataloader()


class FastaDataModule(L.LightningDataModule):
    def __init__(self, fasta_file: str, batch_size: int, max_seq_len: int = 512, train_frac: float = 0.8, num_workers: int = 0):
        super().__init__()
        self.fasta_file = fasta_file
        self.train_frac, self.val_frac = train_frac, 1 - train_frac
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_seq_len = max_seq_len
    
    def setup(self, stage: str = "fit"):
        ds = FastaDataset(self.fasta_file, cache_indices=True)
        seed = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(ds, [self.train_frac, self.val_frac], generator=seed)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False
        )
    
    def test_dataloader(self):
        return self.val_dataloader()




if __name__ == "__main__":
    # datadir = "/homefs/home/lux70/data/cath"
    # pklfile = "/homefs/home/lux70/data/cath/sequences.pkl"
    # dm = CATHShardedDataModule(
    #     shard_dir=datadir,
    #     header_to_sequence_file=pklfile,
    # )
    # dm.setup("fit")
    # train_dataloader = dm.train_dataloader()
    # batch = next(iter(train_dataloader))
    fasta_file = "/shared/amyxlu/data/uniref90/truncated.fasta"
    dm = FastaDataModule(fasta_file)
    dm.setup("fit")
    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))
    import IPython

    IPython.embed()
