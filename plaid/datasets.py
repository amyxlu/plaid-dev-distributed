import warnings
import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
from typing import List, Tuple
import h5py

from evo.dataset import FastaDataset

from safetensors.torch import load_file
import glob
from pathlib import Path
import pickle
import typing as T
import lightning as L

from .transforms import mask_from_seq_lens
from .constants import ACCEPTED_LM_EMBEDDER_TYPES


class TensorShardDataset(torch.utils.data.Dataset):
    """Loads entire dataset as one Safetensor dataset. Returns the embedding, mask, and pdb id."""

    def __init__(
        self,
        split: T.Optional[str] = None,
        shard_dir: str = "/shared/amyxlu/data/cath/shards",
        header_to_sequence_file: str = "/shared/amyxlu/data/cath/sequences.pkl",
        seq_len: int = 64,
        dtype: str = "bf16",
        *args,
        **kwargs,
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
    """Loads H5 dataset, which is able to actually store strings, but is
    not able to support bf16 storage."""

    def __init__(
        self,
        split: T.Optional[str] = None,
        shard_dir: str = "/shared/amyxlu/data/cath/shards",
        embedder: str = "esmfold",
        max_seq_len: int = 64,
        dtype: str = "fp32",
        ids_to_drop: T.Optional[T.List[str]] = None,
    ):
        super().__init__()
        self.ids_to_drop = ids_to_drop
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.shard_dir = Path(shard_dir)
        self.embedder = embedder

        self.data = self.load_partition(split, embedder)
        self.pdb_ids = list(self.data.keys())

    def drop_protein(self, pid):
        drop = False
        if not (self.ids_to_drop is None) and (pid in self.ids_to_drop):
            drop = True
        return drop

    def load_partition(self, split: T.Optional[str] = None, embedder: T.Optional[str] = None):
        """
        2024/02/15: path format:
        ${shard_dir}/${split}/${embedder}/${seqlen}/${precision}/shard0000.h5
        """
        datadir = self.shard_dir

        if not split is None:
            assert split in ("train", "val")
            datadir = datadir / split
        
        if not embedder is None:
            assert embedder in ACCEPTED_LM_EMBEDDER_TYPES
            datadir = datadir / embedder

        datadir = datadir / f"seqlen_{self.max_seq_len}" / self.dtype
        outdict = {}

        with h5py.File(datadir / "shard0000.h5", "r") as f:
            emb = torch.from_numpy(np.array(f["embeddings"]))
            sequence = list(f["sequences"])
            pdb_id = list(f["pdb_id"])
            for i in range(len(pdb_id)):
                pid = pdb_id[i].decode()
                if not self.drop_protein(pid):
                    outdict[pid] = (emb[i, ...], sequence[i].decode())
        return outdict

    def __len__(self):
        return len(self.pdb_ids)

    def get(self, idx: int) -> T.Tuple[str, T.Tuple[torch.Tensor, torch.Tensor]]:
        # return (self.embs[idx, ...], self.sequences[idx], self.pdb_id[idx])
        pid = self.pdb_ids[idx]
        return pid, self.data[pid]

    def __getitem__(
        self, idx: int
    ) -> T.Tuple[str, T.Tuple[torch.Tensor, torch.Tensor]]:
        # wrapper for non-structure dataloaders, rearrange output tuple
        pdb_id, (emb, seq) = self.get(idx)
        return emb, seq, pdb_id


class CATHStructureDataset(H5ShardDataset):
    """Wrapper around H5 shard dataset. Returns actual structure features as well."""

    def __init__(
        self,
        split: T.Optional[str] = None,
        shard_dir: str = "/shared/amyxlu/data/cath/shards",
        pdb_path_dir: str = "/shared/amyxlu/data/cath/full/dompdb",
        max_seq_len: int = 64,
        dtype: str = "fp32",
        path_to_dropped_ids: T.Optional[str] = None,
    ):
        ids_to_drop = None
        if not path_to_dropped_ids is None:
            # no longer actually needed after fixing the chain
            print("using CATH dropped IDs at", path_to_dropped_ids)
            with open(path_to_dropped_ids, "r") as f:
                ids_to_drop = f.read().splitlines()
        
        super().__init__(split, shard_dir, max_seq_len, dtype, ids_to_drop)

        from plaid.utils import StructureFeaturizer

        self.structure_featurizer = StructureFeaturizer()
        self.pdb_path_dir = Path(pdb_path_dir)
        self.max_seq_len = max_seq_len

    def __getitem__(self, idx: int):
        pdb_id, (emb, seq) = self.get(idx)
        pdb_path = self.pdb_path_dir / pdb_id
        with open(pdb_path, "r") as f:
            pdb_str = f.read()
        # try:
        #     structure_features = self.structure_featurizer(pdb_str, self.max_seq_len)
        #     return emb, seq, structure_features 
        # except KeyError as e:
        #     with open("bad_ids.txt", "a") as f:
        #         print(pdb_id, e)
        #         f.write(f"{pdb_id}\n")
        #     pass
        structure_features = self.structure_featurizer(pdb_str, self.max_seq_len)
        return emb, seq, structure_features 


class CATHShardedDataModule(L.LightningDataModule):
    def __init__(
        self,
        storage_type: str = "safetensors",
        shard_dir: str = "/shared/amyxlu/data/cath/shards",
        header_to_sequence_file: T.Optional[str] = None,
        seq_len: int = 64,
        batch_size: int = 32,
        num_workers: int = 0,
        dtype: str = "fp32",
    ):
        super().__init__()
        self.shard_dir = shard_dir
        self.dtype = dtype
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.storage_type = storage_type

        assert storage_type in ("safetensors", "hdf5")
        if storage_type == "safetensors":
            assert not header_to_sequence_file is None
            self.header_to_sequence_file = header_to_sequence_file
            self.dataset_fn = TensorShardDataset
        elif storage_type == "hdf5":
            self.dataset_fn = H5ShardDataset

    def setup(self, stage: str = "fit"):
        kwargs = {}
        if self.storage_type == "safetensors":
            kwargs["header_to_sequence_file"] = self.header_to_sequence_file

        if stage == "fit":
            self.train_dataset = self.dataset_fn(
                "train", self.shard_dir, self.seq_len, dtype=self.dtype, **kwargs
            )
            self.val_dataset = self.dataset_fn(
                "val", self.shard_dir, self.seq_len, dtype=self.dtype, **kwargs
            )
        elif stage == "predict":
            self.test_dataset = self.dataset_fn(
                "val", self.shard_dir, self.seq_len, dtype=self.dtype, **kwargs
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


class CATHStructureDataModule(L.LightningDataModule):
    """Lightning datamodule that loads cached CATH tensors and structure features.
    Returns a tuple of (embedding, sequence_strings, dictionary_of_structure_features)

    Note: the structure features includes a list of keys that includes 'sequence'; this is the sequence
    parsed from the PDB file, and includes the full, untrimmed sequence.
    The sequence strings are saved from the original sequences that produced the
    embeddings, and always trimmed to self.seq_len. The string identity should be usually the
    same, unless in cases where the FASTA and PDB files in the CATH database differs in sidechain identity.
    """

    def __init__(
        self,
        shard_dir: str = "/shared/amyxlu/data/cath/shards",
        pdb_path_dir: str = "/shared/amyxlu/data/cath/full/dompdb",
        seq_len: int = 64,
        batch_size: int = 32,
        num_workers: int = 0,
        path_to_dropped_ids: T.Optional[str] = None,
    ):
        super().__init__()
        self.shard_dir = shard_dir
        self.pdb_path_dir = pdb_path_dir
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_fn = CATHStructureDataset
        self.dtype = "fp32"
        self.path_to_dropped_ids = path_to_dropped_ids

    def setup(self, stage: str = "fit"):
        if stage == "fit":
            self.train_dataset = self.dataset_fn(
                "train",
                self.shard_dir,
                self.pdb_path_dir,
                self.seq_len,
                self.dtype,
                self.path_to_dropped_ids,
            )
            self.val_dataset = self.dataset_fn(
                "val",
                self.shard_dir,
                self.pdb_path_dir,
                self.seq_len,
                self.dtype,
                self.path_to_dropped_ids,
            )
        elif stage == "predict":
            self.test_dataset = self.dataset_fn(
                "val",
                self.shard_dir,
                self.pdb_path_dir,
                self.seq_len,
                self.dtype,
                self.path_to_dropped_ids,
            )
        else:
            raise ValueError(f"stage must be one of ['fit', 'predict'], got {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True
        )

    def predict_dataloader(self):
        return self.test_dataloader()


class FastaDataModule(L.LightningDataModule):
    def __init__(
        self,
        fasta_file: str,
        batch_size: int,
        max_seq_len: int = 512,
        train_frac: float = 0.8,
        num_workers: int = 0,
    ):
        super().__init__()
        self.fasta_file = fasta_file
        self.train_frac, self.val_frac = train_frac, 1 - train_frac
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_seq_len = max_seq_len

    def setup(self, stage: str = "fit"):
        ds = FastaDataset(self.fasta_file, cache_indices=True)
        seed = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            ds, [self.train_frac, self.val_frac], generator=seed
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )

    def test_dataloader(self):
        return self.val_dataloader()


class EmbedFastaDataModule(FastaDataModule):
    """Overrides the dataloader functions to embed with ESMFold instead."""

    def __init__(
        self,
        esmfold: torch.nn.Module,
        fasta_file: str,
        batch_size: int,
        max_seq_len: int = 512,
        train_frac: float = 0.8,
        num_workers: int = 0,
    ):
        super().__init__(fasta_file, batch_size, max_seq_len, train_frac, num_workers)
        self.esmfold = esmfold
        self.esmfold = self.esmfold.eval().requires_grad_(False).cuda()

    def embed_fn(self, header, sequence: List[str]) -> Tuple[torch.Tensor, List[str]]:
        from plaid.utils import get_random_sequence_crop_batch

        sequence = get_random_sequence_crop_batch(sequence, self.max_seq_len)
        with torch.no_grad():
            output = self.esmfold.infer_embedding(sequence)
        return output["s"], header, sequence

    def train_dataloader(self):
        """Dataloader will load the embedding and the original sequence (truncated to max length)"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            collate_fn=self.embed_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.embed_fn,
        )

    def test_dataloader(self):
        return self.val_dataloader()


if __name__ == "__main__":
    # from plaid.esmfold import esmfold_v1
    # datadir = "/homefs/home/lux70/data/cath"
    # pklfile = "/homefs/home/lux70/data/cath/sequences.pkl"
    # dm = CATHShardedDataModule(
    #     shard_dir=datadir,
    #     header_to_sequence_file=pklfile,
    # )
    # dm.setup("fit")
    # train_dataloader = dm.train_dataloader()
    # batch = next(iter(train_dataloader))
    # fasta_file = "/shared/amyxlu/data/uniref90/truncated.fasta"

    # esmfold = esmfold_v1()
    # esmfold = esmfold.eval().requires_grad_(False).cuda()

    # dm = EmbedFastaDataModule(esmfold, fasta_file, batch_size=32)
    # dm.setup("fit")
    # train_dataloader = dm.train_dataloader()
    # batch = next(iter(train_dataloader))
    # transform = esmfold.infer_embedding

    shard_dir = "/homefs/home/lux70/storage/data/cath/shards/"
    pdb_dir = "/data/bucket/lux70/data/cath/dompdb"
    path_to_dropped_ids = "/homefs/home/lux70/code/plaid/cath_ids_to_drop.txt"

    dm = CATHStructureDataModule(
        shard_dir,
        pdb_dir,
        seq_len=256,
        batch_size=32,
        num_workers=0,
        path_to_dropped_ids=path_to_dropped_ids,
    )
    dm.setup()
    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))
    print(len(train_dataloader.dataset))
