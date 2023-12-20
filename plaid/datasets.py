import torch
from torch.utils.data import IterableDataset, DataLoader
from safetensors.torch import load_file
import glob
from pathlib import Path

from plaid.transforms import mask_from_seq_lens


class ShardedTensorDataset(IterableDataset):
    def __init__(self, shard_dir, split=None):
        shard_dir = Path(shard_dir)
        shard_files = glob.glob(str(shard_dir / "*.pt"))
        num_shards = len(shard_files)
        assert num_shards > 0, f"no shards found in {shard_dir}"
    
        if split is not None:
            if split == "train":
                shard_files = shard_files[:int(0.8 * num_shards)]  # 5 shards for CATH
            elif split == "val":
                shard_files = shard_files[int(0.8 * num_shards):]  # 1 shard for CATH
            else:
                raise ValueError(f"split must be one of ['train', 'val'], got {split}")
        self.shard_files = shard_files

    def __iter__(self):
        for file in self.shard_files:
            tensor_dict = load_file(file)
            assert set(tensor_dict.keys()) == set(("embeddings", "seq_len")) 
            embs, seqlens = tensor_dict["embeddings"], tensor_dict["seq_len"]
            mask = mask_from_seq_lens(embs, seqlens)
            shuffled_idxs = torch.randperm(embs.size(0))
            for i in shuffled_idxs:
                yield embs[i, ...], mask[i, ...]


import lightning as L

class CATHShardedDataModule(L.LightningDataModule):
    def __init__(
        self,
        shard_dir: str = "/shared/amyxlu/data/cath/shards/full",
        seq_len: int = 64,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.shard_dir = Path(shard_dir) / f"seqlen_{seq_len}"
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage: str = "fit"):
        if stage == "fit":
            self.train_dataset = ShardedTensorDataset(self.shard_dir, split="train")
            self.val_dataset = ShardedTensorDataset(self.shard_dir, split="val")
        elif stage == "predict":
            self.test_dataset = ShardedTensorDataset(self.shard_dir, split="val")
        else:
            raise ValueError(f"stage must be one of ['fit', 'predict'], got {stage}")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    
    def predict_dataloader(self):
        return self.test_dataloader()
    

if __name__ == "__main__":
    dm = CATHShardedDataModule()
    dm.setup("fit")
    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))
    import IPython; IPython.embed()