import torch
from torch.utils.data import IterableDataset, DataLoader
from safetensors.torch import load_file
import glob
from pathlib import Path


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
            shuffled_idxs = torch.randperm(embs.size(0))
            for i in shuffled_idxs:
                yield embs[i, ...], seqlens[i, ...]

