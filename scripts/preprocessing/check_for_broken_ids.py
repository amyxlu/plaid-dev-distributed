from plaid.datasets import CATHStructureDataModule
from tqdm import tqdm


shard_dir = "/data/lux70/data/cath/shards/"
pdb_dir = "/data/bucket/lux70/data/cath/dompdb"
# path_to_dropped_ids = "/homefs/home/lux70/code/plaid/plaid/cath_dropped_pdb_list.txt"

"""
Using a try-except block in the datamodule as:

class CATHStructureDataset(H5ShardDataset):
    ...
    
    def __getitem__(self, idx: int):

        
    def __getitem__(self, idx: int):
        pdb_id, (emb, seq) = self.get(idx)
        pdb_path = self.pdb_path_dir / pdb_id
        with open(pdb_path, "r") as f:
            pdb_str = f.read()
        try:
            structure_features = self.structure_featurizer(pdb_str, self.max_seq_len)
            return emb, seq, structure_features 
        except:
            with open("bad_ids.txt", "a") as f:
                f.write(f"{pdb_id}\n")
            return emb, seq, {}
"""
CHECK_PHASE = False

if CHECK_PHASE:
    dm = CATHStructureDataModule(
        shard_dir, pdb_dir, seq_len=256, batch_size=512, num_workers=0, path_to_dropped_ids=None
    )
    dm.setup()
    train_dataset = dm.train_dataloader().dataset
    val_dataset = dm.val_dataloader().dataset

    for i in range(len(train_dataset)):
        _ = train_dataset[i]

    for i in range(len(val_dataset)):
        _ = val_dataset

else:
    dm = CATHStructureDataModule(
        shard_dir,
        pdb_dir,
        seq_len=256,
        batch_size=512,
        num_workers=0,
        path_to_dropped_ids=None,  # "bad_ids.txt"
    )
    dm.setup()
    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()

    for batch in tqdm(train_dataloader):
        pass
    print("done training dataloader loop")

    for batch in tqdm(val_dataloader):
        pass
    print("done val dataloader loop")
