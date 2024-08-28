

if __name__ == "__main__":
    from plaid.datasets import FunctionOrganismDataModule
    import torch

    datamodule = FunctionOrganismDataModule(
        train_shards="/data/lux70/data/pfam/compressed/j1v1wv6w/train/shard{0000..4423}.tar",
        val_shards="/data/lux70/data/pfam/compressed/j1v1wv6w/val/shard{0000..0863}.tar",
        config_file="/data/lux70/data/pfam/compressed/j1v1wv6w/config.json",
        go_metadata_fpath="/data/lux70/data/pfam/pfam2go.csv",
        organism_metadata_fpath="/data/lux70/data/pfam/organism_counts.csv",
        cache_dir="/homefs/home/lux70/cache/plaid_data_cache/j1v1wv6w",
        train_epoch_num_batches=20,
        val_epoch_num_batches=20,
        shuffle_buffer=20,
        shuffle_initial=20,
        num_workers=4,
    )
    datamodule.setup()

    train_dataloader = datamodule.train_dataloader()
    train_dataset = datamodule.train_ds
    val_dataloader = datamodule.val_dataloader()

    for i, batch in enumerate(train_dataloader):
        if i > 20: 
            break
        print(i)
        embedding, mask, go_idx, organism_idx, pfam_id, sample_id, local_path = batch
        print(batch[2:5])
        assert embedding.shape == torch.Size([64, 512, 32])
        assert mask.shape == torch.Size([64, 512])
        assert go_idx.shape == torch.Size([64])
        assert organism_idx.shape == torch.Size([64])
        print("\n\n")
        
    print("val")
    
    for i, batch in enumerate(val_dataloader):
        if i > 20: 
            break
        print(batch[-2:])
        print("\n\n")

    print("Train")
        
    for i, batch in enumerate(train_dataloader):
        if i > 20: 
            break
        print(i)
        embedding, mask, go_idx, organism_idx, pfam_id, sample_id, local_path = batch
        print(batch[2:5])
        assert embedding.shape == torch.Size([64, 512, 32])
        assert mask.shape == torch.Size([64, 512])
        assert go_idx.shape == torch.Size([64])
        assert organism_idx.shape == torch.Size([64])
        print("\n\n")
    
    import IPython;IPython.embed()