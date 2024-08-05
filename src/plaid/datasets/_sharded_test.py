

if __name__ == "__main__":
    from plaid.datasets import FunctionOrganismDataModule

    datamodule = FunctionOrganismDataModule(
        train_shards="/data/lux70/data/pfam/compressed/j1v1wv6w/train/shard{000..160}.tar",
        val_shards="/data/lux70/data/pfam/compressed/j1v1wv6w/val/shard{000..022}.tar",
        config_file="/data/lux70/data/pfam/compressed/j1v1wv6w/config.json",
        go_metadata_fpath="/data/lux70/data/pfam/pfam2go.csv",
        organism_metadata_fpath="/data/lux70/data/pfam/organism_counts.csv",
        cache_dir="/homefs/home/lux70/cache/plaid_data_cache/j1v1wv6w",
        train_epoch_num_batches=50_000,
        val_epoch_num_batches=1_000,
        shuffle_buffer=10_000,
        shuffle_initial=10_000,
        num_workers=2,
    )
    datamodule.setup()

    train_dataloader = datamodule.train_dataloader()
    train_dataset = datamodule.train_ds

    for i, batch in enumerate(train_dataloader):
        if i > 20: 
            break
        print(batch[2:5])
        print("\n\n")
        
    val_dataloader = datamodule.val_dataloader()
    for i, batch in enumerate(val_dataloader):
        if i > 20: 
            break
        print(batch[-2:])
        print("\n\n")
        