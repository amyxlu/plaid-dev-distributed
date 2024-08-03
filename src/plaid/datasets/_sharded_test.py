# import unittest

# class TestMetadataHelpers(unittest.TestCase):
#     def setUp(self):
#         self.metadata_parser = MetadataParser(
#             go_metadata_fpath = "/data/lux70/data/pfam/pfam2go.csv",
#             organism_metadata_fpath = "/data/lux70/data/pfam/organism_counts.csv"
#         ) 

#     def test_go_term_parsing():
#         sample_headers = [
#             'A0A418XQZ6_9BURK/10-72 A0A418XQZ6.1 PF00392.25;GntR;',
#             'A0A1E7JVW7_9ACTN/15-77 A0A1E7JVW7.1 PF00392.25;GntR;',
#             'A0A4Y9SLN2_9BURK/19-82 A0A4Y9SLN2.1 PF00392.25;GntR;',
#         ]


if __name__ == "__main__":
    from plaid.datasets import FunctionOrganismDataModule

    datamodule = FunctionOrganismDataModule(
        train_shards="/data/lux70/data/pfam/compressed/jzlv54wl/shard{00000..00014}.tar",
        val_shards="/data/lux70/data/pfam/compressed/jzlv54wl/shard{00015..00019}.tar",
        go_metadata_fpath="/data/lux70/data/pfam/pfam2go.csv",
        organism_metadata_fpath="/data/lux70/data/pfam/organism_counts.csv",
        cache_dir="/data/lux70/data/pfam/compressed/jzlv54wl/cache/"
    )
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    train_dataset = datamodule.train_ds


    for i, batch in enumerate(train_dataloader):
        if i > 5: 
            break
        print(batch)
        print("\n\n")
        
    import IPython;IPython.embed()
    