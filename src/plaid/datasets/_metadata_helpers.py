import pandas as pd


NUM_FUNCTION_CLASSES = 2219
NUM_ORGANISM_CLASSES = 3617


class MetadataParser:
    def __init__(
        self,
        go_metadata_fpath = "/data/lux70/data/pfam/pfam2go.csv",
        organism_metadata_fpath = "/data/lux70/data/pfam/organism_counts.csv"
    ):
        """
        Class to extract relevant information from Pfam headers using specific data structures.

        go_metadata is created with notebooks/DATA_pfam_to_GO.ipynb
        organism indexing is created with notebooks/DATA_organism_counts.ipynb
        """
        self.go_metadata = pd.read_csv(go_metadata_fpath)
        self.organism_metadata = pd.read_csv(organism_metadata_fpath)

        assert self.go_metadata['GO_idx'].max() + 1 == NUM_FUNCTION_CLASSES, f"GO_idx max is {self.go_metadata['GO_idx'].max()}, expected {NUM_FUNCTION_CLASSES}"
        assert self.organism_metadata['organism_index'].max() + 1 == NUM_ORGANISM_CLASSES, f"organism_index max is {self.organism_metadata['organism_index'].max()}, expected {NUM_ORGANISM_CLASSES}"

        self.dummy_go_idx = self.go_metadata['GO_idx'].max() + 1
        self.dummy_organism_idx = self.organism_metadata['organism_index'].max() + 1
    
    def header_to_pfam_id(self, header: str):
        return header.split(" ")[-1].split(".")[0]

    def header_to_organism(self, header: str):
        return header.split("/")[0].split("_")[-1]

    def header_to_go_idx(self, header: str):
        pfam_id = self.header_to_pfam_id(header)
        try:
            return self.go_metadata[self.go_metadata['pfam_id'] == pfam_id]['GO_idx'].item()
        except ValueError:
            return self.dummy_go_idx 

    def header_to_organism_idx(self, header: str):
        organism = self.header_to_organism(header)
        try:
            return self.organism_metadata[self.organism_metadata.organism_id == organism]['organism_index'].item()
        except ValueError:
            return self.dummy_organism_idx


