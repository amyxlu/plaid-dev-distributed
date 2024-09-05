from typing import Optional
import pandas as pd

import torch
from cheap.esmfold import esmfold_v1

from ..typed import PathLike


class Fold:
    def __init__(
        self,
        fasta_file: PathLike,
        num_recycles: int = 4,
        batch_size: int = 512,
        max_seq_len: int = 512,
        esmfold: Optional[torch.nn.Module] = None,
    ):
        self.fasta_file = fasta_file
        self.seq_df = self.load_sequences()
        self.num_recycles = num_recycles
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        if esmfold is None:
            esmfold = esmfold_v1() 
        self.esmfold = esmfold

    def load_sequences(self):
        ret = {
            "headers": [],
            "sequences": []
        }

        with open(self.fasta_file, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    ret["headers"].append(line.strip())
                else:
                    ret["sequences"].append(line.strip())
        return pd.DataFrame(ret)

    def fold_batch(self, sequences):
        self.esmfold.infer_pdbs(sequences)
        pass

    def run(self):
        for start_idx in range(0, len(self.sequences), self.batch_size):
            end_idx = start_idx + self.batch_size
            batch = self.seq_df.iloc[start_idx:end_idx, :]
            pdb_strs = self.fold_batch(batch['sequences'].values)

