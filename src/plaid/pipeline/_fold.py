from typing import Optional, List, Tuple

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from evo.dataset import FastaDataset

from ..esmfold import output_to_pdb, esmfold_v1
from ..typed import PathLike, DeviceLike
from ..utils import save_pdb_strs_to_disk
from ..transforms import get_random_sequence_crop_batch


class FoldPipeline:
    def __init__(
        self,
        fasta_file: PathLike,
        outdir: Optional[PathLike],
        esmfold: Optional[torch.nn.Module] = None,
        num_recycles: int = 4,
        batch_size: int = -1,
        max_seq_len: Optional[int] = 512,
        save_as_single_pdb_file: bool = False,
        device: DeviceLike = torch.device("cuda"),
        max_num_batches: Optional[int] = None,
        shuffle: bool = False
    ):
        self.fasta_file = fasta_file
        self.outdir = outdir
        self.num_recycles = num_recycles
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.device = device
        self.max_num_batches = max_num_batches
        self.save_as_single_pdb_file = save_as_single_pdb_file

        self.ds = FastaDataset(fasta_file)
        if batch_size == -1:
            batch_size = len(self.ds)
        self.dataloader = torch.utils.data.DataLoader(
            self.ds, batch_size=batch_size, shuffle=shuffle, num_workers=4
        )

        if esmfold is None:
            esmfold = esmfold_v1() 
        self.esmfold = esmfold
        self.esmfold.to(self.device)
    
    def to(self, device):
        self.device = device
        self.esmfold.to(device)
        return self

    def fold_batch(self, sequences) -> Tuple[List[str], List[float]]:
        # if not self.max_seq_len is None:
        #     sequences = get_random_sequence_crop_batch(sequences, self.max_seq_len, min_len=30)
        if not self.max_seq_len is None:
            sequences = [seq[:self.max_seq_len] for seq in sequences]
        
        with torch.no_grad():
            # output = self.esmfold.infer(sequences)
            # plddts = output['plddt'].cpu().numpy()
            # assert plddts.ndim == 3
            return self.esmfold.infer_pdbs(sequences, num_recycles=self.num_recycles)

    def write(self, pdb_strs, headers):
        if self.outdir is None:
            print("Skipping PDB writing as _save_mode is None.")
        else:
            save_pdb_strs_to_disk(pdb_strs, self.outdir, headers, self.save_as_single_pdb_file)

    def run(self):
        if self.outdir is not None:
            print("Warning: make sure there are no repeated headers in the FASTA file.\n"
                  "Otherwise, the resulting structures will be overwritten.")

        all_headers = []
        all_pdb_strs = []

        num_batches = min(len(self.ds) // self.batch_size, self.max_num_batches)

        for i, batch in tqdm(enumerate(self.dataloader), total=num_batches):
            if self.max_num_batches is not None and i >= self.max_num_batches:
                print(f"Stopping at {self.max_num_batches} batches.")
                break

            headers, sequences = batch
            pdb_strs = self.fold_batch(sequences)
            all_headers.extend(headers)
            all_pdb_strs.extend(pdb_strs)

            # write samples for this batch: 
            headers = [f"pfam{i}" for i in range(len(pdb_strs))]
            self.write(pdb_strs, headers)
        
        return all_pdb_strs
