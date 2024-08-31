from pathlib import Path

import torch
import numpy as np
import pandas as pd

from cheap.proteins import LatentToSequence, LatentToStructure
from cheap.pretrained import CHEAP_pfam_shorten_2_dim_32
from plaid.utils import outputs_to_avg_metric, npy, write_pdb_to_disk, write_to_fasta
from plaid.typed import PathLike


class DecodeLatent:
    """
    Given a NPZ of sampled latent, reconstruct the sequence
    """
    def __init__(
        self,
        outdir: PathLike = None,
        num_recycles=4,
        batch_size=64,
        device="cuda"
    ):
        self.num_recycles = num_recycles
        self.batch_size = batch_size
        self.device = device
        self.outdir = outdir

        self.sequence_constructor = None
        self.structure_constructor = None
        self.cheap_pipeline = CHEAP_pfam_shorten_2_dim_32()
        self.hourglass = self.cheap_pipeline.hourglass_model
    
    def load_sampled(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        x = torch.tensor(data["samples"])
        return x
    
    def process_x(self, x, mask=None):
        return self.hourglass.decode(x, mask)

    def construct_sequence(
        self,
        x_processed,
    ):
        device = self.device
        if self.sequence_constructor is None:
            self.sequence_constructor = LatentToSequence()

        # forward pass
        self.sequence_constructor.to(device)
        x_processed = x_processed.to(device)
        with torch.no_grad():
            _, _, sequences = self.sequence_constructor.to_sequence(x_processed, return_logits=False)
        
        return sequences

    def construct_structure(self, x_processed, seq_str):
        device = self.device
        if self.structure_constructor is None:
            # warning: this implicitly creates an ESMFold inference model, can be very memory consuming
            self.structure_constructor = LatentToStructure()

        self.structure_constructor.to(device)
        x_processed = x_processed.to(device=device)

        with torch.no_grad():
            pdb_strs, outputs = self.structure_constructor.to_structure(
                x_processed,
                sequences=seq_str,
                num_recycles=self.num_recycles,
                batch_size=self.batch_size,
                return_raw_outputs=True
            )

        metrics = outputs_to_avg_metric(outputs)

        # `outputs` values are numpy arrays on CPU
        plddt_per_position = npy(outputs["plddt"].flatten())
        pae_per_position = npy(outputs["predicted_aligned_error"].flatten())

        plddt = npy(metrics["plddt"].flatten())
        pae = npy(metrics["predicted_aligned_error"].flatten())
        ptm = npy(metrics["ptm"].flatten())

        N = x_processed.shape[0]

        log_dict = {
            f"sampled/plddt_mean": np.mean(plddt),
            f"sampled/plddt_std": np.std(plddt),
            f"sampled/plddt_hist": np.histogram(plddt, bins=min(N, 50)),
            f"sampled/plddt_per_position_hist": np.histogram(plddt_per_position, bins=min(N, 50)),
            f"sampled/pae_mean": np.mean(pae),
            f"sampled/pae_std": np.std(pae),
            f"sampled/pae_hist": np.histogram(pae, bins=min(pae.shape[0], 50)),
            f"sampled/pae_per_position_hist": np.histogram(pae_per_position, bins=min(N, 50)),
            f"sampled/ptm_mean": np.mean(pae),
            f"sampled/ptm_std": np.std(pae),
            f"sampled/ptm_hist": np.histogram(ptm, bins=min(ptm.shape[0], 50)),
        }
        return pdb_strs, metrics, log_dict
    
    def run(self, x):
        x_processed = self.process_x(x)
        seq_str = self.construct_sequence(x_processed)
        pdb_strs, metrics, log_dict = self.construct_structure(x_processed, seq_str)
        return pdb_strs, metrics, log_dict

    def write_structures_to_disk(self, pdb_strs):
        assert not self.outdir is None
        paths = []
        for i, pdbstr in enumerate(pdb_strs):
            outpath = Path(self.outdir) / f"sample{i}.pdb"
            outpath = write_pdb_to_disk(pdbstr, outpath)
            paths.append(outpath)
        return paths

    def write_sequences_to_disk(self):
        assert not self.outdir is None
        write_to_fasta(self.sequences, Path(self.outdir) / "generated.fasta")

from plaid.datasets import NUM_FUNCTION_CLASSES, NUM_ORGANISM_CLASSES

def run_decode_latent(
        diffusion_model_id: str = "",
        function_idx: int = NUM_ORGANISM_CLASSES,
        organism_idx: int = NUM_FUNCTION_CLASSES,
        latent_timestamp: str = "",
    ):
        decoder = DecodeLatent(outdir=outdir)
        pdb_strs, metrics, log_dict = decoder.run(x)
        paths = decoder.write_structures_to_disk(pdb_strs)
        return paths, metrics, log_dict