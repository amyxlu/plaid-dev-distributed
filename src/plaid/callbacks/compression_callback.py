from pathlib import Path
import time

from tqdm import tqdm, trange
import numpy as np
from biotite import structure
import torch
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only

from plaid.esmfold.misc import batch_encode_sequences
from plaid.datasets import CATHStructureDataModule
from plaid.transforms import trim_or_pad_batch_first
from plaid.utils import LatentScaler, pdb_path_to_biotite_atom_array, alpha_carbons_from_atom_array 
from plaid.proteins import LatentToStructure
from plaid.evaluation import run_tmalign
from plaid.evaluation import lDDT
from plaid.compression.hourglass_vq import HourglassVQLightningModule


def load_compression_model(model_id):
    dirpath = Path(f"/homefs/home/lux70/storage/plaid/checkpoints/hourglass_vq/{model_id}")
    return HourglassVQLightningModule.load_from_checkpoint(dirpath / "last.ckpt")


def maybe_print(msg):
    if rank_zero_only.rank == 0:
        print(msg)


class CompressionReconstructionCallback(Callback):
    """
    For compression experiments, evaluate the reconstruction quality.
    """
    def __init__(
            self,
            compression_model,
            device,
            batch_size,
            esmfold=None,
            shard_dir = "/homefs/home/lux70/storage/data/cath/shards/",
            pdb_dir = "/data/bucket/lux70/data/cath/dompdb",
            num_samples: int = 32,
            max_seq_len: int = 256,
            num_recycles: int = 4
        ):
        self.device = device
        self.compression_model = compression_model

        self.latent_scaler = LatentScaler()
        self.structure_constructor = LatentToStructure(esmfold=esmfold)
        self.structure_constructor.to(device)
        self.compression_model.to(device)

        self.batch_size = batch_size
        self.shard_dir = shard_dir
        self.pdb_dir = pdb_dir
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.num_recycles = num_recycles
        self.base_pdb_dir = Path("/homefs/home/lux70/cache/")

        self.quantize_scheme = self.compression_model.quantize_scheme
        x, mask, sequences, gt_structures = self._get_validation_data() 

        self.x = x
        self.mask = mask
        self.sequences = sequences
        self.gt_structures = gt_structures
        import IPython;IPython.embed()

    def _get_validation_data(self):
        start = time.time()
        print(f"Creating reference validation data of {self.num_samples} points...")

        # only preprocess num_samples data points and load all in one batch
        dm = CATHStructureDataModule(
            self.shard_dir,
            self.pdb_dir,
            seq_len=self.max_seq_len,
            batch_size=self.num_samples,
            max_num_samples=self.num_samples,
            shuffle_val_dataset=False
        ) 
        dm.setup()
        val_dataloader = dm.val_dataloader()
        batch = next(iter(val_dataloader))

        x = batch[0]
        sequences = batch[1]
        gt_structures = batch[-1]

        # make mask
        _, mask, _, _, _ = batch_encode_sequences(sequences)
        mask = trim_or_pad_batch_first(mask, pad_to=self.max_seq_len, pad_idx=0)
        
        end = time.time()
        print(f"Created reference structure validation dataset in {end - start:.2f} seconds.")
        return x, mask, sequences, gt_structures

    def _compress_and_reconstruct(self):
        print("Running dataset through model bottleneck...")
        x_norm = self.latent_scaler.scale(self.x).to(self.device)
        mask = self.mask.bool().to(self.device)
        recons_norm, loss, log_dict, quant_out = self.compression_model(x_norm, mask, log_wandb=False)
        recons = self.latent_scaler.unscale(recons_norm)

        if self.quantize_scheme == "vq":
            N, L, _ = x_norm.shape
            print(quant_out['min_encoding_indices'].shape)
            print(quant_out['min_encoding_indices'].reshape(N, -1).shape)
            print(quant_out['min_encoding_indices'].reshape(N, L, -1).shape)
            compressed_representation = quant_out['min_encoding_indices'].reshape(N, L, -1)

        elif self.quantize_scheme == "fsq":
            codebook = quant_out['codebook']
            print(codebook.shape)
            print(codebook.max())
            compressed_representation = codebook.reshape(-1, self.quantizer.num_dimensions)

        else:
            # no quantization, quant_out is the output of the encoder
            compressed_representation = quant_out
        
        # TODO: analysis with the latents
        return recons, loss, log_dict, compressed_representation 
    
    def _save_pdbs(self, struct_features, prefix=""):
        assert prefix in ["", "recons", "orig"]
        filenames = [str(self.base_pdb_dir / f"{prefix}_{i}.pdb") for i in range(len(struct_features))]
        for i in trange(len(struct_features), desc=f"Writing PDBs for {prefix} at {str(self.base_pdb_dir)}..."):
            with open(filenames[i], "w") as f:
                f.write(struct_features[i])
        return filenames
    
    def _structure_features_from_latent(self, latent_recons):
        shared_args = {
            "return_raw_features": True,
            "batch_size": self.batch_size,
            "num_recycles": self.num_recycles
        }
        recons_struct = self.structure_constructor.to_structure(latent_recons, self.sequences, **shared_args)
        orig_pred_struct = self.structure_constructor.to_structure(self.x, self.sequences, **shared_args)
        # only the first of the tuple is the structure feature 
        return recons_struct[0], orig_pred_struct[0]

    def _run_tmalign(self, orig_pdbs, recons_pdbs):
        all_scores = []
        for orig, recons in zip(orig_pdbs, recons_pdbs):
            tmscore = run_tmalign(orig, recons)
            all_scores.append(tmscore)
        print("mean:", np.mean(all_scores))
        print("median:", np.median(all_scores)) 
        return all_scores

    def validate(self):
        # compress latent and reconstruct
        recons, loss, log_dict, compressed_representation = self._compress_and_reconstruct() 

        # coerce latent back into structure features for both reconstruction and the original prediction 
        # TODO: also compare to the ground truth structure? 
        recons_struct, orig_pred_struct = self._structure_features_from_latent(recons)
        recons_pdb_paths = self._save_pdbs(recons_struct, "recons")
        orig_pdb_paths = self._save_pdbs(orig_pred_struct, "orig")

        # calculate the TM-scores with implicity alignment
        tm_scores_list = self._run_tmalign(orig_pdb_paths, recons_pdb_paths)
        log_dict['tm_score_mean'] = np.mean(tm_scores_list)
        log_dict['tm_score_median'] = np.median(tm_scores_list)

        # pdb parse returns an atom stack; take only the first atom array
        orig_atom_arrays = [pdb_path_to_biotite_atom_array(fpath)[0] for fpath in orig_pdb_paths]
        recons_atom_arrays = [pdb_path_to_biotite_atom_array(fpath)[0] for fpath in recons_pdb_paths]
        recons_superimposed = [structure.superimpose(orig, recons)[0] for (orig, recons) in zip(orig_atom_arrays, recons_atom_arrays)]

        # calculate superimposed RMSD
        superimposed_rmsd_scores = [structure.rmsd(orig, recons) for (orig, recons) in zip(orig_atom_arrays, recons_superimposed)]
        log_dict['rmsd_mean'] = np.mean(superimposed_rmsd_scores)
        log_dict['rmsd_median'] = np.median(superimposed_rmsd_scores)

        # calculate lDDT from alpha carbons
        orig_ca_pos = [alpha_carbons_from_atom_array(aarr) for aarr in orig_atom_arrays]
        recons_ca_pos  = [alpha_carbons_from_atom_array(aarr) for aarr in recons_superimposed]
        lddts = [lDDT(torch.from_numpy(orig_ca_pos[i].coord), torch.from_numpy(recons_ca_pos[i].coord)) for i in range(len(orig_ca_pos))]
        log_dict['lddt_mean'] = np.mean(lddts)
        log_dict['lddt_median'] = np.median(lddts)

        # calculate RMSD between pairwise distances (superimposition independent)
        rmspd_scores = [structure.rmspd(orig, recons) for (orig, recons) in zip(orig_atom_arrays, recons_superimposed)]
        log_dict['rmspd_mean'] = np.mean(rmspd_scores)
        log_dict['rmspd_median'] = np.median(rmspd_scores)

    

if __name__ == "__main__":
    model = load_compression_model("2024-03-17T23-21-19")
    device = torch.device("cuda")
    callback = CompressionReconstructionCallback(model, device, batch_size=4)