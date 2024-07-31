from pathlib import Path
import time
import wandb

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
from plaid.utils import (
    LatentScaler,
    pdb_path_to_biotite_atom_array,
    alpha_carbons_from_atom_array,
    get_model_device,
    write_pdb_to_disk,
    npy,
    to_tensor,
)
from plaid.proteins import LatentToStructure
from plaid.evaluation import run_tmalign
from plaid.evaluation import lDDT
from plaid.compression.hourglass_vq import HourglassVQLightningModule


def load_compression_model(model_id, ckpt_dir="/data/lux70/plaid/checkpoints/hourglass_vq"):
    dirpath = Path(ckpt_dir) / model_id
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
        batch_size,
        esmfold=None,
        shard_dir="/data/lux70/data/cath/shards/",
        pdb_dir="/data/bucket/lux70/data/cath/dompdb",
        out_dir="/homefs/home/lux70/cache/",
        num_samples: int = 32,
        max_seq_len: int = 256,
        num_recycles: int = 4,
        run_every_n_steps: int = 10000,
    ):
        self.latent_scaler = LatentScaler()
        self.structure_constructor = LatentToStructure(esmfold=esmfold) 

        self.batch_size = batch_size
        self.shard_dir = shard_dir
        self.pdb_dir = pdb_dir
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.num_recycles = num_recycles
        self.base_pdb_dir = Path(out_dir)
        self.run_every_n_steps = run_every_n_steps

        self.x, self.mask, self.sequences, self.gt_structures = self._get_validation_data()

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
            shuffle_val_dataset=False,
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

    def _compress_and_reconstruct(self, compression_model, max_samples=None):
        print("Running dataset through model bottleneck...")
        device = get_model_device(compression_model)
        quantize_scheme = compression_model.quantize_scheme

        x_norm = self.latent_scaler.scale(self.x).to(device)
        mask = self.mask.bool().to(device)

        if max_samples is not None:
            x_norm = x_norm[:max_samples, ...]
            mask = mask[:max_samples, ...]

        recons_norm, loss, log_dict, compressed_representation, downsampled_mask = compression_model(
            x_norm, mask, log_wandb=False
        )
        recons = self.latent_scaler.unscale(recons_norm)
        return recons, loss, log_dict, compressed_representation

    def _save_pdbs(self, struct_features, prefix=""):
        assert prefix in ["", "recons", "orig"]
        filenames = [str(self.base_pdb_dir / f"{prefix}_{i}.pdb") for i in range(len(struct_features))]
        for i in trange(
            len(struct_features),
            desc=f"Writing PDBs for {prefix} at {str(self.base_pdb_dir)}...",
        ):
            write_pdb_to_disk(struct_features[i], filenames[i])
        return filenames

    def _structure_features_from_latent(self, latent_recons, max_samples=None):
        if max_samples is None:
            max_samples = latent_recons.shape[0]
        else:
            assert max_samples <= latent_recons.shape[0]

        shared_args = {
            "return_raw_features": True,
            "batch_size": self.batch_size,
            "num_recycles": self.num_recycles,
        }
        recons_struct = self.structure_constructor.to_structure(
            latent_recons[:max_samples, ...], self.sequences[:max_samples], **shared_args
        )
        orig_pred_struct = self.structure_constructor.to_structure(
            self.x[:max_samples, ...], self.sequences[:max_samples], **shared_args
        )
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

    def validate(self, model, max_samples=None):
        # compress latent and reconstruct
        # assumes that model is already on the desired device
        start = time.time()
        torch.cuda.empty_cache()
        device = model.device
        self.structure_constructor.to(device)

        recons, loss, log_dict, compressed_representation = self._compress_and_reconstruct(
            model, max_samples=max_samples
        )
        compressed_representation = npy(compressed_representation)
        log_dict["compressed_rep_hist"] = wandb.Histogram(compressed_representation.flatten(), num_bins=50)

        # coerce latent back into structure features for both reconstruction and the original prediction
        # TODO: also compare to the ground truth structure?
        recons_struct, orig_pred_struct = self._structure_features_from_latent(
            recons, max_samples=max_samples
        )
        recons_pdb_paths = self._save_pdbs(recons_struct, "recons")
        orig_pdb_paths = self._save_pdbs(orig_pred_struct, "orig")

        # delete more tensors from GPU; rest of the operations happen on CPU.
        del recons_struct, orig_pred_struct

        # calculate the TM-scores with implicit alignment
        tm_scores_list = self._run_tmalign(orig_pdb_paths, recons_pdb_paths)
        n = len(tm_scores_list)
        log_dict["tmscore_mean"] = np.mean(tm_scores_list)
        log_dict["tmscore_median"] = np.median(tm_scores_list)
        log_dict["tmscore_hist"] = wandb.Histogram(tm_scores_list, num_bins=n)

        # pdb parse returns an atom stack; take only the first atom array
        orig_atom_arrays = [pdb_path_to_biotite_atom_array(fpath)[0] for fpath in orig_pdb_paths]
        recons_atom_arrays = [pdb_path_to_biotite_atom_array(fpath)[0] for fpath in recons_pdb_paths]
        recons_superimposed = [
            structure.superimpose(orig, recons)[0]
            for (orig, recons) in zip(orig_atom_arrays, recons_atom_arrays)
        ]

        # calculate superimposed RMSD
        superimposed_rmsd_scores = [
            structure.rmsd(orig, recons) for (orig, recons) in zip(orig_atom_arrays, recons_superimposed)
        ]
        log_dict["rmsd_mean"] = np.mean(superimposed_rmsd_scores)
        log_dict["rmsd_median"] = np.median(superimposed_rmsd_scores)
        log_dict["rmsd_hist"] = wandb.Histogram(superimposed_rmsd_scores, num_bins=n)

        # calculate lDDT from alpha carbons
        orig_ca_pos = [alpha_carbons_from_atom_array(aarr) for aarr in orig_atom_arrays]
        recons_ca_pos = [alpha_carbons_from_atom_array(aarr) for aarr in recons_superimposed]
        lddts = [
            lDDT(
                to_tensor(orig_ca_pos[i].coord),
                to_tensor(recons_ca_pos[i].coord),
            )
            for i in range(len(orig_ca_pos))
        ]
        log_dict["lddt_mean"] = np.mean(lddts)
        log_dict["lddt_median"] = np.median(lddts)
        log_dict["lddt_hist"] = wandb.Histogram(lddts, num_bins=n)

        # calculate RMSD between pairwise distances (superimposition independent)
        rmspd_scores = [
            structure.rmspd(orig, recons) for (orig, recons) in zip(orig_atom_arrays, recons_superimposed)
        ]
        log_dict["rmspd_mean"] = np.mean(rmspd_scores)
        log_dict["rmspd_median"] = np.median(rmspd_scores)
        log_dict["rmspd_hist"] = wandb.Histogram(rmspd_scores, num_bins=n)

        end = time.time()
        print(f"Structure reconstruction validation completed in {end - start:.2f} seconds.")
        return log_dict

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.global_step % self.run_every_n_steps == 0) and not (trainer.global_step == 0):
            device = pl_module.device
            max_samples = min(trainer.datamodule.batch_size, self.num_samples)

            # move the structure decoder onto GPU only when using validation
            self.structure_constructor.to(device)

            log_dict = self.validate(pl_module, max_samples=max_samples)
            log_dict = {f"structure_reconstruction/{k}": v for k, v in log_dict.items()}
            for k, v in log_dict.items():
                if "hist" in k:
                    pl_module.logger.experiment.log({k: v})  # cannot log histograms with pl_module.log
                else:
                    pl_module.log(k, v)

            # clear up GPU
            self.structure_constructor.to(torch.device("cpu"))
            torch.cuda.empty_cache()
        else:
            pass

    def on_sanity_check_start(self, trainer, pl_module):
        device = pl_module.device
        self.structure_constructor.to(device)

        _ = self.validate(pl_module, max_samples=2)

        # move strcuture decoder back to CPU to save some space
        self.structure_constructor.to(torch.device("cpu"))
        torch.cuda.empty_cache()


def main():
    """For use as a standalone script"""
    import sys

    model_id = sys.argv[1]

    model = load_compression_model(model_id)
    device = torch.device("cuda")
    model.to(device)

    callback = CompressionReconstructionCallback(batch_size=4)
    log_dict = callback.validate(model)
    print(log_dict)
