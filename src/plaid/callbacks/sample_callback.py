"""
Custom callbacks sampling and evaluation.
"""

import typing as T
import os
import json
from pathlib import Path

import wandb
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only
import safetensors.torch as st
import pandas as pd
import numpy as np
from tqdm import tqdm, trange

from plaid.denoisers import BaseDenoiser
from plaid.diffusion import GaussianDiffusion
from plaid.evaluation import RITAPerplexity, parmar_fid, parmar_kid 
from plaid.proteins import LatentToSequence, LatentToStructure
from plaid.utils import LatentScaler, write_pdb_to_disk
from plaid.constants import CACHED_TENSORS_DIR


def maybe_print(msg):
    if rank_zero_only.rank == 0:
        print(msg)


class SampleCallback(Callback):
    def __init__(
        self,
        diffusion: GaussianDiffusion,  # most sampling logic is here
        model: BaseDenoiser,
        n_to_sample: int = 16,
        n_to_construct: int = 4,
        gen_seq_len: int = 64,
        batch_size: int = -1,
        log_to_wandb: bool = False,
        calc_structure: bool = True,
        calc_sequence: bool = True,
        calc_fid: bool = True,
        latent_scaler: T.Optional[LatentScaler] = LatentScaler(),
        fid_holdout_tensor_fpath: str = "",
        calc_perplexity: bool = True,
        save_generated_structures: bool = False,
        num_recycles: int = 4,
        outdir: str = "sampled",
        sequence_decode_temperature: float = 1.0,
        sequence_constructor: T.Optional[LatentToSequence] = None,
        structure_constructor: T.Optional[LatentToStructure] = None,
        run_every_n_epoch: int = 1
    ):
        super().__init__()
        self.outdir = Path(outdir)
        self.diffusion = diffusion
        self.model = model
        self.log_to_wandb = log_to_wandb

        self.calc_fid = calc_fid
        self.calc_structure = calc_structure
        self.calc_sequence = calc_sequence
        self.calc_perplexity = calc_perplexity
        self.save_generated_structures = save_generated_structures
        if calc_perplexity:
            assert calc_sequence
        if save_generated_structures:
            assert calc_structure

        self.fid_holdout_tensor_fpath = fid_holdout_tensor_fpath
        self.n_to_sample = n_to_sample
        self.num_recycles = num_recycles
        self.sequence_decode_temperature = sequence_decode_temperature

        self.is_save_setup = False
        self.is_fid_setup = False
        self.is_perplexity_setup = False
        self.sequence_constructor = sequence_constructor
        self.structure_constructor = structure_constructor

        if latent_scaler is None:
            self.latent_scaler = LatentScaler("identity")
        else:
            self.latent_scaler = latent_scaler

        batch_size = self.n_to_sample if batch_size == -1 else batch_size
        n_to_construct = self.n_to_sample if n_to_construct == -1 else n_to_construct
        self.batch_size = batch_size
        self.n_to_construct = n_to_construct
        self.gen_seq_len = gen_seq_len
        self.run_every_n_epoch = run_every_n_epoch

    def _save_setup(self):
        if not self.outdir.exists():
            self.outdir.mkdir(parents=True)
            print("Created directory", self.outdir)
        self.is_save_setup = True

    def _perplexity_setup(self, device):
        self.perplexity_calc = RITAPerplexity(device)
        self.is_perplexity_setup = True

    def _fid_setup(self, device):

        # safetensors version - deprecated
        # def load_saved_features(location, device="cpu"):
        #     return st.load_file(location)["features"].to(device)

        def load_saved_features(location, device="cpu"):
            import h5py
            with h5py.File(location, "r") as f:
                tensor = f['embeddings'][()]
            tensor = torch.from_numpy(tensor).to(device)
            return tensor.mean(dim=1)

        self.real_features = load_saved_features(self.fid_holdout_tensor_fpath, device=device)
        self.real_features = self.real_features[
            torch.randperm(self.real_features.size(0))[: self.n_to_sample]
        ]

        # calculate FID/KID in the normalized space
        self.real_features = self.latent_scaler.scale(self.real_features)

    def sample_latent(self, shape):
        all_samples, n_samples = [], 0
        for _ in trange(0, self.n_to_sample, shape[0]):
        # while n_samples < self.n_to_sample:
            # TODO: add a pbar
            # make sure to not unscale s.t. we can calculate KID/FID in the normalized space!
            sample = self.diffusion.sample(shape, clip_denoised=True, unscale=False)
            all_samples.append(sample.detach().cpu())
            n_samples += sample.shape[0]
        x_0 = torch.cat(all_samples, dim=0)
        log_dict = {
            "sampled/latent_mean": x_0.mean(),
            "sampled/latent_std": x_0.std(),
        }
        return x_0, log_dict

    def calculate_fid(self, sampled_latent, device):
        if not self.is_fid_setup:
            self._fid_setup(device)
        fake_features = sampled_latent.mean(dim=1)

        # just to be consistent, but not necessary
        indices = torch.randperm(self.real_features.size(0))[: fake_features.shape[0]]
        real_features = self.real_features[indices]
        assert real_features.ndim == fake_features.ndim == 2

        fake_features = fake_features.cpu().numpy()
        real_features = real_features.cpu().numpy()

        fid = parmar_fid(fake_features, real_features)
        kid = parmar_kid(fake_features, real_features)

        log_dict = {f"sampled/fid": fid, f"sampled/kid": kid}
        return log_dict

    def construct_sequence(
        self,
        x_0,
        device,
    ):
        if self.sequence_constructor is None:
            self.sequence_constructor = LatentToSequence()

        # forward pass
        self.sequence_constructor.to(device)
        x_0 = x_0.to(device=device)
        with torch.no_grad():
            probs, idxs, strs = self.sequence_constructor.to_sequence(
                x_0, return_logits=False
            )

        # organize results for logging
        sequence_results = pd.DataFrame(
            {
                "sequences": strs,
                "mean_residue_confidence": probs.mean(dim=1).cpu().numpy(),
            }
        )
        log_dict = {f"sampled/sequences": wandb.Table(dataframe=sequence_results)}

        if self.calc_perplexity:
            if not self.is_perplexity_setup:
                self._perplexity_setup(device)
            perplexities = self.perplexity_calc.batch_eval(strs, return_mean=False)
            print(f"Mean perplexity: {np.mean(perplexities):.3f}")
            log_dict[f"sampled/perplexity_mean"] = np.mean(perplexities)
            log_dict[f"sampled/perplexity_hist"] = wandb.Histogram(
                np.array(perplexities).flatten(),
                num_bins=max(len(perplexities), 50)
            )

        return strs, log_dict

    def construct_structure(self, x_0, seq_str, device):
        if self.structure_constructor is None:
            # warning: this implicitly creates an ESMFold inference model, can be very memory consuming
            self.structure_constructor = LatentToStructure()

        self.structure_constructor.to(device)
        x_0 = x_0.to(device=device)

        with torch.no_grad():
            pdb_strs, outputs = self.structure_constructor.to_structure(
                x_0,
                sequences=seq_str,
                num_recycles=self.num_recycles,
                batch_size=self.batch_size,
            )
            from plaid.utils import outputs_to_avg_metric
            metrics = outputs_to_avg_metric(outputs)

        plddt = metrics['plddt'].cpu().numpy().flatten(),
        log_dict = {
            f"sampled/plddt_mean": np.mean(plddt),
            f"sampled/plddt_std": np.std(plddt),
            f"sampled/plddt_min": np.min(plddt),
            f"sampled/plddt_max": np.max(plddt),
            f"sampled/plddt_hist": wandb.Histogram(plddt, num_bins=max(len(plddt), 50))
        }
        return pdb_strs, metrics, log_dict

    def on_val_epoch_start(self, *_):
        print("val epoch start")

    def _run(self, pl_module, shape, log_to_wandb=True):
        torch.cuda.empty_cache()
        log_to_wandb = self.log_to_wandb and log_to_wandb

        device = pl_module.device
        logger = pl_module.logger.experiment

        maybe_print("sampling latent...")
        x, log_dict = self.sample_latent(shape)
        if log_to_wandb:
            logger.log(log_dict)
        if self.calc_fid:
            maybe_print("calculating FID...")
            log_dict = self.calculate_fid(x, device)
            if log_to_wandb:
                logger.log(log_dict)

        if not self.n_to_construct == -1:
            maybe_print(
                f"subsampling to only reconstruct {self.n_to_construct} samples..."
            )
            x = x[torch.randperm(x.shape[0])][: self.n_to_construct]

        if self.calc_sequence:
            maybe_print("constructing sequence...")
            seq_str, log_dict = self.construct_sequence(x, device)
            if log_to_wandb:
                logger.log(log_dict)

        if self.calc_structure:
            maybe_print("constructing structure...")
            pdb_strs, metrics, log_dict = self.construct_structure(x, seq_str, device)
            if log_to_wandb:
                logger.log(log_dict)

            if self.save_generated_structures:
                self.save_structures_to_disk(pdb_strs, pl_module.current_epoch)
    
    def save_structures_to_disk(self, pdb_strs, epoch_or_step: int):
        for i, pdbstr in enumerate(pdb_strs):
            outpath = self.outdir / f"generated_structures_{epoch_or_step}" / f"sample{i}.pdb"
            write_pdb_to_disk(pdbstr, outpath)

    def on_sanity_check_start(self, trainer, pl_module):
        _sampling_timesteps = pl_module.sampling_timesteps
        _n_to_sample = self.n_to_sample
        _n_to_construct = self.n_to_construct

        # hack
        pl_module.sampling_timesteps = 3
        self.n_to_sample, self.n_to_construct = 2, 2
        if rank_zero_only.rank == 0:
            dummy_shape = (2, 16, self.diffusion.model.input_dim)
            self._run(pl_module, shape=dummy_shape, log_to_wandb=False)
        
        # restore
        pl_module.sampling_timesteps = _sampling_timesteps
        self.n_to_sample = _n_to_sample
        self.n_to_construct = _n_to_construct
        print("done sampling sanity check")

        torch.cuda.empty_cache()

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch % self.run_every_n_epoch == 0) and not (trainer.current_epoch == 0):
            shape = (self.batch_size, self.gen_seq_len, self.diffusion.model.input_dim)
            self._run(pl_module, shape, log_to_wandb=True)
            torch.cuda.empty_cache()
        else:
            pass


if __name__ == "__main__":
    from plaid.diffusion import GaussianDiffusion
    from plaid.denoisers import UTriSelfAttnDenoiser
    from plaid.compression.uncompress import UncompressContinuousLatent

    uncompressor = UncompressContinuousLatent("tk3g5edg")
    denoiser = UTriSelfAttnDenoiser(
        hid_dim=1024,
        num_blocks=3,
        input_dim_if_different=8
    )
    diffusion = GaussianDiffusion(denoiser, uncompressor=uncompressor, sampling_timesteps=5)
    callback = SampleCallback(
        diffusion,
        denoiser,
        fid_holdout_tensor_fpath="/homefs/home/lux70/storage/data/rocklin/shards/val/esmfold/seqlen_256/fp32/shard0000.h5")
    device = torch.device("cuda:0")

    import IPython

    IPython.embed()

    x, log_dict = callback.sample_latent((4, 46, 8))
    log_dict = callback.calculate_fid(x, device)

    x = x[torch.randperm(x.shape[0])][: callback.n_to_construct]
    seq_str, log_dict = callback.construct_sequence(x, device)
    pdb_strs, outputs = callback.construct_structure(x, seq_str, device)

