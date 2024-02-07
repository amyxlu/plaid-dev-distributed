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

from plaid.denoisers import BaseDenoiser
from plaid.diffusion import GaussianDiffusion
from plaid.evaluation import RITAPerplexity, calc_fid_fn, calc_kid_fn
from plaid.proteins import LatentToSequence, LatentToStructure, write_pdb_to_disk
import pandas as pd
import wandb


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
        calc_fid: bool = True,
        fid_reference_dataset: str = "uniref",
        calc_perplexity: bool = True,
        num_recycles: int = 4,
        outdir: str = "sampled",
        sequence_decode_temperature: float = 1.0,
        sequence_constructor: T.Optional[LatentToSequence] = None,
        structure_constructor: T.Optional[LatentToStructure] = None,
    ):
        super().__init__()
        self.outdir = Path(outdir)
        self.diffusion = diffusion
        self.model = model
        self.log_to_wandb = log_to_wandb
        self.calc_structure = calc_structure
        self.calc_perplexity = calc_perplexity
        self.calc_fid = calc_fid
        self.fid_reference_dataset = fid_reference_dataset
        self.n_to_sample = n_to_sample
        self.num_recycles = num_recycles
        self.sequence_decode_temperature = sequence_decode_temperature

        self.is_save_setup = False
        self.is_fid_setup = False
        self.is_perplexity_setup = False
        self.sequence_constructor = sequence_constructor
        self.structure_constructor = structure_constructor

        batch_size = self.n_to_sample if batch_size == -1 else batch_size
        n_to_construct = self.n_to_sample if n_to_construct == -1 else n_to_construct
        self.batch_size = batch_size
        self.n_to_construct = n_to_construct
        self.gen_seq_len = gen_seq_len

    def _save_setup(self):
        if not self.outdir.exists():
            self.outdir.mkdir(parents=True)
            print("Created directory", self.outdir)
        self.is_save_setup = True

    def _perplexity_setup(self, device):
        self.perplexity_calc = RITAPerplexity(device)
        self.is_perplexity_setup = True

    def _fid_setup(self, device):
        if self.fid_reference_dataset == "uniref":
            cached_tensors_path = (
                Path(os.path.dirname(__file__))
                / "../../cached_tensors/holdout_esmfold_feats.st"
            )
        elif self.fid_reference_dataset == "cath":
            cached_tensors_path = (
                Path(os.path.dirname(__file__))
                / "../../cached_tensors/cath_esmfold_feats.st"
            )

        def load_saved_features(location, device="cpu"):
            return st.load_file(location)["features"].to(device)

        self.real_features = load_saved_features(cached_tensors_path, device=device)
        self.real_features = self.real_features[
            torch.randperm(self.real_features.size(0))[: self.n_to_sample]
        ]

    def sample_latent(self, shape):
        all_samples, n_samples = [], 0
        while n_samples < self.n_to_sample:
            sample = self.diffusion.sample(shape, clip_denoised=True)
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

        # just to be consistent since 50,000 features were saved. Not necessary though
        indices = torch.randperm(self.real_features.size(0))[: fake_features.shape[0]]
        real_features = self.real_features[indices]
        fake_features = fake_features.to(device=device)
        real_features = real_features.to(device=device)
        assert real_features.ndim == fake_features.ndim == 2

        fid = calc_fid_fn(fake_features, real_features)
        kid = calc_kid_fn(fake_features, real_features)

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
            perplexity = self.perplexity_calc.batch_eval(strs)
            print(f"Perplexity: {perplexity:.3f}")
            log_dict[f"sampled/perplexity"] = (perplexity,)

        return strs, log_dict

    def construct_structure(self, x_0, seq_str, device):
        if self.structure_constructor is None:
            # warning: this implicitly creates an ESMFold inference model, can be very memory consuming
            self.structure_constructor = LatentToStructure()

        self.structure_constructor.to(device)
        x_0 = x_0.to(device=device)

        with torch.no_grad():
            pdb_strs, metrics = self.structure_constructor.to_structure(
                x_0,
                sequences=seq_str,
                num_recycles=self.num_recycles,
                batch_size=self.batch_size,
            )

        log_dict = {
            f"sampled/plddt_mean": metrics["plddt"].mean(),
            f"sampled/plddt_std": metrics["plddt"].std(),
            f"sampled/plddt_min": metrics["plddt"].min(),
            f"sampled/plddt_max": metrics["plddt"].max(),
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

        maybe_print("constructing sequence...")
        seq_str, log_dict = self.construct_sequence(x, device, log_to_wandb)
        if log_to_wandb:
            logger.log(log_dict)

        if self.calc_structure:
            maybe_print("constructing structure...")
            pdb_strs, metrics, log_dict = self.construct_structure(x, seq_str, device)
            if log_to_wandb:
                logger.log(log_dict)

    def on_sanity_check_start(self, trainer, pl_module):
        _sampling_timesteps = pl_module.sampling_timesteps
        # hack
        pl_module.sampling_timesteps = 3
        if rank_zero_only.rank == 0:
            dummy_shape = (2, 16, self.diffusion.model.hid_dim)
            self._run(pl_module, shape=dummy_shape, log_to_wandb=False)
        pl_module.sampling_timesteps = _sampling_timesteps
        print("done sampling sanity check")
        torch.cuda.empty_cache()

    def on_train_epoch_end(self, trainer, pl_module):
        shape = (self.batch_size, self.gen_seq_len, self.diffusion.model.hid_dim)
        self._run(pl_module, shape, log_to_wandb=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    from plaid.diffusion import GaussianDiffusion
    from plaid.denoisers import UTriSelfAttnDenoiser

    denoiser = UTriSelfAttnDenoiser(1024, 3)
    diffusion = GaussianDiffusion(denoiser, sampling_timesteps=5)
    callback = SampleCallback(diffusion, denoiser)
    device = torch.device("cuda:0")

    x, log_dict = callback.sample_latent()
    log_dict = callback.calculate_fid(x, device)
    x = x[torch.randperm(x.shape[0])][: callback.n_to_construct]
    seq_str, log_dict = callback.construct_sequence(x, device)
    pdb_strs, metrics, log_dict = callback.construct_structure(x, seq_str, device)
    import IPython

    IPython.embed()
