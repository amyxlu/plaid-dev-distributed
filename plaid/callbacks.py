import typing as T
import os
import json
from pathlib import Path
import wandb
import torch
from lightning.pytorch.callbacks import Callback
import safetensors as st

from plaid.denoisers import BaseDenoiser
from plaid.diffusion import GaussianDiffusion
from plaid.evaluation import RITAPerplexity, fid, kid
from plaid.utils import LatentToSequence, LatentToStructure, write_pdb_to_disk
import pandas as pd
import wandb


class SampleCallback(Callback):
    def __init__(
        self,
        diffusion: GaussianDiffusion,  # most sampling logic is here
        model: BaseDenoiser,
        n_to_sample: int = 16,
        n_to_construct: int = 4,
        batch_size: int = -1,
        log_to_wandb: bool = True,
        log_structure: bool = True,
        calc_perplexity: bool = True,
        outdir: str = "sampled",
    ):
        super().__init__()
        self.diffusion = diffusion
        self.model = model
        self.log_to_wandb = log_to_wandb
        self.log_structure = log_structure
        self.calc_perplexity = calc_perplexity
        self.n_to_sample = n_to_sample
        self.n_to_construct = n_to_construct
        self.sequence_constructor, self.structure_constructor = None, None
        self.outdir = Path(outdir)

        batch_size = self.n_to_sample if batch_size == -1 else batch_size
        self.batch_size = batch_size
        self.n_to_construct = n_to_construct

    def _save_setup(self):
        if not self.outdir.exists():
            self.outdir.mkdir(parents=True)
            print("Created directory", self.outdir)
        self.is_save_setup = True

    def _perplexity_setup(self, device):
        self.perplexity_calc = RITAPerplexity(device)
        self.is_perplexity_setup = True

    def _fid_setup(self, device):
        cached_tensors_path = (
            Path(os.path.dirname(__file__))
            / "../cached_tensors/holdout_esmfold_feats.st"
        )

        def load_saved_features(location, device="cpu"):
            return st.load_file(location)["features"].to(device)

        self.real_features = load_saved_features(
            cached_tensors_path, device=device
        )
        self.real_features = self.real_features[
            torch.randperm(self.real_features.size(0))[: self.config.n_to_sample]
        ]

    def sample_latent(self):
        all_samples, n_samples = [], 0
        while n_samples < self.n_to_sample:
            sample = self.diffusion.sample(self.batch_size, clip_denoised=True)
            all_samples.append(sample.detach().cpu())
            n_samples += sample.shape[0]
        x_0 = torch.cat(all_samples, dim=0)
        log_dict = {
            "sampled/latent_mean": x_0.mean(),
            "sampled/latent_std": x_0.std(),
        }
        if self.log_to_wandb:
            wandb.log(log_dict)
        return x_0

    def calculate_fid(self, sampled_latent, device):
        if not self.is_fid_setup:
            self._fid_setup()
        fake_features = sampled_latent.mean(dim=1)

        # just to be consistent since 50,000 features were saved. Not necessary though
        indices = torch.randperm(self.real_features.size(0))[: fake_features.shape[0]]
        real_features = self.real_features[indices]
        fake_features = fake_features.to(device=device)
        real_features = real_features.to(device=device)
        assert real_features.ndim == fake_features.ndim == 2

        fid = fid(fake_features, real_features)
        kid = kid(fake_features, real_features)

        log_dict = {f"fid": fid, f"kid": kid}
        if self.log_to_wandb:
            wandb.log(log_dict)
        return fid, kid

    def construct_sequence(
        self,
        x_0,
        device,
    ):
        if self.sequence_constructor is None:
            self.sequence_constructor = LatentToSequence(
                device=device, temperature=self.config.sequence_decode_temperature
            )

        probs, idxs, strs = self.sequence_constructor.to_sequence(x_0)
        sequence_results = pd.DataFrame(
            {
                "sequences": strs,
                "mean_residue_confidence": probs.mean(dim=1).cpu().numpy(),
            }
        )
        if self.calc_perplexity:
            if not self.is_perplexity_setup:
                self._perplexity_setup()
            perplexity = self.perplexity_calc.batch_eval(strs)
            print(f"Perplexity: {perplexity:.3f}")

        log_dict = {
            f"sampled/sequences": wandb.Table(dataframe=sequence_results),
            f"sampled/perplexity": perplexity,
        }
        if self.log_to_wandb:
            wandb.log(log_dict)
        return strs

    def construct_structure(self, x_0, seq_str, device):
        # TODO: maybe save & log artifact
        if self.structure_constructor is None:
            self.structure_constructor = LatentToStructure(device=device)
        pdb_strs, metrics = self.structure_constructor.to_structure(
            x_0,
            sequences=seq_str,
            num_recycles=self.config.num_recycles,
            batch_size=self.config.batch_size,
        )

        log_dict = {
            f"sampled/plddt_mean": metrics["plddt"].mean(),
            f"sampled/plddt_std": metrics["plddt"].std(),
            f"sampled/plddt_min": metrics["plddt"].min(),
            f"sampled/plddt_max": metrics["plddt"].max(),
        }
        if self.log_to_wandb:
            wandb.log(log_dict)

        if self.log_structure:
            if not self.is_save_setup:
                self._save_setup()
            for i, pdb_str in enumerate(pdb_strs[:4]):
                outpath = write_pdb_to_disk(pdb_str, self.outdir / f"{i:03}.pdb")
                wandb.log({f"sampled/structure": wandb.Molecule(outpath)})
        return pdb_strs, metrics

    def on_train_epoch_end(self, trainer, pl_module):
        device = pl_module.device
        x = self.sample_latent()
        _ = self.calculate_fid(x, device)
        if not self.n_to_construct == -1:
            x = x[torch.randperm(x.shape[0])][self.n_to_construct]
        seq_str = self.construct_sequence(x, device)
        _ = self.construct_structure(x, seq_str, device)
        pl_module.training_step_outputs.clear()
