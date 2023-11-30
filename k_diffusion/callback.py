import typing as T
import os
import json
from pathlib import Path
import wandb
from dataclasses import dataclass, field
import torch
import enum

from .model_loading_utils import load_model, infer_ckpt_path
from .config import ModelConfig, SampleCallbackConfig, make_sample_solver_fn, dataclass_to_dict
from . import utils, normalization, sampling, proteins, evaluation, layers 
import pandas as pd
import safetensors.torch as st


PROJECT_NAME = "kdplaid_sample"


class SampleCallback:
    def __init__(
        self,
        model: layers.Denoiser,
        config: SampleCallbackConfig,
        model_config: ModelConfig,
        origin_dataset: str = "uniref",
        is_wandb_setup: bool = False,
        device: T.Optional[torch.device] = None,
    ):
        """Callback class that can be evoked as a standalone script or as a callback to a Trainer object.

        TODO: If using this during training and wanting to mulitprocess, need to use accelerate.Accelerator to wrap the model and the callback.

        Args:
            model (torch.nn.Module): _description
            config (SampleCallbackConfig): _description_
            use_wandb (bool, optional): _description_. Defaults to False.
        """
        self.model = model
        self.config = config
        self.model_config = model_config
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model.to(self.device)

        self.is_save_setup = False
        self.is_wandb_setup = is_wandb_setup
        self.is_perplexity_setup = False
        self.is_fid_setup = False
        self.cur_model_step = config.model_step
        self.model_id = config.model_id
        self.uid = str(utils.timestamp())
        self.origin_dataset = origin_dataset
        self.latent_scaler = normalization.LatentScaler(
            mode=model_config.normalize_latent_by,
            origin_dataset=origin_dataset,
            lm_embedder_type=getattr(model_config, "lm_embedder_type", "esmfold")
        )

        self.sequence_constructor, self.structure_constructor = None, None

    def update_cur_step(self, cur_step):
        self.cur_model_step = cur_step

    def update_model_id(self, model_id):
        self.model_id = model_id

    def batch_sample_fn(self, n, clip_range=None):
        cfg = self.config
        size = (n, cfg.seq_len, self.model_config.d_model)
        x = torch.randn(size, device=self.device) * cfg.sigma_max
        model_fn, extra_args = self.model, {
            "mask": torch.ones(n, cfg.seq_len, device=self.device).long(),
        }
        sigmas = sampling.get_sigmas_karras(
            cfg.n_steps,
            cfg.sigma_min,
            cfg.sigma_max,
            rho=cfg.rho,
            device=self.device,
        )
        sample_fn = make_sample_solver_fn(cfg.solver_type)
        sample_fn_kwargs = {"sigmas": sigmas, "clip_range": clip_range, "extra_args": extra_args}
        x_0_raw = sample_fn(model_fn, x, **sample_fn_kwargs)

        # 2) Downproject latent space, maybe
        if self.model_config.d_model != self.model_config.input_dim:
            x_0 = self.model.inner_model.project_to_input_dim(x_0_raw)
        else:
            x_0 = x_0_raw

        # 1) Normalize, maybe
        x_0 = self.latent_scaler.unscale(x_0)
        return x_0, x_0_raw

    def sample_latent(self, clip_range=None, save=True, log_wandb_stats=False, return_raw=False):
        all_samples, all_raw_samples, n_samples = [], [], 0
        while n_samples < self.config.n_to_sample:
            sample, raw_sample = self.batch_sample_fn(self.config.batch_size, clip_range=clip_range)
            all_samples.append(sample.detach().cpu())
            all_raw_samples.append(raw_sample.detach().cpu())
            n_samples += sample.shape[0]
        x_0 = torch.cat(all_samples, dim=0)
        x_0_raw = torch.cat(all_raw_samples, dim=0)

        print("Sampled latent mean:", x_0.mean(), "std:", x_0.std())
        print("Raw sampled latent mean:", x_0_raw.mean(), "std:", x_0_raw.std())

        if save:
            if not self.is_save_setup:
                self._save_setup()
            torch.save(x_0, self.outdir / "samples.pth")
            torch.save(x_0_raw, self.outdir / "samples_raw.pth")
        
        log_dict = {
            "sampled_latent_mean": x_0.mean(),
            "sampled_latent_std": x_0.std(),
            "raw_sampled_latent_mean": x_0_raw.mean(),
            "raw_sampled_latent_std": x_0_raw.std(),
            "step": self.config.model_step,
        }

        if log_wandb_stats:
            if not self.is_wandb_setup:
                self._wandb_setup()
            wandb.log(log_dict)
        
        if return_raw:
            return x_0, x_0_raw, log_dict
        else:
            return x_0, log_dict

    def _save_setup(self):
        base_artifact_dir = Path(self.config.base_artifact_dir)
        self.outdir = base_artifact_dir / "samples" / self.model_id / self.uid
        if not self.outdir.exists():
            self.outdir.mkdir(parents=True)
            print("Created directory", self.outdir)

        args_dict = (dataclass_to_dict(self.config),)
        with open(self.outdir / "config.json", "w") as f:
            f.write(json.dumps(args_dict))
        self.is_save_setup = True

    def _wandb_setup(self):
        print(f"Setting up wandb logging for model {self.model_id}...")
        print("Warning: this should only be used if you're NOT using the class as a Trainer callback.")
        config = dataclass_to_dict(self.config)
        wandb.init(
            id=self.uid,
            name=f"{self.model_id}_{str(self.config.solver_type)}",
            project=PROJECT_NAME,
            entity="amyxlu",
            config=config,
            # resume="allow",
        )
        self.is_wandb_setup = True

    def _perplexity_setup(self):
        self.perplexity_calc = evaluation.RITAPerplexity(self.device)
        self.is_perplexity_setup = True

    def _fid_setup(self):
        cached_tensors_path = (
            Path(os.path.dirname(__file__)) / "../cached_tensors/holdout_esmfold_feats.st"
        )

        def load_saved_features(location, device="cpu"):
            return st.load_file(location)["features"].to(device=device)

        self.real_features = load_saved_features(
            cached_tensors_path, device=self.device
        )
        self.real_features = self.real_features[torch.randperm(self.real_features.size(0))[:self.config.n_to_sample]]

    def calculate_fid(self, sampled_latent, log_to_wandb=False, wandb_prefix: str = ""):
        if not self.is_fid_setup:
            self._fid_setup()
        fake_features = sampled_latent.mean(dim=1)

        # just to be consistent since 50,000 features were saved. Not necessary though
        indices = torch.randperm(self.real_features.size(0))[: fake_features.shape[0]]
        real_features = self.real_features[indices]
        fake_features = fake_features.to(device=self.device)
        real_features = real_features.to(device=self.device)
        assert real_features.ndim == fake_features.ndim == 2

        fid = evaluation.fid(fake_features, real_features)
        kid = evaluation.kid(fake_features, real_features)

        log_dict = {f"{wandb_prefix}fid": fid, f"{wandb_prefix}kid": kid, "step": self.config.model_step}
        if log_to_wandb:
            if not self._wandb_setup:
                self._wandb_setup()
            wandb.log(log_dict)
        return fid, kid, log_dict

    def construct_sequence(
        self,
        x_0,
        calc_perplexity: bool = True,
        save_to_disk: bool = False,
        log_to_wandb: bool = False,
        wandb_prefix: str = "",
    ):
        if self.sequence_constructor is None:
            self.sequence_constructor = proteins.LatentToSequence(
                device=self.device, temperature=self.config.sequence_decode_temperature
            )

        probs, idxs, strs = self.sequence_constructor.to_sequence(x_0)
        sequence_results = pd.DataFrame(
            {
                "sequences": strs,
                "mean_residue_confidence": probs.mean(dim=1).cpu().numpy(),
                # add additional log-to-disk metrics here
            }
        )
        if calc_perplexity:
            if not self.is_perplexity_setup:
                self._perplexity_setup()
            perplexity = self.perplexity_calc.batch_eval(strs)
            print(f"Perplexity: {perplexity:.3f}")

        if save_to_disk:
            if not self.is_save_setup:
                self._save_setup()
            utils.write_to_fasta(strs, self.outdir / "sequences.fasta")
            with open(self.outdir / "batch_perplexity.txt", "w") as f:
                f.write(f"{perplexity:.3f}")

        log_dict = {
            f"{wandb_prefix}sequences": wandb.Table(dataframe=sequence_results),
            f"{wandb_prefix}batch_perplexity": perplexity,
            "step": self.config.model_step,
        }
        if log_to_wandb:
            if not self.is_wandb_setup:
                self._wandb_setup()
            wandb.log(log_dict)
        return probs, idxs, strs, log_dict

    def construct_structure(self, x_0, seq_str, save_to_disk=True, log_to_wandb=False, wandb_prefix: str = ""):
        if self.structure_constructor is None:
            self.structure_constructor = proteins.LatentToStructure(
                device=self.device
            )
        pdb_strs, metrics = self.structure_constructor.to_structure(
            x_0,
            sequences=seq_str,
            num_recycles=self.config.num_recycles,
            batch_size=self.config.batch_size,
        )
        if save_to_disk:
            if not self.is_save_setup:
                self._save_setup()
            for i, pdb_str in enumerate(pdb_strs):
                utils.write_pdb_to_disk(pdb_str, self.outdir / f"{i:03}.pdb")
            metrics.to_csv(self.outdir / "structure_metrics.csv")

        log_dict = {
                f"{wandb_prefix}plddt_mean": metrics["plddt"].mean(),
                f"{wandb_prefix}plddt_std": metrics["plddt"].std(),
                f"{wandb_prefix}plddt_min": metrics["plddt"].min(),
                f"{wandb_prefix}plddt_max": metrics["plddt"].max(),
                f"step": self.config.model_step,
            }
        if log_to_wandb:
            wandb.log(log_dict)
        return pdb_strs, metrics, log_dict


def main(
    config: SampleCallbackConfig,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = infer_ckpt_path(config.model_dir, config.model_id, config.model_step)
    model, inner_model, model_config = load_model(ckpt_path, config.use_ema)
    config.model_step = int(Path(ckpt_path).name.split(".")[0])

    # instantiate sampler object
    print("Instantiating sampler callback object...")
    sampler = SampleCallback(
        model=model,
        config=config,
        model_config=model_config,
        is_wandb_setup=not config.log_to_wandb,  # skip set up if we're not planning to log to wandb
        device=device,
    )

    # Load checkpoint from model_id and step
    # sample latent and calculate KID/FID to the saved known distribution
    print("Sampling latent...")
    sampled_latent, sampled_raw, _ = sampler.sample_latent(
        clip_range=config.clip_range, save=True, log_wandb_stats=config.log_to_wandb, return_raw=True
    )
    sampled_raw_expanded = inner_model.project_to_input_dim(sampled_raw.to(device))

    if config.calc_fid:
        print("Calculating FID/KID...")
        fid, kid = sampler.calculate_fid(
            sampled_latent, log_to_wandb=config.log_to_wandb
        )
        print("FID:", fid, "KID:", kid)

        # print("Calculating unnormalized FID/KID...")
        # fid, kid, _ = sampler.calculate_fid(
        #     sampled_raw_expanded, log_to_wandb=config.log_to_wandb, wandb_prefix="raw_unnormalized_"
        # )

    # potentially take a smaller subset to decode into structure/sequence and evaluate
    if not config.n_to_construct == -1:
        sampled_latent = sampled_latent[torch.randperm(sampled_latent.shape[0])][
            : config.n_to_construct
        ]

    print("Constructing sequences...")
    _, _, strs, _ = sampler.construct_sequence(
        sampled_latent,
        calc_perplexity=config.calc_perplexity,
        save_to_disk=config.save_to_disk,
        log_to_wandb=config.log_to_wandb,
    )

    # print("Constructing sequences from unnormalized latent...")
    # _, _, strs = sampler.construct_sequence(
    #     sampled_raw_expanded,
    #     calc_perplexity=config.calc_perplexity,
    #     save_to_disk=config.save_to_disk,
    #     log_to_wandb=config.log_to_wandb,
    #     wandb_prefix="raw_unnormalized_"
    # )

    print("Constructing structures...")
    _, metrics. _ = sampler.construct_structure(
        sampled_latent,
        strs,
        save_to_disk=config.save_to_disk,
        log_to_wandb=config.log_to_wandb,
    )
    print(metrics)


if __name__ == "__main__":
    import tyro

    config = tyro.cli(SampleCallbackConfig)
    try:
        main(config)
    except:
        import pdb, sys, traceback

        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
