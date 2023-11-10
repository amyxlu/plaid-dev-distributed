import typing as T
import os
import json
from pathlib import Path
import wandb
from dataclasses import dataclass, field
import torch
import enum

# import accelerate
import k_diffusion as K
from k_diffusion.config import ModelConfig, SampleCallbackConfig
import pandas as pd


# TODO: add other sampling methods
PROJECT_NAME = "kdplaid_sample"


class SampleCallback:
    def __init__(
        self,
        model: K.Denoiser,
        config: SampleCallbackConfig,
        model_config: ModelConfig,
        is_wandb_setup: bool = False,
        device: T.Optional[torch.device] = None,
    ):
        """Callback class that can be evoked as a standalone script or as a callback to a Trainer object.

        TODO: If using this during training and wanting to mulitprocess, need to use accelerate.Accelerator to wrap the model and the callback.

        Args:
            model (torch.nn.Module): _description_
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

        self.is_save_setup = False
        self.is_wandb_setup = is_wandb_setup
        self.is_perplexity_setup = False
        self.is_fid_setup = False
        self.cur_model_step = config.model_step
        self.model_id = config.model_id
        self.uid = str(K.utils.timestamp())

        self.sequence_constructor, self.structure_constructor = None, None

    def update_cur_step(self, cur_step):
        self.cur_model_step = cur_step

    def update_model_id(self, model_id):
        self.model_id = model_id

    def batch_sample_fn(self, n):
        cfg = self.config
        size = (n, cfg.seq_len, self.model_config.d_model)
        x = torch.randn(size, device=self.device) * cfg.sigma_max
        model_fn, extra_args = self.model, {
            "mask": torch.ones(n, cfg.seq_len, device=self.device).long()
        }
        sigmas = K.sampling.get_sigmas_karras(
            cfg.n_steps,
            cfg.sigma_min,
            cfg.sigma_max,
            rho=cfg.rho,
            device=self.device,
        )

        if cfg.sampling_method == "sample_dpmpp_2m_sde":
            x_0_raw = K.sampling.sample_dpmpp_2m_sde(
                model_fn,
                x,
                sigmas,
                extra_args=extra_args,
                eta=0.0,
                solver_type=self.config.solver_type,
                disable=True,
            )
        else:
            raise ValueError(f"unsupported sampling method {cfg.sampling_method}")

        # 2) Downproject latent space, maybe
        if self.model_config.d_model != self.model_config.input_dim:
            x_0 = self.model.inner_model.project_to_input_dim(x_0_raw)

        # 1) Normalize, maybe
        x_0 = K.normalization.undo_scale_embedding(
            x_0, self.model_config.normalize_latent_by
        )
        return x_0, x_0_raw

    def sample_latent(self, save=True):
        all_samples, all_raw_samples, n_samples = [], [], 0
        while n_samples < self.config.n_to_sample:
            sample, raw_sample = self.batch_sample_fn(self.config.batch_size)
            all_samples.append(sample.detach().cpu())
            all_raw_samples.append(raw_sample.detach().cpu())
            n_samples += sample.shape[0]
        x_0 = torch.cat(all_samples, dim=0)
        x_0_raw = torch.cat(all_raw_samples, dim=0)
        if save:
            if not self.is_save_setup:
                self._save_setup()
            torch.save(x_0, self.outdir / "samples.pth")
            torch.save(x_0_raw, self.outdir / "samples_raw.pth")
        return x_0

    def _save_setup(self):
        base_artifact_dir = Path(self.config.base_artifact_dir)
        self.outdir = base_artifact_dir / "samples" / self.model_id / self.uid
        if not self.outdir.exists():
            self.outdir.mkdir(parents=True)
            print("Created directory", self.outdir)

        args_dict = (K.config.dataclass_to_dict(self.config),)
        with open(self.outdir / "config.json", "w") as f:
            f.write(json.dumps(args_dict))
        self.is_save_setup = True

    def _wandb_setup(self):
        print(f"Setting up wandb logging for model {self.model_id}...")
        config = K.config.dataclass_to_dict(self.config)
        wandb.init(
            id=self.uid,
            name=self.model_id,
            project=PROJECT_NAME,
            entity="amyxlu",
            config=config,
            # resume="allow",
        )
        # config.pop("model_config")
        # wandb.config.update(config)
        self.is_wandb_setup = True

    def _perplexity_setup(self):
        self.perplexity_calc = K.evaluation.RITAPerplexity(self.device)
        self.is_perplexity_setup = True

    def _fid_setup(self):
        cached_tensors_path = os.path.join(
            os.__file__, "../cached_tensors/holdout_esmfold_feats.st"
        )

        def load_saved_features(self, location, device="cpu"):
            import safetensors.torch as st

            return st.load_file(location)["features"].to(device=device)

        self.real_features = load_saved_features(
            cached_tensors_path, device=self.device
        )

    def calculate_fid(self, sampled_latent, log_to_wandb=False):
        if not self.is_fid_setup:
            self._fid_setup()
        fake_features = sampled_latent.mean(dim=1)
        # just to be consistent since 50,000 features were saved. Not necessary though
        indices = torch.randperm(self.real_features.size(0))[: fake_features.shape[0]]
        real_features = self.real_features[indices]

        assert real_features.ndim == fake_features.ndim == 2
        fid = K.evaluation.fid(fake_features, real_features)
        kid = K.evaluation.kid(fake_features, real_features)

        if log_to_wandb:
            if not self._wandb_setup:
                self._wandb_setup()
            wandb.log({"fid": fid, "kid": kid, "step": self.config.model_step})
        return fid, kid

    def construct_sequence(
        self,
        x_0,
        calc_perplexity: bool = True,
        save_to_disk: bool = False,
        log_to_wandb: bool = False,
    ):
        if self.sequence_constructor is None:
            self.sequence_constructor = K.proteins.LatentToSequence(
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
            K.utils.write_to_fasta(strs, self.outdir / "sequences.fasta")
            with open(self.outdir / "batch_perplexity.txt", "w") as f:
                f.write(f"{perplexity:.3f}")

        if log_to_wandb:
            if not self.is_wandb_setup:
                self._wandb_setup()
            wandb.log(
                {
                    "sequences": wandb.Table(dataframe=sequence_results),
                    "batch_perplexity": perplexity,
                    "step": self.config.model_step,
                }
            )
        return probs, idxs, strs

    def construct_structure(self, x_0, seq_str, save_to_disk=True, log_to_wandb=False):
        if self.structure_constructor is None:
            self.structure_constructor = K.proteins.LatentToStructure(
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
                K.utils.write_pdb_to_disk(pdb_str, self.outdir / f"{i:03}.pdb")
            metrics.to_csv(self.outdir / "structure_metrics.csv")

        if log_to_wandb:
            wandb.log(
                {
                    "plddt_mean": metrics["plddt"].mean(),
                    "plddt_std": metrics["plddt"].std(),
                    "plddt_min": metrics["plddt"].min(),
                    "plddt_max": metrics["plddt"].max(),
                    "step": self.config.model_step,
                }
            )
        return pdb_strs, metrics


def main(
    config: SampleCallbackConfig,
):
    # Load checkpoint from model_id and step
    base_artifact_dir = Path(config.base_artifact_dir)
    if not config.model_step:
        assert not config.model_id is None
        state_path = base_artifact_dir / "checkpoints" / config.model_id / "state.json"
        state = json.load(open(state_path))
        filename = state["latest_checkpoint"]
        config.model_step = int(state["latest_checkpoint"].split("/")[-1].split(".")[0])
    else:
        filename = str(
            base_artifact_dir
            / "checkpoints"
            / config.model_id
            / f"{config.model_step:08}.pth"
        )

    print("Loading checkpoint from", filename)
    ckpt = torch.load(filename, map_location="cpu")
    model_config = K.config.ModelConfig(**ckpt["config"]["model_config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inner_model = (
        K.config.make_model(model_config).eval().requires_grad_(False).to(device)
    )
    if config.use_ema:
        inner_model.load_state_dict(ckpt["model_ema"])
    else:
        inner_model.load_state_dict(ckpt["model"])
    model = K.config.make_denoiser_wrapper(model_config)(inner_model)

    # instantiate sampler object
    print("Instantiating sampler callback object...")
    sampler = SampleCallback(
        model=model,
        config=config,
        model_config=model_config,
        is_wandb_setup=not config.log_to_wandb,  # skip set up if we're not planning to log to wandb
    )

    # sample latent and calculate KID/FID to the saved known distribution
    print("Sampling latent...")
    sampled_latent = sampler.sample_latent()

    print("Calculating FID/KID...")
    fid, kid = sampler.calculate_fid(sampled_latent, log_to_wandb=config.log_to_wandb)

    # potentially take a smaller subset to decode into structure/sequence and evaluate
    if not config.n_to_construct == -1:
        sampled_latent = torch.randperm(sampled_latent.shape[0])[
            : config.n_to_construct
        ]

    print("Constructing sequences...")
    _, _, strs = sampler.construct_sequence(
        sampled_latent,
        calc_perplexity=config.calc_perplexity,
        save_to_disk=config.save_to_disk,
        log_to_wandb=config.log_to_wandb,
    )

    print("Constructing structures...")
    pdb_strs, metrics = sampler.construct_structure(
        sampled_latent,
        strs,
        save_to_disk=config.save_to_disk,
        log_to_wandb=config.log_to_wandb,
    )


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
