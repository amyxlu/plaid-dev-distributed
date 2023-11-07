import typing as T
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


class SampleCallback:
    def __init__(
        self,
        model: K.Denoiser,
        config: SampleCallbackConfig,
        model_config: ModelConfig,
        is_wandb_setup: bool = False,
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
        self.device = torch.device(f"cuda:{config.device_id}")
        
        self.is_save_setup = False
        self.is_wandb_setup = is_wandb_setup
        self.is_perplexity_setup = False
        self.cur_model_step = config.model_step
        self.model_id = config.model_id

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
            x_0 = K.sampling.sample_dpmpp_2m_sde(
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
            x_0 = self.model.inner_model.project_to_input_dim(x_0)

        # 1) Normalize, maybe
        x_0 = K.normalization.undo_scale_embedding(
            x_0, self.model_config.normalize_latent_by
        )
        return x_0

    def sample_latent(self, save=True):
        all_samples, n_samples = [], 0
        while n_samples < self.config.n_to_sample:
            sampled = self.batch_sample_fn(self.config.batch_size)
            all_samples.append(sampled.detach().cpu())
            n_samples += sampled.shape[0]
        x_0 = torch.cat(all_samples, dim=0)
        if save:
            if not self.is_save_setup:
                self._save_setup()
            torch.save(x_0, self.outdir / "samples.pth")
        return x_0

    def _save_setup(self):
        base_artifact_dir = Path(self.config.base_artifact_dir)
        self.outdir = (
            base_artifact_dir
            / "samples"
            / self.model_id
            / str(K.utils.timestamp())
        )
        if not self.outdir.exists():
            self.outdir.mkdir(parents=True)
            print("Created directory", self.outdir)
        json.dump(
            K.config.dataclass_to_dict(self.config),
            open(self.outdir / "config.json", "w"),
        )
        self.is_save_setup = True

    def _wandb_setup(self):
        print(f"Setting up wandb logging for model {self.model_id}...")
        wandb.init(
            id=self.model_id, resume="allow", project="kdplaid", entity="amyxlu"
        )
        config = K.config.dataclass_to_dict(self.config)
        # config.pop("model_config")
        wandb.config.update(config)
        self.is_wandb_setup = True

    def _perplexity_setup(self):
        self.perplexity_calc = K.evaluation.RITAPerplexity(self.device)
        self.is_perplexity_setup = True

    def construct_sequence(
        self,
        x_0,
        calc_perplexity: bool = True,
        save_to_disk: bool = False,
        log_to_wandb: bool = False,
    ):
        if self.sequence_constructor is None:
            self.sequence_constructor = K.proteins.LatentToSequence(device=self.device)

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
            # wandb.log({"sequences": wandb.Table(dataframe=sequence_results)})
            wandb.log({"batch_perplexity": perplexity}, step=self.config.model_step)
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
            wandb.log({
                "plddt_mean": metrics["plddt"].mean(),
                "plddt_std": metrics["plddt"].std(),
                "plddt_min": metrics["plddt"].min(),
                "plddt_max": metrics["plddt"].max(),
            })
        return pdb_strs, metrics


def main(
    args: SampleCallbackConfig,
):
    # Load checkpoint from model_id and step
    base_artifact_dir = Path(args.base_artifact_dir)
    if not args.model_step:
        assert not args.model_id is None
        state_path = base_artifact_dir / "checkpoints" / args.model_id / "state.json"
        state = json.load(open(state_path))
        filename = state["latest_checkpoint"]
        args.model_step = int(state['latest_checkpoint'].split('/')[-1].split('.')[0])
    else:
        filename = str(
            base_artifact_dir
            / "checkpoints"
            / args.model_id
            / f"{args.model_step:08}.pth"
        )

    print("Loading checkpoint from", filename)
    ckpt = torch.load(filename, map_location="cpu")
    model_config = K.config.ModelConfig(**ckpt["config"]["model_config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    inner_model = K.config.make_model(model_config).eval().requires_grad_(False).to(device)
    if args.use_ema:
        inner_model.load_state_dict(ckpt['model_ema'])
    else:
        inner_model.load_state_dict(ckpt['model'])
    model = K.config.make_denoiser_wrapper(model_config)(inner_model)

    # instantiate sampler object
    print("Instantiating sampler callback object...")
    sampler = SampleCallback(
        model=model,
        config=args,
        model_config=model_config,
        is_wandb_setup=not args.log_to_wandb,  # skip set up if we're not planning to log to wandb
    )

    # sample latent and construct sequences and structures
    print("Sampling latent...")
    sampled_latent = sampler.sample_latent()

    print("Constructing sequences...")
    _, _, strs = sampler.construct_sequence(
        sampled_latent,
        calc_perplexity=args.calc_perplexity,
        save_to_disk=args.save_to_disk,
        log_to_wandb=args.log_to_wandb,
    )

    print("Constructing structures...")
    pdb_strs, metrics = sampler.construct_structure(
        sampled_latent, strs, save_to_disk=args.save_to_disk, log_to_wandb=args.log_to_wandb
    )


if __name__ == "__main__":
    import tyro
    args = tyro.cli(SampleCallbackConfig)
    try:
        main(args)
    except:
        import pdb, sys, traceback

        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
