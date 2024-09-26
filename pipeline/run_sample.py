from plaid.pipeline import SampleLatent
import hydra
from omegaconf import DictConfig, OmegaConf


def run(cfg: DictConfig):
    """Hydra configurable instantiation, for imports in full pipeline."""
    sample_latent = SampleLatent(
        model_id=cfg.model_id,
        model_ckpt_dir=cfg.model_ckpt_dir,
        organism_idx=cfg.organism_idx,
        function_idx=cfg.function_idx,
        cond_scale=cfg.cond_scale,
        num_samples=cfg.num_samples,
        beta_scheduler_name=cfg.beta_scheduler_name,
        sampling_timesteps=cfg.sampling_timesteps,
        batch_size=cfg.batch_size,
        length=cfg.length,
        return_all_timesteps=cfg.return_all_timesteps,
        output_root_dir=cfg.output_root_dir,
    )
    sample_latent = sample_latent.run()
    with open(sample_latent.outdir / "sample.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))


@hydra.main(config_path="/homefs/home/lux70/code/plaid/configs/pipeline/sample", config_name="sample_latent")
def hydra_run(cfg: DictConfig):
    """Hydra configurable instantiation for running as standalone script."""
    run(cfg)


if __name__ == "__main__":
    hydra_run()