from plaid.pipeline._sample import DPMSolverSampleLatent
import hydra
from omegaconf import DictConfig


def run(cfg: DictConfig):
    """Hydra configurable instantiation, for imports in full pipeline."""
    sample_latent = DPMSolverSampleLatent(
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
        ### DPM specific:
        t_start=cfg.t_start,
        t_end=cfg.t_end,
        order=cfg.order,
        skip_type=cfg.skip_type,
        method=cfg.method,
        lower_order_final=cfg.lower_order_final,
        denoise_to_zero=cfg.denoise_to_zero,
        solver_type=cfg.solver_type,
        atol=cfg.atol,
        rtol=cfg.rtol,
    )
    sample_latent.run()


@hydra.main(config_path="../configs/pipeline", config_name="dpm_sample_latent")
def hydra_run(cfg: DictConfig):
    """Hydra configurable instantiation for running as standalone script."""
    run(cfg)


if __name__ == "__main__":
    hydra_run()