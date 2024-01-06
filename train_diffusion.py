import typing as T
import os

import hydra
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf
import torch


@hydra.main(version_base=None, config_path="configs", config_name="train_diffusion")
def train(cfg: DictConfig):
    # general set up
    torch.set_float32_matmul_precision("medium")
    
    log_cfg = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)
    if rank_zero_only.rank == 0:
        print(OmegaConf.to_yaml(log_cfg))

    # lightning data and model modules
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup(stage="fit")

    denoiser = hydra.utils.instantiate(cfg.denoiser)
    beta_scheduler = hydra.utils.instantiate(cfg.beta_scheduler)
    latent_scaler = hydra.utils.instantiate(cfg.latent_scaler)
    diffusion = hydra.utils.instantiate(
        cfg.diffusion,
        model=denoiser,
        beta_scheduler=beta_scheduler,
        latent_scaler=latent_scaler,
    )

    job_id = os.environ.get("SLURM_JOB_ID")  # is None if not using SLURM

    if not cfg.dryrun:
        logger = WandbLogger(
            project="plaid",
            entity="lu-amy-al1",
            id=job_id
        )
        logger.watch(diffusion, log="all", log_graph=False)
    else:
        logger = None

    # callbacks/home/amyxlu/plaid/plaid/denoisers
    lr_monitor = hydra.utils.instantiate(cfg.callbacks.lr_monitor)
    checkpoint_callback = hydra.utils.instantiate(cfg.callbacks.checkpoint)
    # todo: add a wasserstein distance callback
    sample_callback = hydra.utils.instantiate(
        cfg.callbacks.sample,
        diffusion=diffusion,
        model=denoiser,
        log_to_wandb=not cfg.dryrun,
    )

    # run training
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=[lr_monitor, checkpoint_callback, sample_callback],
    )
    if rank_zero_only.rank == 0 and isinstance(trainer.logger, WandbLogger):
        trainer.logger.experiment.config.update({"cfg": log_cfg}, allow_val_change=True)

    if not cfg.dryrun:
        trainer.fit(diffusion, datamodule=datamodule)


if __name__ == "__main__":
    train()
