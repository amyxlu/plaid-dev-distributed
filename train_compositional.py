import typing as T
from pathlib import Path

import wandb
import hydra
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only, rank_zero_info

from omegaconf import DictConfig, OmegaConf
import torch
import re

from plaid import constants
from plaid.utils import count_parameters, find_latest_checkpoint


_PROJECT_NAME = "plaid_compositional_conditioning"


def resume_setup(cfg, project_name):
    # maybe use prior job id, else generate new ID
    if cfg.resume_from_model_id is not None:
        job_id = cfg.resume_from_model_id
        is_resumed = True
    else:
        job_id = wandb.util.generate_id()
        is_resumed = False

    # set up checkpoint and config yaml paths
    outdir = Path(cfg.paths.checkpoint_dir) / project_name / job_id
    config_path = outdir / "config.yaml"
    if config_path.exists():
        cfg = OmegaConf.load(config_path)
        rank_zero_info(f"Overriding config from job ID {job_id}!")
    else:
        outdir.mkdir(parents=True)
        if not config_path.exists():
            OmegaConf.save(cfg, config_path)

    log_cfg = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)
    rank_zero_info(OmegaConf.to_yaml(log_cfg))
    return log_cfg, job_id, outdir, is_resumed


@hydra.main(version_base=None, config_path="configs", config_name="train_compositional")
def train(cfg: DictConfig):
    import torch
    torch.set_float32_matmul_precision("medium")

    if rank_zero_only.rank == 0:
        log_cfg, job_id, outdir, is_resumed = resume_setup(cfg, _PROJECT_NAME)

    ####################################################################################################
    # Data and model setup  
    ####################################################################################################

    # WebDataset loader over sharded pre-computed embeddings
    rank_zero_info("Creating datamodule...")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    # dimensions
    input_dim = constants.COMPRESSION_INPUT_DIMENSIONS[cfg.compression_model_id]
    shorten_factor = constants.COMPRESSION_SHORTEN_FACTORS[cfg.compression_model_id]

    # model
    rank_zero_info("Creating model...")
    denoiser = hydra.utils.instantiate(cfg.denoiser, input_dim=input_dim)
    diffusion = hydra.utils.instantiate(cfg.diffusion, model=denoiser)

    # logging details
    if rank_zero_only.rank == 0:
        trainable_parameters = count_parameters(diffusion, require_grad_only=True)
        total_parameters = count_parameters(diffusion, require_grad_only=False)
        log_cfg["trainable_params_millions"] = trainable_parameters / 1_000_000
        log_cfg["total_params_millions"] = total_parameters / 1_000_000
        log_cfg["shorten_factor"] = shorten_factor
        log_cfg["input_dim"] = input_dim
        logger = hydra.utils.instantiate(cfg.logger, id=job_id)

    # checkpoint and LR callbacks
    lr_monitor = hydra.utils.instantiate(cfg.callbacks.lr_monitor)
    checkpoint_callback = hydra.utils.instantiate(cfg.callbacks.checkpoint, dirpath=outdir)

    callbacks = [lr_monitor, checkpoint_callback]

    ####################################################################################################
    # Trainer
    ####################################################################################################

    rank_zero_info("Initializing training...")

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=callbacks,
    )

    if rank_zero_only.rank == 0 and isinstance(trainer.logger, WandbLogger):
        trainer.logger.experiment.config.update({"cfg": log_cfg}, allow_val_change=True)

    if is_resumed:
        assert ckpt_fname.exists()
        rank_zero_info("Resuming from ", ckpt_fname)
        ckpt_fname = outdir / find_latest_checkpoint(outdir)
        trainer.fit(diffusion, datamodule=datamodule, ckpt_path=outdir / ckpt_fname)

    else:
        trainer.fit(diffusion, datamodule=datamodule)


if __name__ == "__main__":
    train()
