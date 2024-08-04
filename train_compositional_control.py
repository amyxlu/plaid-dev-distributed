import typing as T
import os
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
from plaid.utils import find_latest_checkpoint, extract_step


# def format_wandb_run_name(cfg_name, compression_model_id):
#     if isinstance(cfg_name, str):
#         cfg_name += "_"
#     else:
#         cfg_name == ""
#     input_dim = constants.COMPRESSION_INPUT_DIMENSIONS[compression_model_id]
#     shorten_factor = constants.COMPRESSION_SHORTEN_FACTORS[compression_model_id]
#     return f"{cfg_name}dim{input_dim}_shorten{shorten_factor}"


@hydra.main(version_base=None, config_path="configs", config_name="train_diffusion")
def train(cfg: DictConfig):
    ####################################################################################################
    # Load old config if resuming
    ####################################################################################################
    import torch

    torch.set_float32_matmul_precision("medium")

    if rank_zero_only.rank == 0:
        # maybe use prior job id, else generate new ID
        if cfg.resume_from_model_id is not None:
            job_id = cfg.resume_from_model_id
            IS_RESUMED = True
        else:
            job_id = wandb.util.generate_id()
            IS_RESUMED = False

        # set up checkpoint and config yaml paths
        dirpath = Path(cfg.paths.checkpoint_dir) / "diffusion" / job_id
        outdir = Path(cfg.paths.artifacts_dir) / "samples" / job_id

        config_path = dirpath / "config.yaml"
        if config_path.exists():
            cfg = OmegaConf.load(config_path)
            rank_zero_info("*" * 10, "\n", "Overriding config from job ID", job_id, "\n", "*" * 10)
        else:
            dirpath.mkdir(parents=True)
            if not config_path.exists():
                OmegaConf.save(cfg, config_path)

        log_cfg = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)
        rank_zero_info(OmegaConf.to_yaml(log_cfg))

    ####################################################################################################
    # Data and beta scheduler
    ####################################################################################################

    # lightning data and model modules
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    # dimensions
    input_dim = constants.COMPRESSION_INPUT_DIMENSIONS[cfg.compression_model_id]
    shorten_factor = constants.COMPRESSION_SHORTEN_FACTORS[cfg.compression_model_id]
    denoiser = hydra.utils.instantiate(cfg.denoiser, input_dim=input_dim)

    ####################################################################################################
    # Diffusion module
    ####################################################################################################
    from plaid.utils import count_parameters
    diffusion = hydra.utils.instantiate()
    trainable_parameters = count_parameters(diffusion, require_grad_only=True)
    total_parameters = count_parameters(diffusion, require_grad_only=False)
    log_cfg["trainable_params_millions"] = trainable_parameters / 1_000_000
    log_cfg["total_params_millions"] = total_parameters / 1_000_000
    log_cfg["shorten_factor"] = shorten_factor
    log_cfg["input_dim"] = input_dim

    if not cfg.dryrun:
        run_name = format_wandb_run_name(cfg.logger.name, cfg.compression_model_id)
        # this will automatically log to the same wandb page
        logger = hydra.utils.instantiate(cfg.logger, id=job_id, name=run_name)
        # logger.watch(model, log="all", log_graph=False)
    else:
        logger = None

    ####################################################################################################
    # Set up callbacks
    ####################################################################################################

    lr_monitor = hydra.utils.instantiate(cfg.callbacks.lr_monitor)
    checkpoint_callback = hydra.utils.instantiate(cfg.callbacks.checkpoint, dirpath=dirpath)
    # ema_callback = hydra.utils.instantiate(cfg.callbacks.ema)

    callbacks = [lr_monitor, checkpoint_callback]  # , ema_callback]

    ####################################################################################################
    # Train
    ####################################################################################################

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=callbacks,
    )
    if rank_zero_only.rank == 0 and isinstance(trainer.logger, WandbLogger):
        trainer.logger.experiment.config.update({"cfg": log_cfg}, allow_val_change=True)

    if not cfg.dryrun:
        if IS_RESUMED:
            # job id / dirpath was already updated to match the to-be-resumed directory
            ckpt_fname = dirpath / find_latest_checkpoint(dirpath)
            rank_zero_info("Resuming from ", ckpt_fname)
            assert ckpt_fname.exists()
            trainer.fit(diffusion, datamodule=datamodule, ckpt_path=dirpath / ckpt_fname)
        else:
            trainer.fit(diffusion, datamodule=datamodule)


if __name__ == "__main__":
    train()
