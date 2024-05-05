"""
Elucidated diffusion model with classifier-free guidance and multi-GPU scaling.
"""

import typing as T
import os
from pathlib import Path
import logging

import wandb
import hydra
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf
import torch

from plaid.proteins import LatentToSequence, LatentToStructure
from plaid.utils import count_parameters

logger = logging.getLogger("lightning.pytorch")
logger.setLevel(logging.DEBUG)

logger = logging.getLogger("torch._dynamo")
logger.setLevel(logging.DEBUG)


@hydra.main(version_base=None, config_path="configs", config_name="train_edm")
def train(cfg: DictConfig):

    """
    Job resuming and config settings
    """
    torch.set_float32_matmul_precision("medium")

    if cfg.resume_from_model_id is not None:
        job_id = cfg.resume_from_model_id
    else:
        job_id = wandb.util.generate_id()

    # set up checkpoint and config yaml paths 
    dirpath = Path(cfg.paths.checkpoint_dir) / "hourglass_vq" / job_id
    config_path = dirpath / "config.yaml"
    if config_path.exists():
        cfg = OmegaConf.load(config_path)
        print("*" * 10, "\n", "Overriding config from job ID", job_id,  "\n", "*" * 10)
    else:
        dirpath.mkdir(parents=False)
        if not config_path.exists():
            OmegaConf.save(cfg, config_path)

    log_cfg = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)
    if rank_zero_only.rank == 0:
        print(OmegaConf.to_yaml(log_cfg))

    # save config
    dirpath = Path(cfg.paths.checkpoint_dir) / "diffusion" / job_id
    dirpath.mkdir(parents=False)
    config_path = dirpath / "config.yaml"
    if not config_path.exists():
        OmegaConf.save(cfg, config_path)



    """
    Auxiliary networks and preprocessing modules
    """

    def to_load_sequence_constructor(cfg):
        # if (cfg.diffusion.sequence_decoder_weight > 0.0) or (
        #     cfg.callbacks.sample.calc_perplexity
        # ):
        if cfg.callbacks.sample.calc_perplexity:
            # this loads the trained decoder:
            return LatentToSequence(temperature=1.0)
        else:
            return None

    def to_load_structure_constructor(cfg):
        # if (cfg.diffusion.structure_decoder_weight > 0.0) or (
        #     cfg.callbacks.sample.calc_structure
        # ):
        if cfg.callbacks.sample.calc_structure:
            # this creates the esmfold trunk on CPU, without the LM:
            return LatentToStructure()
        else:
            return None
    
    def to_load_uncompressor(cfg):
        if isinstance(cfg.compression_model_id, str):
            from plaid.compression.uncompress import UncompressContinuousLatent
            return UncompressContinuousLatent(
                cfg.compression_model_id,
                cfg.compression_ckpt_dir
            ) 
        else:
            return None

    # maybe make sequence/structure constructors
    sequence_constructor = to_load_sequence_constructor(cfg)
    structure_constructor = to_load_structure_constructor(cfg)
    uncompressor = to_load_uncompressor(cfg)
    latent_scaler = hydra.utils.instantiate(cfg.latent_scaler)



    """
    Data modules
    """
    # lightning data and model modules
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup(stage="fit")



    """
    Set up denoiser, diffusion, and sigma scheduler
    """

    # make denoiser
    denoiser = hydra.utils.instantiate(cfg.denoiser)

    USE_COMPILE = cfg.use_compile
    if USE_COMPILE:
        denoiser = torch.compile(denoiser)
    
    # make sigma density
    sigma_density_generator = hydra.utils.instantiate(cfg.beta_scheduler)

    # make diffusion
    diffusion = hydra.utils.instantiate(
        denoiser=denoiser,
        sigma_density_generator=sigma_density_generator,
        unscaler=latent_scaler,
        uncompressor=uncompressor,
        ema_checkpoint_folder=str(dirpath / "post-hoc-ema-checkpoints")
    )

    # log parameters
    trainable_parameters = count_parameters(diffusion, require_grad_only=True)
    total_parameters = count_parameters(diffusion, require_grad_only=False)
    log_cfg['trainable_params'] = trainable_parameters
    log_cfg['total_params'] = total_parameters


    """
    Callbacks
    """

    # logging
    if not cfg.dryrun:
        # this will automatically log to the same wandb page
        logger = hydra.utils.instantiate(cfg.logger, id=job_id)
        logger.watch(diffusion, log="all", log_graph=False)
    else:
        logger = None


    # sampling
    # TODO: set up DDP
    # TODO: add a wasserstein distance callback to sequence properties
    outdir = Path(cfg.paths.artifacts_dir) / "samples" / job_id

    sample_callback = hydra.utils.instantiate(
        cfg.callbacks.sample,
        outdir=outdir,
        diffusion=diffusion,
        log_to_wandb=not cfg.dryrun,
        sequence_constructor=sequence_constructor,
        structure_constructor=structure_constructor,
    )

    # lr monitoring and checkpoint saving
    lr_monitor = hydra.utils.instantiate(cfg.callbacks.lr_monitor)
    checkpoint_callback = hydra.utils.instantiate(cfg.callbacks.checkpoint, dirpath=dirpath)


    """
    Training
    """
    # TODO: set up DDP better
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
