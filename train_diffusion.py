import typing as T
import os
from pathlib import Path

import wandb
import hydra
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf
import torch

from plaid.proteins import LatentToSequence, LatentToStructure
from plaid import constants


@hydra.main(version_base=None, config_path="configs", config_name="train_diffusion")
def train(cfg: DictConfig):
    # general set up
    torch.set_float32_matmul_precision("medium")

    # maybe use prior job id, else generate new ID
    if cfg.resume_from_model_id is not None:
        job_id = cfg.resume_from_model_id 
    else:
        job_id = wandb.util.generate_id() 
    
    # TODO: load config if resuming run

    log_cfg = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)
    if rank_zero_only.rank == 0:
        print(OmegaConf.to_yaml(log_cfg))

    def to_load_sequence_constructor(cfg):
        if (cfg.diffusion.sequence_decoder_weight > 0.0) or (
            cfg.callbacks.sample.calc_perplexity
        ):
            # this loads the trained decoder:
            return LatentToSequence(temperature=1.0)
        else:
            return None

    def to_load_structure_constructor(cfg):
        if (cfg.diffusion.structure_decoder_weight > 0.0) or (
            cfg.callbacks.sample.calc_structure
        ):
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

    # lightning data and model modules
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup(stage="fit")

    # dimensions
    input_dim = constants.COMPRESSION_INPUT_DIMENSIONS[cfg.compression_model_id]
    shorten_factor = constants.COMPRESSION_SHORTEN_FACTORS[cfg.compression_model_id]

    denoiser = hydra.utils.instantiate(cfg.denoiser, input_dim=input_dim)
    beta_scheduler = hydra.utils.instantiate(cfg.beta_scheduler)

    from plaid.utils import count_parameters

    diffusion = hydra.utils.instantiate(
        cfg.diffusion,
        model=denoiser,
        beta_scheduler=beta_scheduler,
        sequence_constructor=sequence_constructor,
        structure_constructor=structure_constructor,
        unscaler=latent_scaler,
        uncompressor=uncompressor,
        shorten_factor=shorten_factor
    )

    trainable_parameters = count_parameters(diffusion, require_grad_only=True)
    total_parameters = count_parameters(diffusion, require_grad_only=False)
    log_cfg['trainable_params'] = trainable_parameters
    log_cfg['total_params'] = total_parameters

    # save config
    dirpath = Path(cfg.paths.checkpoint_dir) / "diffusion" / job_id
    dirpath.mkdir(parents=False)
    config_path = dirpath / "config.yaml"
    if not config_path.exists():
        OmegaConf.save(cfg, config_path)

    # other callbacks
    lr_monitor = hydra.utils.instantiate(cfg.callbacks.lr_monitor)
    checkpoint_callback = hydra.utils.instantiate(cfg.callbacks.checkpoint, dirpath=dirpath)
    ema_callback = hydra.utils.instantiate(cfg.callbacks.ema)

    if not cfg.dryrun:
        # this will automatically log to the same wandb page
        logger = hydra.utils.instantiate(cfg.logger, id=job_id)
        # logger.watch(model, log="all", log_graph=False)
    else:
        logger = None

    # todo: add a wasserstein distance callback
    outdir = Path(cfg.paths.artifacts_dir) / "samples" / job_id

    sample_callback = hydra.utils.instantiate(
        cfg.callbacks.sample,
        outdir=outdir,
        diffusion=diffusion,
        log_to_wandb=not cfg.dryrun,
        sequence_constructor=sequence_constructor,
        structure_constructor=structure_constructor,
    )

    # run training
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=[lr_monitor, checkpoint_callback, sample_callback, ema_callback],
    )
    if rank_zero_only.rank == 0 and isinstance(trainer.logger, WandbLogger):
        trainer.logger.experiment.config.update({"cfg": log_cfg}, allow_val_change=True)

    if not cfg.dryrun:
        trainer.fit(diffusion, datamodule=datamodule)


if __name__ == "__main__":
    train()
