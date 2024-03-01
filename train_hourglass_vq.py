import typing as T
import os

import hydra
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf
import torch

from plaid.proteins import LatentToSequence, LatentToStructure
from plaid.transforms import ESMFoldEmbed
from plaid.datasets import FastaDataModule


@hydra.main(version_base=None, config_path="configs", config_name="train_hourglass_vq")
def train(cfg: DictConfig):
    # general set up
    torch.set_float32_matmul_precision("medium")
    log_cfg = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)
    if rank_zero_only.rank == 0:
        print(OmegaConf.to_yaml(log_cfg))

    # lightning data modules, scalar, and maybe sequence embedder
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup(stage="fit")

    latent_scaler = hydra.utils.instantiate(cfg.latent_scaler)
    if isinstance(datamodule, FastaDataModule):
        seq_emb_fn = ESMFoldEmbed(shorten_len_to=cfg.datamodule.seq_len)
    else:
        seq_emb_fn = None

    model = hydra.utils.instantiate(
        cfg.hourglass,
        latent_scaler=latent_scaler,
        seq_emb_fn=seq_emb_fn
    )

    job_id = None
    if not cfg.dryrun:
        logger = hydra.utils.instantiate(cfg.logger, id=job_id)
    else:
        logger = None

    checkpoint_callback = hydra.utils.instantiate(cfg.callbacks.checkpoint)
    lr_monitor = hydra.utils.instantiate(cfg.callbacks.lr_monitor)
    trainer = hydra.utils.instantiate(
        cfg.trainer, logger=logger, callbacks=[checkpoint_callback, lr_monitor]
    )
    if rank_zero_only.rank == 0 and isinstance(trainer.logger, WandbLogger):
        trainer.logger.experiment.config.update({"cfg": log_cfg}, allow_val_change=True)

    if not cfg.dryrun:
        trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    train()
