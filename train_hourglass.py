import typing as T
import os

import hydra
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf
import torch

from plaid.proteins import LatentToSequence, LatentToStructure


@hydra.main(version_base=None, config_path="configs", config_name="train_hourglass")
def train(cfg: DictConfig):
    # general set up
    torch.set_float32_matmul_precision("medium")

    if cfg.make_structure_constructor:
        structure_constructor = LatentToStructure()
    else:
        structure_constructor = None
    if cfg.make_sequence_constructor:
        sequence_constructor = LatentToSequence(temperature=0.0)
    else:
        sequence_constructor = None

    log_cfg = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)
    if rank_zero_only.rank == 0:
        print(OmegaConf.to_yaml(log_cfg))

    # lightning data and model modules
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup(stage="fit")
    latent_scaler = hydra.utils.instantiate(cfg.latent_scaler)
    model = hydra.utils.instantiate(
        cfg.hourglass,
        latent_scaler=latent_scaler,
        sequence_constructor=sequence_constructor,
        structure_constructor=structure_constructor
    )

    job_id = None
    if not cfg.dryrun:
        logger = hydra.utils.instantiate(cfg.logger, id=job_id)
    else:
        logger = None

    checkpoint_callback = hydra.utils.instantiate(cfg.callbacks.checkpoint)
    trainer = hydra.utils.instantiate(
        cfg.trainer, logger=logger, callbacks=[checkpoint_callback]
    )
    if rank_zero_only.rank == 0 and isinstance(trainer.logger, WandbLogger):
        trainer.logger.experiment.config.update({"cfg": log_cfg}, allow_val_change=True)

    if not cfg.dryrun:
        trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    train()
