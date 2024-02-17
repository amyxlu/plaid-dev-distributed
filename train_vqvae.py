import typing as T
import os

import hydra
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf
import torch

from plaid.proteins import LatentToSequence, LatentToStructure


@hydra.main(version_base=None, config_path="configs", config_name="train_vqvae")
def train(cfg: DictConfig):
    # general set up
    torch.set_float32_matmul_precision("medium")

    def to_load_sequence_constructor(cfg):
        if (cfg.vqvae.sequence_decoder_weight > 0.0):
            return LatentToSequence(temperature=1.0)
        else:
            return None

    def to_load_structure_constructor(cfg):
        if (cfg.vqvae.structure_decoder_weight > 0.0):
            # this creates the esmfold trunk on CPU, without the LM:
            return LatentToStructure()
        else:
            return None

    sequence_constructor = to_load_sequence_constructor(cfg)
    structure_constructor = to_load_structure_constructor(cfg)

    log_cfg = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)
    if rank_zero_only.rank == 0:
        print(OmegaConf.to_yaml(log_cfg))

    # lightning data and model modules
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup(stage="fit")
    latent_scaler = hydra.utils.instantiate(cfg.latent_scaler)
    model = hydra.utils.instantiate(
        cfg.vqvae,
        latent_scaler=latent_scaler,
        sequence_constructor=sequence_constructor,
        structure_constructor=structure_constructor,
    )

    # job_id = os.environ.get("SLURM_JOB_ID")  # is None if not using SLURM
    job_id = None

    if not cfg.dryrun:
        logger = hydra.utils.instantiate(cfg.logger, id=job_id)
        # logger.watch(model, log="all", log_graph=False)
    else:
        logger = None

    # callbacks/home/amyxlu/plaid/plaid/denoisers
    # lr_monitor = hydra.utils.instantiate(cfg.callbacks.lr_monitor)
    checkpoint_callback = hydra.utils.instantiate(cfg.callbacks.checkpoint)
    # run training

    trainer = hydra.utils.instantiate(
#         cfg.trainer, logger=logger, callbacks=[lr_monitor, checkpoint_callback]
        cfg.trainer, logger=logger, callbacks=[checkpoint_callback]
    )
    if rank_zero_only.rank == 0 and isinstance(trainer.logger, WandbLogger):
        trainer.logger.experiment.config.update({"cfg": log_cfg}, allow_val_change=True)

    if not cfg.dryrun:
        trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    train()
