import time
from pathlib import Path

import wandb
import hydra
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf
import torch


@hydra.main(version_base=None, config_path="configs", config_name="train_causal")
def train(cfg: DictConfig):
    # general set up
    torch.set_float32_matmul_precision("medium")
    log_cfg = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)
    if rank_zero_only.rank == 0:
        print(OmegaConf.to_yaml(log_cfg))

    # token module from saved embeddings
    start = time.time()
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup(stage="fit")
    end = time.time()
    print(f"Datamodule set up in {end - start:.2f} seconds.")

    model = hydra.utils.instantiate(cfg.causal_model)
    job_id = wandb.util.generate_id()
    print("job id:", job_id)

    if not cfg.dryrun:
        logger = hydra.utils.instantiate(cfg.logger, id=job_id)
    else:
        logger = None

    # callback options
    dirpath = Path(cfg.paths.checkpoint_dir) / "causal" / job_id
    checkpoint_callback = hydra.utils.instantiate(cfg.callbacks.checkpoint, dirpath=dirpath)
    lr_monitor = hydra.utils.instantiate(cfg.callbacks.lr_monitor)

    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=[checkpoint_callback, lr_monitor])

    if rank_zero_only.rank == 0 and isinstance(trainer.logger, WandbLogger):
        trainer.logger.experiment.config.update({"cfg": log_cfg}, allow_val_change=True)

    if not cfg.dryrun:
        trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    train()
