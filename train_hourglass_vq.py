import time
from pathlib import Path

import wandb
import hydra
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf
import torch

from plaid.transforms import ESMFoldEmbed
from plaid.datasets import FastaDataModule


@hydra.main(version_base=None, config_path="configs", config_name="train_hourglass_vq")
def train(cfg: DictConfig):
    # general set up
    torch.set_float32_matmul_precision("medium")
    log_cfg = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)
    if rank_zero_only.rank == 0:
        print(OmegaConf.to_yaml(log_cfg))

    # lightning data modules
    start = time.time()
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup(stage="fit")
    end = time.time()
    print(f"Datamodule set up in {end - start:.2f} seconds.")

    # normalize by channel
    latent_scaler = hydra.utils.instantiate(cfg.latent_scaler)
    if isinstance(datamodule, FastaDataModule):
        seq_emb_fn = ESMFoldEmbed(shorten_len_to=cfg.datamodule.seq_len)
    else:
        seq_emb_fn = None

    # set up lightning module
    model = hydra.utils.instantiate(
        cfg.hourglass,
        latent_scaler=latent_scaler,
        seq_emb_fn=seq_emb_fn
    )
    
    if cfg.resume_from_model_id is not None:
        job_id = cfg.resume_from_model_id 
    else:
        job_id = wandb.util.generate_id() 
    
    if not cfg.dryrun:
        # this will automatically log to the same wandb page
        logger = hydra.utils.instantiate(cfg.logger, id=job_id)
        # logger.watch(model, log="all", log_graph=False)
    else:
        logger = None

    # callback options
    dirpath = Path(cfg.paths.checkpoint_dir) / "hourglass_vq" / job_id
    dirpath.mkdir(parents=False)
    config_path = dirpath / "config.yaml"
    
    if not config_path.exists():
        OmegaConf.save(cfg, config_path)

    checkpoint_callback = hydra.utils.instantiate(cfg.callbacks.checkpoint, dirpath=dirpath)
    lr_monitor = hydra.utils.instantiate(cfg.callbacks.lr_monitor)
    compression_callback = hydra.utils.instantiate(cfg.callbacks.compression)  # creates ESMFold on CPU

    trainer = hydra.utils.instantiate(
        cfg.trainer, logger=logger, callbacks=[checkpoint_callback, lr_monitor, compression_callback]
    )

    if rank_zero_only.rank == 0 and isinstance(trainer.logger, WandbLogger):
        trainer.logger.experiment.config.update({"cfg": log_cfg}, allow_val_change=True)

    if not cfg.dryrun:
        if cfg.resume_from_model_id is None:
            trainer.fit(model, datamodule=datamodule)
        else:
            # job id / dirpath was already updated to match the to-be-resumed directory 
            trainer.fit(model, datamodule=datamodule, ckpt_path=dirpath / "last.ckpt")

if __name__ == "__main__":
    train()
