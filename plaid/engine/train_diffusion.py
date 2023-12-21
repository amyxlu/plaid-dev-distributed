import typing as T
import hydra
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.callbacks import LearningRateMonitor, DeviceStatsMonitor
from lightning.pytorch.callbacks import Callback
import torch


def instantiate_sampling_callback(callback_cfg: DictConfig) -> Callback:
    """Instantiates callbacks from config."""
    if not callback_cfg:
        print("[instantiate_callbacks] No callback configs found! Skipping..")
        return None

    if not isinstance(callback_cfg, DictConfig):
        raise TypeError(
            "[instantiate_callbacks] Callbacks config must be a DictConfig!"
        )

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            print(
                f"[instantiate_callbacks] Instantiating callback <{cb_conf._target_}>"
            )
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


@hydra.main(
    version_base=None, config_path="../../configs", config_name="train_diffusion"
)
def train(cfg: DictConfig):
    torch.set_float32_matmul_precision("medium")

    log_cfg = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)
    print(OmegaConf.to_yaml(log_cfg))

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

    if not cfg.dryrun:
        logger = WandbLogger(project="plaid")
        logger.watch(diffusion, log="all", log_freq=500)
    else:
        logger = None

    # callbacks
    lr_monitor = hydra.utils.instantiate(cfg.callbacks.lr_monitor)
    device_stats = hydra.utils.instantiate(cfg.callbacks.device_stats)
    checkpoint_callback = hydra.utils.instantiate(cfg.callbacks.checkpoint)
    sample_callback = hydra.utils.instantiate(
        cfg.callbacks.sample, diffusion=diffusion, model=denoiser
    )

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=[lr_monitor, device_stats, checkpoint_callback, sample_callback],
    )
    if not cfg.dryrun:
        trainer.fit(diffusion, datamodule=datamodule)


if __name__ == "__main__":
    train()
