import hydra
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    version_base=None, config_path="../../configs", config_name="train_diffusion"
)
def train(cfg: DictConfig):
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
    else:
        logger = None

    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    if not cfg.dryrun:
        trainer.fit(diffusion, datamodule=datamodule)


if __name__ == "__main__":
    train()
