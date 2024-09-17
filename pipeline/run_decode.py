import os
from plaid.pipeline._decode import DecodeLatent
import hydra
from omegaconf import DictConfig

def run(cfg: DictConfig):
    if cfg.npz_path is None:
        timestamps = os.listdir(f"/data/lux70/plaid/artifacts/samples/{cfg.plaid_model_id}/{cfg.cond_code}")
        timestamp = timestamps[-1]
        npz_path = f"/data/lux70/plaid/artifacts/samples/{cfg.plaid_model_id}/{cfg.cond_code}/{timestamp}/latent.npz"
    else:
        npz_path = cfg.npz_path

    """Hydra configurable instantiation, for imports in full pipeline."""
    decode_latent = DecodeLatent(
        npz_path=npz_path,
        output_root_dir=cfg.output_root_dir,
        num_recycles=cfg.num_recycles,
        batch_size=cfg.batch_size,
        device=cfg.device
    )
    decode_latent.run()


@hydra.main(config_path="../configs/pipeline", config_name="decode_latent", version_base=None)
def hydra_run(cfg: DictConfig):
    """Hydra configurable instantiation for running as standalone script."""
    run(cfg)


if __name__ == "__main__":
    hydra_run()