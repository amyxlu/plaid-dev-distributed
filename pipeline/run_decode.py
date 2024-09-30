import os
from pathlib import Path
from plaid.pipeline._decode import DecodeLatent
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../configs/pipeline/decode", config_name="default")
def hydra_run(cfg: DictConfig):
    """Hydra configurable instantiation for running"""
    npz_path = Path(cfg.npz_path)
    output_root_dir = npz_path.parent
    decode_latent = hydra.utils.instantiate(cfg, output_root_dir=output_root_dir)
    decode_latent.run()


if __name__ == "__main__":
    hydra_run()