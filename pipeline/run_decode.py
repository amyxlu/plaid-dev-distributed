import os
from pathlib import Path
from plaid.pipeline._decode import DecodeLatent
import hydra
from omegaconf import DictConfig, OmegaConf

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64)
args = parser.parse_args()

# cfg = OmegaConf.load("/homefs/home/lux70/code/plaid/configs/pipeline/decode/default.yaml")

def run(npz_path):
    """Hydra configurable instantiation, for imports in full pipeline."""
    npz_path = Path(npz_path)
    output_root_dir = npz_path.parent

    decode_latent = DecodeLatent(
        npz_path=npz_path,
        output_root_dir=output_root_dir,
        num_recycles=4,
        batch_size=args.batch_size,
        device="cuda"
    )
    decode_latent.run()


if __name__ == "__main__":
    from pathlib import Path
    sample_dir = Path("/data/lux70/plaid/artifacts/samples/5j007z42/ddim/5j007z42")


    import os
    for cond_code in os.listdir(sample_dir):
        for timestamp in os.listdir(sample_dir / cond_code):
            root_dir = sample_dir / cond_code / timestamp
            if not "generated" in os.listdir(root_dir):
                npz_path = root_dir / "latent.npz"
                print(npz_path)

                run(npz_path)

