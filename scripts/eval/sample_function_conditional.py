import argparse

import pandas as pd
from omegaconf import OmegaConf, DictConfig

from plaid.utils import round_to_multiple
from plaid.pipeline import SampleLatent, DPMSolverSampleLatent

"""
Script configs
"""

parser = argparse.ArgumentParser()
parser.add_argument("--return_all_timesteps", type=bool, default=True)
parser.add_argument("--num_samples", type=int, default=512)
parser.add_argument("--start_idx", type=int, default=-1)
parser.add_argument("--end_idx", type=int, default=-1)
args = parser.parse_args()

# cfg = OmegaConf.load("/homefs/home/lux70/code/plaid/configs/pipeline/sample_latent.yaml")
cfg = OmegaConf.load("/homefs/home/lux70/code/plaid/configs/pipeline/dpm_sample_latent.yaml")
try:
    cfg.pop("defaults")
except KeyError:
    pass

cfg['return_all_timesteps'] = args.return_all_timesteps
cfg['num_samples'] = args.num_samples

"""
Get unique GO terms in val dataset
"""
df = pd.read_parquet("/data/lux70/data/pfam/val.parquet")
unique_go_idxs = df.GO_idx.unique()
unique_go_idxs.sort()

"""
Loop through
"""

start_idx = args.start_idx if args.start_idx != -1 else 0
end_idx = args.end_idx if args.end_idx != -1 else len(unique_go_idxs)

for i, idx in enumerate(unique_go_idxs):
    if start_idx < i < end_idx:
        median_len = int(df[df.GO_idx == idx].seq_len.median())
        sample_len = round_to_multiple(median_len / 2, multiple=4)

        # for each unique GO term, keep organism as the "unconditional" index and use median length
        cfg.function_idx = int(idx)
        cfg.length = int(sample_len)

        # sample_latent = SampleLatent(**cfg)
        sample_latent = DPMSolverSampleLatent(**cfg)
        sample_latent.run()