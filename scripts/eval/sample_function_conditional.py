import argparse

import pandas as pd
from omegaconf import OmegaConf

from plaid.utils import round_to_multiple
from plaid.pipeline import SampleLatent, DecodeLatent

"""
Script configs
"""

parser = argparse.ArgumentParser()
parser.add_argument("--start_idx", type=int, default=-1)
parser.add_argument("--end_idx", type=int, default=-1)
parser.add_argument("--GO_term_contains", type=str, default=None)
args = parser.parse_args()

# cfg = OmegaConf.load("/homefs/home/lux70/code/plaid/configs/pipeline/sample_latent_config/sample_latent.yaml")
sample_cfg = OmegaConf.load("/homefs/home/lux70/code/plaid/configs/pipeline/sample_latent_config/dpm_sample_latent.yaml")
decode_cfg = OmegaConf.load("/homefs/home/lux70/code/plaid/configs/pipeline/decode_latent.yaml")

try:
    sample_cfg.pop("defaults")
    decode_cfg.pop("defaults")
except KeyError:
    pass

"""
Get unique GO terms in val dataset
"""
df = pd.read_parquet("/data/lux70/data/pfam/val.parquet")

if args.GO_term_contains is not None:
    df = df[df.GO_term.str.contains(args.GO_term_contains)]

print("Unique GO terms in val dataset:", df.GO_idx.unique())

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
        sample_cfg.function_idx = int(idx)
        sample_cfg.length = int(sample_len)

        sample_latent = SampleLatent(**sample_cfg)
        npz_path = sample_latent.run()

        # Decode the latent
        try:
            decode_cfg.pop("npz_path", None)
            decode_cfg.pop("output_root_dir", None)
        except:
            pass

        decode_latent = DecodeLatent(npz_path=npz_path, **decode_cfg)
        decode_latent.run()