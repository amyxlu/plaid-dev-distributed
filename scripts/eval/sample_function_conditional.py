import pandas as pd
from omegaconf import OmegaConf, DictConfig

from plaid.utils import round_to_multiple
from plaid.pipeline import SampleLatent

"""
Script configs
"""

# NUM_TO_EVAL = 10

script_config = {
    "return_all_timesteps": True,
    "num_samples": 512
}

cfg = OmegaConf.load("/homefs/home/lux70/code/plaid/configs/pipeline/sample_latent.yaml")
cfg.pop("defaults")

"""
Get unique GO terms in val dataset
"""
df = pd.read_parquet("/data/lux70/data/pfam/val.parquet")
unique_go_idxs = df.GO_idx.unique()
unique_go_idxs.sort()

"""
Loop through
"""
for i, idx in enumerate(unique_go_idxs):
    # if i > NUM_TO_EVAL:
    #     break
    if i < 10:
        continue

    median_len = int(df[df.GO_idx == idx].seq_len.median())
    sample_len = round_to_multiple(median_len / 2, multiple=4)

    for k, v in script_config.items():
        cfg[k] = v

    # for each unique GO term, keep organism as the "unconditional" index and use median length
    cfg.function_idx = int(idx)
    cfg.length = int(sample_len)

    sample_latent = SampleLatent(**cfg)
    sample_latent.run()