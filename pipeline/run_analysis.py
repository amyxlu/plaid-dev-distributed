from pathlib import Path
from plaid.pipeline import run_analysis

import os
from plaid.evaluation import RITAPerplexity

sample_dir = Path("/data/lux70/plaid/artifacts/samples/5j007z42/by_length/")
root_sample_dir = Path("/data/lux70/plaid/artifacts/samples/by_length/")

rita_perplexity = RITAPerplexity()

for length in os.listdir(root_sample_dir):
    sample_dir = root_sample_dir / str(length)
    print("=====================================")
    print(f"Running analysis on {sample_dir}")
    print("=====================================")
    run_analysis(sample_dir, rita_perplexity=rita_perplexity)
