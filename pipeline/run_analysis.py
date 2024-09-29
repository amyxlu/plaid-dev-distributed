from pathlib import Path
from plaid.pipeline import run_analysis

import os
from plaid.evaluation import RITAPerplexity

import sys

sample_dir = Path(sys.argv[1])
# sample_dir = Path("/data/lux70/plaid/baselines/proteingenerator/100_200_300")
rita_perplexity = RITAPerplexity()
run_analysis(sample_dir, rita_perplexity=rita_perplexity)


# root_sample_dir = Path("/data/lux70/plaid/artifacts/samples/by_length/")

# rita_perplexity = RITAPerplexity()

# for length in os.listdir(root_sample_dir):
#     sample_dir = root_sample_dir / str(length)
#     print("=====================================")
#     print(f"Running analysis on {sample_dir}")
#     print("=====================================")
#     run_analysis(sample_dir, rita_perplexity=rita_perplexity)
