import argparse
from pathlib import Path

from plaid.evaluation import foldseek_easysearch, foldseek_easycluster

parser = argparse.ArgumentParser()
parser.add_argument('--samples_dir', default="/data/lux70/plaid/artifacts/samples/5j007z42/val_dist/")
parser.add_argument("--structure_subdir_name", default="None")
parser.add_argument("--no_use_designability_filter", action="store_false")
args = parser.parse_args()

if args.structure_subdir_name == "None":
    if args.no_use_designability_filter:
        subdir_name = "generated/structures" 
    else:
        subdir_name = "designable"
else:
    subdir_name = args.structure_subdir_name

subdir_name = Path(subdir_name)
samples_dir = Path(args.samples_dir)

foldseek_easysearch(samples_dir, subdir_name)
# foldseek_easycluster(samples_dir, subdir_name) 
