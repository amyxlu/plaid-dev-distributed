import uuid
import subprocess
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--n_gpus", type=int, default=1)
parser.add_argument("--flags", type=str, default="")
args = parser.parse_args()

flags = ""
if args.n_gpus > 1:
    flags += " ++trainer.device=-1 ++trainer.strategy=ddp_find_unused_parameters_true "
flags += args.flags

defaults = f"""#!/usr/bin/env bash
#SBATCH --job-name train 
#SBATCH --nodes 1 
#SBATCH --gpus-per-node {args.n_gpus} 
#SBATCH --partition gpu2
#SBATCH --cpus-per-gpu 20
#SBATCH --mem 100G
#SBATCH --time=10-00:00:00

eval "$(conda shell.bash hook)"
conda activate plaid 

export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
cd /homefs/home/lux70/code/plaid/

nvidia-smi
srun python train_diffusion.py {flags}
"""

hashid = uuid.uuid4().hex[:7]
slrm_spec_fname = f"tmp-{hashid}.slrm" 

with open(slrm_spec_fname, "w") as f:
    f.write(defaults)

subprocess.run(["sbatch", slrm_spec_fname])
os.remove(slrm_spec_fname)