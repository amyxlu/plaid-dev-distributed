import uuid
import subprocess
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--n_gpus", type=int, default=1)
parser.add_argument("--n_nodes", type=int, default=1)
parser.add_argument("--n_cpus", type=int, default=16)
parser.add_argument("--flags", type=str, default="")
args = parser.parse_args()

flags = ""
flags += args.flags

defaults = f"""#!/usr/bin/env bash
#SBATCH --nodes {args.n_nodes} 
#SBATCH --gpus {args.n_gpus}
#SBATCH --ntasks-per-node {args.n_gpus // args.n_nodes}
#SBATCH --mem-per-cpu=8G
#SBATCH -p gpu2
#SBATCH -c {args.n_cpus} 
#SBATCH --time=10-00:00:00
#SBATCH --job-name train 

eval "$(conda shell.bash hook)"
conda activate plaid 

export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=INFO
cd /homefs/home/lux70/code/plaid/

nvidia-smi
srun python train_compositional.py {flags}
"""

hashid = uuid.uuid4().hex[:7]
slrm_spec_fname = f"tmp-{hashid}.slrm"

with open(slrm_spec_fname, "w") as f:
    f.write(defaults)

subprocess.run(["sbatch", slrm_spec_fname])
os.remove(slrm_spec_fname)
