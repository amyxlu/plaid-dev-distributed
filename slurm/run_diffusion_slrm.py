import uuid
import subprocess
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--n_gpus_per_node", type=int, default=1)
parser.add_argument("-n", "--n_nodes", type=int, default=1)
parser.add_argument("-c", "--n_cpus_per_task", type=int, default=12)
parser.add_argument("-f", "--flags", type=str, default="")
args = parser.parse_args()

flags = ""

flags += args.flags

defaults = f"""#!/usr/bin/env bash
#SBATCH --partition gpu2
#SBATCH --nodes {args.n_nodes} 
#SBATCH --ntasks-per-node {args.n_gpus_per_node} 
#SBATCH --gpus-per-node {args.n_gpus_per_node}
#SBATCH --cpus-per-task {args.n_cpus_per_task}
#SBATCH --time=10-00:00:00
#SBATCH --job-name train 

source ~/.bashrc
micromamba activate plaid

export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=TRACE
cd /homefs/home/lux70/code/plaid/

nvidia-smi

srun python train_compositional.py {flags} \
    ++trainer.devices=$SLURM_GPUS_PER_NODE \
    ++trainer.num_nodes=$SLURM_JOB_NUM_NODES
"""

hashid = uuid.uuid4().hex[:7]
slrm_spec_fname = f"tmp-{hashid}.slrm"

with open(slrm_spec_fname, "w") as f:
    f.write(defaults)

subprocess.run(["sbatch", slrm_spec_fname])
os.remove(slrm_spec_fname)
