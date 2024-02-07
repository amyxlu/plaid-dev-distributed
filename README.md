# k-diffusion
# PLAID (Protein LAtent Induced Diffusion)

To run a wandb sweep, from the project directory:
```
wandb sweep sweeps/my_sweep.yml   # this will launch the sweep agent; note the sweep ID for reference.
bash slurm/agent_sweep_diffusion.sh  # this launches a series of `sbatch` commands that then launches a wandb agent in each
```

Alternatively, one can also launch an agent (i.e. individual run as configured by a sweep) locally via:
```
wandb agent entity/project/sweep_id
```

# Development Notes
## Useful Commands
`find . -type f -mtime +XXX -exec rm {} \;`
`find . -type f -mtime +4`


To run an image that mounts locally (on DGX5) as root user:
```
docker run -it --net=host --user root --mount type=bind,source="/home/amyxlu/kdiffusion",target=/mnt/kdiffusion mambaorg/micromamba:jammy-cuda-12.2.0 /bin/bash

# if nvidia-container-toolkit is installed
docker run -it --name mambacuda --gpus all --net=host --user root --mount type=bind,source="/home/amyxlu/kdiffusion",target=/mnt/kdiffusion mambaorg/micromamba:jammy-cuda-12.2.0 /bin/bash
```
Replace `source="..."` with the source directory on your local machine.

This base image will use `micromamba` instead of `conda`, but most common conda commands can be used by directly swapping out `conda` for `micromamba`.

```
pip install "fair-esm[esmfold]"
# OpenFold and its remaining dependency
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'
```

## Other Notes
* `https://github.com/amyxlu/plaid-dev/commit/6ac51a838a172a8761e02dbe98fc0b0eb439fb61#diff-0da7caefdb1257895597fa2dfd7c4c9380642739bb740c242bea69d81815b54c` dropped list of PDB ids calculated in prev. project