# for wandb sweep: sweep.slrm specifies compute requirements for wandb agent project/agentid
# njobs=4
# for i in $(seq 1 $njobs); do
#     sbatch sweep.slrm
# done

sbatch train_diffusion.slrm