# for wandb sweep: sweep.slrm specifies compute requirements for wandb agent project/agentid
njobs=2
wandb_agent_id="mtvgfeo8"
for i in $(seq 1 $njobs); do
    sbatch /homefs/home/lux70/code/plaid/slurm/sweep_wandb.slrm $wandb_agent_id
done
