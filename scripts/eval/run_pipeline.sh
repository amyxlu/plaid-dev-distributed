len=48
samples_dir=/data/lux70/plaid/artifacts/samples/5j007z42/100_200_300_v2/$len
wandb_job_name=100_200_300_v2_$len

sbatch consistency.slrm

# arg_list=(32 40 48 56 64)
# sbatch run_pipeline.slrm "${arg_list[@]}"
