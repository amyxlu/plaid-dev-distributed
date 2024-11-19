# arg_list=(32 40 48 56 64)
# sbatch run_pipeline.slrm "${arg_list[@]}"

# cd /homefs/home/lux70/code/plaid/scripts
# for len in 100 200 300 350 400 450 500 550; do
#     python structure_dir_to_fasta.py -p /data/lux70/plaid/baselines/protpardelle/samples_large/maxlen${len}/generated/structures
# done

# for ((len=32; len<=175; len+=4)); do
#     sbatch run_pipeline.slrm \
#     ++sample.length=$len \
#     ++sample.output_root_dir="/data/lux70/plaid/artifacts/samples/by_length/$len"
#     ++log_to_wandb=False
# done

##############################
# Timestep
##############################

sbatch run_pipeline.slrm 




##############################
#  
##############################