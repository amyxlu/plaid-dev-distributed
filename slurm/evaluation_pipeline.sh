for len in 48 100 152; do  # 100, 200, 300
    sbatch evaluation_pipeline.slrm experiment=bs32 ++sample.length=100 ++sample.num_samples=128 ++sample.output_root_dir="/data/lux70/plaid/artifacts/samples/by_length"
done

python evaluation_pipeline.py --experiment=bs32 --sample.length=152 --sample.num_samples=128 --sample.output_root_dir="/data/lux70/plaid/artifacts/samples/by_length"
evaluation_pipeline.slrm 

# for len in 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256; do
#     sbatch evaluation_pipeline.slrm experiment=bs16 ++sample.length=$len ++sample.num_samples=32 ++sample.output_root_dir="/data/lux70/plaid/artifacts/samples/by_length"
# done


# cd /homefs/home/lux70/code/plaid

# while IFS= read -r idx; do
#     function_idx="$idx"
#     echo "Processing index: $function_idx"

#     if [[ -z "$idx" ]]; then
#         continue
#     fi

#     sbatch slurm/evaluation_pipeline.slrm ++sample.cond_scale=10. ++sample.sampling_timesteps=1000 ++sample.function_idx=$function_idx

# done < best_idxs.txt


