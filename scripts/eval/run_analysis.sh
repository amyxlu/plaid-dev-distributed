#! /bin/bash

sampdir=/data/lux70/plaid/artifacts/samples/scaling

# for ((len=32; len<=256; len+=4)); do
#     sbatch run_analysis.slrm $sampdir/$len
# done

for subdir in "$sampdir"/*/; do
  if [ -d "$subdir" ]; then
    for len in 100 148 48; do
        echo $subdir$len
        sbatch run_analysis.slrm $subdir$len
    done
  fi
done


# for * in sampdir; do
# for len in 100 148 48; do
#     for model_id in 6ryvfi2v 4hdab8dn; do
#         sbatch run_analysis.slrm $sampdir/$model_id/$len
#         sbatch run_analysis.slrm $sampdir/$model_id/$len
#     done
# done