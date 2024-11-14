#! /bin/bash

############### Scaling ############### 
# sampdir=/data/lux70/plaid/artifacts/samples/scaling/6ryvfi2v/

# for len in 100 148 48; do
#     echo $sampdir$len
#     sbatch run_analysis.slrm $sampdir$len 
# done


############### By length ############### 
# sampdir=/data/lux70/plaid/artifacts/samples/5j007z42/scratch_by_length
# for subdir in "$sampdir"/*/; do
#   if [ -d "$subdir" ]; then
#     echo $subdir$len
#     sbatch run_analysis.slrm $subdir$len
#   fi
# done

############### ProteinGenerator############### 
sbatch run_analysis.slrm /data/lux70/plaid/baselines/proteingenerator/by_length/

############### ProtPardelle ############### 
# sampdir=/data/lux70/plaid/baselines/protpardelle/samples_by_length
# for subdir in "$sampdir"/*/; do
#   if [ -d "$subdir" ]; then
#     echo $subdir$len
#     sbatch run_analysis.slrm $subdir$len
#   fi
# done

# ############## Multiflow ############### 

# ############## Natural############### 
# sampdir=/data/lux70/plaid/artifacts/natural
# for subdir in "$sampdir"/*/; do
#   if [ -d "$subdir" ]; then
#     echo $subdir$len
#     sbatch run_analysis.slrm $subdir$len
#   fi
# done