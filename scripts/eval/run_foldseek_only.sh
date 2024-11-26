############### By length ############### 
# sampdir=/data/lux70/plaid/artifacts/samples/5j007z42/by_length
# for subdir in "$sampdir"/*/; do
#   if [ -d "$subdir" ]; then
#     if [ ! -d "$subdir$len/no_filter_foldseek_easycluster.m8_rep_seq.fasta" ]; then
#         echo $subdir$len
#         sbatch run_foldseek_only.slrm $subdir$len
#     fi
#   fi
# done

# fix mmseqs thing
# sampdir=/data/lux70/plaid/artifacts/samples/5j007z42/by_length
# for ((len=176; len<=256; len+=4)); do
#   echo $sampdir/$len
#   sbatch run_mmseqs_only.slrm $sampdir/$len
# done

############### shorter ###############

sampdir=/data/lux70/plaid/artifacts/samples/ksme77o6/by_length
for subdir in "$sampdir"/*/; do
  if [ -d "$subdir" ]; then
    if [ ! -d "$subdir$len/no_filter_foldseek_easycluster.m8_rep_seq.fasta" ]; then
        echo $subdir$len
        sbatch run_foldseek_only.slrm $subdir$len
    fi
  fi
done

############### ProteinGenerator############### 
# sbatch run_foldseek_only.slrm /data/lux70/plaid/baselines/proteingenerator/by_length/
# sampdir="/data/lux70/plaid/artifacts/samples/5j007z42/conditional/cond_scale"

# for subdir in "$sampdir"/*/; do
#   if [ -d "$subdir" ]; then
#       echo $subdir$len
#       sbatch run_foldseek_only.slrm $subdir$len
#       sbatch run_mmseqs_only.slrm $subdir$len
#       sbatch run_foldseek_only.slrm $subdir$len --use_designability_filter
#   fi
# done

############### ProtPardelle ############### 
# sampdir=/data/lux70/plaid/baselines/protpardelle/samples_by_length
# for subdir in "$sampdir"/*/; do
#   if [ -d "$subdir" ]; then
#       echo $subdir$len
#       # sbatch run_foldseek_only.slrm $subdir$len
#       sbatch run_foldseek_only.slrm $subdir$len --use_designability_filter
#   fi
# done

# sbatch run_foldseek_only.slrm /data/lux70/plaid/baselines/protpardelle/samples_by_length/maxlen450 --use_designability_filter
# sbatch run_mmseqs_only.slrm /data/lux70/plaid/baselines/protpardelle/samples_by_length/maxlen450

# ############### Multiflow ############### 
# sampdir=/data/lux70/plaid/baselines/multiflow/skip8_64per
# for subdir in "$sampdir"/*/; do
#   if [ -d "$subdir" ]; then
#     echo $subdir$len
#     sbatch run_mmseqs_only.slrm $subdir$len
#   fi
# done

############### Natural ############### 
# sampdir=/data/lux70/plaid/artifacts/natural_binned_lengths
# for subdir in "$sampdir"/*/; do
#   if [ -d "$subdir" ]; then
#     echo $subdir$len
#     sbatch run_foldseek_only.slrm $subdir$len
#     sbatch run_mmseqs_only.slrm $subdir$len
#     sbatch run_foldseek_only.slrm $subdir$len --use_designability_filter
#   fi
# done

# sampdir='/data/lux70/plaid/artifacts/natural_binned_lengths/binstart424/'
# sbatch run_foldseek_only.slrm $sampdir
