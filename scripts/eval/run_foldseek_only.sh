############### By length ############### 
sampdir=/data/lux70/plaid/artifacts/samples/5j007z42/scratch_by_length
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

############### ProtPardelle ############### 
sampdir=/data/lux70/plaid/baselines/protpardelle/samples_by_length
for subdir in "$sampdir"/*/; do
  if [ -d "$subdir" ]; then
    if [ ! -d "$subdir$len/no_filter_foldseek_easycluster.m8_rep_seq.fasta" ]; then
        echo $subdir$len
        sbatch run_foldseek_only.slrm $subdir$len
    fi
  fi
done

############### Multiflow ############### 
sampdir=/data/lux70/plaid/baselines/multiflow/skip8_64per
for subdir in "$sampdir"/*/; do
  if [ -d "$subdir" ]; then
    if [ ! -d "$subdir$len/no_filter_foldseek_easycluster.m8_rep_seq.fasta" ]; then
        echo $subdir$len
        sbatch run_foldseek_only.slrm $subdir$len
    fi
  fi
done
