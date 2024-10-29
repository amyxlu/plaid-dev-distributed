# sbatch run_consistency.slrm experiment=multiflow_consistency 

# for ((len=32; len<=256; len+=4)); do
#     sbatch run_consistency.slrm ++samples_dir=/data/lux70/plaid/artifacts/samples/by_length/$len
# done

sample_dir=/data/lux70/plaid/artifacts/samples/5j007z42/val_dist
for subdir in "$sample_dir"/*/; do
  if [ -d "$subdir" ]; then
    echo $subdir
    sbatch run_consistency.slrm ++samples_dir=$subdir
  fi
done


# maxlen=100
# echo "Running consistency experiments for max_length=$maxlen"
# sbatch run_consistency.slrm \
#     experiment=protpardelle_consistency.yaml \
#     ++samples_dir="/data/lux70/plaid/baselines/protpardelle/samples_large/maxlen${maxlen}" \
#     ++inverse_generate_sequence.max_length=$maxlen \
#     ++phantom_generate_sequence.max_length=$maxlen \
#     ++inverse_generate_structure.max_seq_len=$maxlen \
#     ++inverse_generate_structure.batch_size=128


# maxlen=200
# echo "Running consistency experiments for max_length=$maxlen"
# sbatch run_consistency.slrm \
#     experiment=protpardelle_consistency.yaml \
#     ++samples_dir="/data/lux70/plaid/baselines/protpardelle/samples_large/maxlen${maxlen}" \
#     ++inverse_generate_sequence.max_length=$maxlen \
#     ++phantom_generate_sequence.max_length=$maxlen \
#     ++inverse_generate_structure.max_length=$maxlen \
#     ++inverse_generate_structure.batch_size=64

# maxlen=300
# echo "Running consistency experiments for max_length=$maxlen"
# sbatch run_consistency.slrm \
#     experiment=protpardelle_consistency.yaml \
#     ++samples_dir="/data/lux70/plaid/baselines/protpardelle/samples_large/maxlen${maxlen}" \
#     ++inverse_generate_sequence.max_length=$maxlen \
#     ++phantom_generate_sequence.max_length=$maxlen \
#     ++inverse_generate_structure.max_length=$maxlen \
#     ++inverse_generate_structure.batch_size=64

# maxlen=350
# echo "Running consistency experiments for max_length=$maxlen"
# sbatch run_consistency.slrm \
#     experiment=protpardelle_consistency.yaml \
#     ++samples_dir="/data/lux70/plaid/baselines/protpardelle/samples_large/maxlen${maxlen}" \
#     ++inverse_generate_sequence.max_length=$maxlen \
#     ++phantom_generate_sequence.max_length=$maxlen \
#     ++inverse_generate_structure.max_length=$maxlen \
#     ++inverse_generate_structure.batch_size=32

# maxlen=400
# echo "Running consistency experiments for max_length=$maxlen"
# sbatch run_consistency.slrm \
#     experiment=protpardelle_consistency.yaml \
#     ++samples_dir="/data/lux70/plaid/baselines/protpardelle/samples_large/maxlen${maxlen}" \
#     ++inverse_generate_sequence.max_length=$maxlen \
#     ++phantom_generate_sequence.max_length=$maxlen \
#     ++inverse_generate_structure.max_seq_len=$maxlen \
#     ++inverse_generate_structure.batch_size=32

# maxlen=450
# echo "Running consistency experiments for max_length=$maxlen"
# sbatch run_consistency.slrm \
#     experiment=protpardelle_consistency.yaml \
#     ++samples_dir="/data/lux70/plaid/baselines/protpardelle/samples_large/maxlen${maxlen}" \
#     ++inverse_generate_sequence.max_length=$maxlen \
#     ++phantom_generate_sequence.max_length=$maxlen \
#     ++inverse_generate_structure.max_seq_len=$maxlen \
#     ++inverse_generate_structure.batch_size=16

# maxlen=500
# echo "Running consistency experiments for max_length=$maxlen"
# sbatch run_consistency.slrm \
#     experiment=protpardelle_consistency.yaml \
#     ++samples_dir="/data/lux70/plaid/baselines/protpardelle/samples_large/maxlen${maxlen}" \
#     ++inverse_generate_sequence.max_length=$maxlen \
#     ++phantom_generate_sequence.max_length=$maxlen \
#     ++inverse_generate_structure.max_seq_len=$maxlen \
#     ++inverse_generate_structure.batch_size=8

# maxlen=550
# echo "Running consistency experiments for max_length=$maxlen"
# sbatch run_consistency.slrm \
#     experiment=protpardelle_consistency.yaml \
#     ++samples_dir="/data/lux70/plaid/baselines/protpardelle/samples_large/maxlen${maxlen}" \
#     ++inverse_generate_sequence.max_length=$maxlen \
#     ++phantom_generate_sequence.max_length=$maxlen \
#     ++inverse_generate_structure.max_seq_len=$maxlen \
#     ++inverse_generate_structure.batch_size=8
