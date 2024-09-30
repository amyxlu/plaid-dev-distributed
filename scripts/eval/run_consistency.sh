for ((len=176; len<=256; len+=4)); do
    sbatch run_consistency.slrm \
    ++samples_dir="/data/lux70/plaid/artifacts/samples/by_length" \
    ++inverse_generate_sequence.max_length=$len \
    ++phantom_generate_sequence.max_length=$len \
    ++inverse_generate_structure.max_seq_len=$len \
    ++inverse_generate_structure.batch_size=8
done
# maxlen=100
# echo "Running consistency experiments for max_length=$maxlen"
# sbatch run_consistency.slrm \
#     experiment=protpardelle_consistency.yaml \
#     ++samples_dir="/data/lux70/plaid/baselines/protpardelle/samples_large/maxlen${maxlen}" \
#     ++inverse_generate_sequence.max_length=$maxlen \
#     ++phantom_generate_sequence.max_length=$maxlen \
#     ++inverse_generate_structure.max_seq_len=$maxlen \
#     ++inverse_generate_structure.batch_size=256


# maxlen=200
# echo "Running consistency experiments for max_length=$maxlen"
# sbatch run_consistency.slrm \
#     experiment=protpardelle_consistency.yaml \
#     ++samples_dir="/data/lux70/plaid/baselines/protpardelle/samples_large/maxlen${maxlen}" \
#     ++inverse_generate_sequence.max_length=$maxlen \
#     ++phantom_generate_sequence.max_length=$maxlen \
#     ++inverse_generate_structure.max_length=$maxlen \
#     ++inverse_generate_structure.batch_size=128

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
#     ++inverse_generate_structure.batch_size=64

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
