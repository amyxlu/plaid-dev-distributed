conda activate omegafold

cd ~/code/plaid/pipeline

root_dir="/data/lux70/plaid/artifacts/natural"

maxlen=$1 # 64
batch_size=$2 #512

# python run_fold.py \
#     --fasta_file ${root_dir}/maxlen${maxlen}/generated/sequences.fasta \
#     --outdir ${root_dir}/maxlen${maxlen}/generated/structures \
#     --max_seq_len $maxlen \
#     --batch_size $batch_size

python consistency_only_pipeline.py \
    ++samples_dir=${root_dir}/maxlen${maxlen} \
    ++inverse_generate_sequence.max_length=$maxlen \
    ++phantom_generate_sequence.max_length=$maxlen \
    ++inverse_generate_structure.batch_size=$batch_size \
    ++inverse_generate_structure.max_seq_len=$maxlen
