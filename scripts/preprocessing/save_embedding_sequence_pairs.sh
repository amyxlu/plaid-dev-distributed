python save_embedding_sequence_pairs.py --batch_size 128 --lm_embedder_type "esmfold" --max_seq_len 512 --max_num_samples 31885 \
  --fasta_file /homefs/home/lux70/storage/data/uniref50/uniref50.fasta \
  --train_output_dir /homefs/home/lux70/storage/data/uniref50/subset_shards/train \
  --val_output_dir /homefs/home/lux70/storage/data/uniref50/subset_shards/val
  # --fasta_file /homefs/home/lux70/storage/data/cath/cath-dataset-nonredundant-S40.atom.fa \
  # --train_output_dir /homefs/home/lux70/storage/data/cath/shards/train \
  # --val_output_dir /homefs/home/lux70/storage/data/cath/shards/val
  # --fasta_file /homefs/home/lux70/storage/data/rocklin/rocklin_stable.fasta \
  # --train_output_dir /homefs/home/lux70/storage/data/rocklin/shards/train \
  # --val_output_dir /homefs/home/lux70/storage/data/rocklin/shards/val \
