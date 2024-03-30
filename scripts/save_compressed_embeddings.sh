python save_compressed_embeddings.py \
    --compressor_model_id "2024-03-30T00-32-11" \
    --fasta_file /homefs/home/lux70/storage/data/rocklin/rocklin_stable.fasta \
    --base_output_dir "/homefs/home/lux70/storage/data/rocklin/compressed/" \
    --batch_size 256 \
    --max_seq_len 512 \
    --num_workers 16
