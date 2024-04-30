# 256 x 64: h9hzw1bp
# 256 x 32: j1v1wv6w
# 512 x 8: jzlv54wl

python save_compressed_h5.py \
    --compression_model_id "j1v1wv6w" \
    --compression_model_name "last.ckpt" \
    --max_dataset_size 30000 \
    --output_dir "/homefs/home/lux70/storage/data/pfam/compressed/subset_30K_with_clan" \
    --max_seq_len 512 \
    --train_split_frac 0.9 \
    --float_type "fp16" \
    --batch_size 64 