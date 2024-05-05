# 256 x 64: h9hzw1bp
# 256 x 32: j1v1wv6w
# 512 x 8: jzlv54wl
# 256 x 8: qjs33lme

python save_compressed_h5.py \
    --compression_model_id "qjs33lme" \
    --compression_model_name "epoch0-step118000-2.495.ckpt" \
    --max_dataset_size -1 \
    --output_dir "/homefs/home/lux70/storage/data/pfam/compressed/subset_full_with_clan/fp16" \
    --max_seq_len 512 \
    --train_split_frac 0.999 \
    --float_type "fp16" \
    --batch_size 128 