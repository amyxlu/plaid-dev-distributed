# "esm2_t48_15B_UR50D" "esm2_t36_3B_UR50D" 
# for EMBED in "esm2_t33_650M_UR50D"  "esm2_t30_150M_UR50D"  "esm2_t12_35M_UR50D"  "esm2_t6_8M_UR50D" "esmfold"; do
#     CUDA_VISIBLE_DEVICES=2 python embed_stats_for_normalization.py --n_val 5000 --lm_embedder_type $EMBED
# done
CUDA_VISIBLE_DEVICES=2 python embed_stats_for_normalization.py --n_val 5000 --lm_embedder_type esmfold