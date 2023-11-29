# CUDA_VISIBLE_DEVICES=7 python train_protein.py \
#     --dataset-config.dataset cath \
#     --dataset-config.path /root/data/cath/shards/ \
#     --artifacts-dir /root/artifacts \
#     --name test_docker

CUDA_VISIBLE_DEVICES=2 python train_protein.py \
    --dataset-config.dataset cath \
    --dataset-config.path /shared/amyxlu/data/cath/shards/ \
    --artifacts-dir /shared/amyxlu/kdplaid \
    --name test_esm2_8M \
    --model-config.lm_embedder_type esm2_t6_8M_UR50D \
    --debug-mode
