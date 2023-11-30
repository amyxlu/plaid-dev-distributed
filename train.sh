# CUDA_VISIBLE_DEVICES=7 python train_protein.py \
#     --dataset-config.dataset cath \
#     --dataset-config.path /root/data/cath/shards/ \
#     --artifacts-dir /root/artifacts \
#     --name test_docker

for embedder in esm2_t6_8M_UR50D esm2_t12_35M_UR50D esm2_t30_150M_UR50D; do
    CUDA_VISIBLE_DEVICES=2 python train_protein.py \
        --dataset-config.dataset cath \
        --dataset-config.path /shared/amyxlu/data/cath/shards/ \
        --artifacts-dir /shared/amyxlu/kdplaid \
        --name 231128_$embedder \
        --model-config.lm_embedder_type $embedder &
done
