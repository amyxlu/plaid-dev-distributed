CUDA_VISIBLE_DEVICES=7 python train_protein.py \
    --dataset-config.dataset cath \
    --dataset-config.path /root/data/cath/shards/ \
    --artifacts-dir /root/artifacts \
    --name test_docker