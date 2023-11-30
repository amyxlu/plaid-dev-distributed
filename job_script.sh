if [ -z "$1" ]; then
    slurm_run () {
        sbatch \
            --account=co_rail --qos=rail_gpu4_normal \
            --partition=savio4_gpu --nodes=1 --ntasks=1 \
            --job-name=train \
            --time=12:00:00 \
            --cpus-per-task=8 \
            --gres=gpu:A100:1 \
            $@
    }
    for embedder in esm2_t6_8M_UR50D esm2_t12_35M_UR50D; do 
        slurm_run "${BASH_SOURCE[0]}" "$embedder"
    done
else
    singularity exec /global/scratch/amyxlu/images/plaid.sif \
        python train_protein.py \
        --dataset-config.dataset cath \
        --dataset-config.path /global/scratch/amyxlu/data/cath/shards/ \
        --artifacts-dir /global/scratch/amyxlu/artifacts/plaid/ \
        --name BRC_$SLURM_JOB_ID \
        --model-config.lm_embedder_type $1
fi
