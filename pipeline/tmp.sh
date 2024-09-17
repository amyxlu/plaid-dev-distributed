PLAID_MODEL_ID="5j007z42"
SAMPLE_ARTIFACTS_ROOT_DIR="/data/lux70/plaid/artifacts/samples"
NPZ_TIMESTAMP="240907_0658"
COND_CODE="f2219_o3617"

# python /homefs/home/lux70/code/plaid/pipeline/run_fold.py \
#     ++fasta_file=${SAMPLE_ARTIFACTS_ROOT_DIR}/${PLAID_MODEL_ID}/${COND_CODE}/${NPZ_TIMESTAMP}/generated/sequences.fasta \
#     ++outdir=${SAMPLE_ARTIFACTS_ROOT_DIR}/${PLAID_MODEL_ID}/${COND_CODE}/${NPZ_TIMESTAMP}/inverse_generate/structures \
#     ++batch_size=32


micromamba activate omegafold


OUTDIR=${SAMPLE_ARTIFACTS_ROOT_DIR}/${PLAID_MODEL_ID}/${COND_CODE}/${NPZ_TIMESTAMP}/phantom_generate/structures 

if [ ! -d $OUTDIR ]; then
    mkdir -p $OUTDIR
fi

omegafold ${SAMPLE_ARTIFACTS_ROOT_DIR}/${PLAID_MODEL_ID}/${COND_CODE}/${NPZ_TIMESTAMP}/generated/sequences.fasta $OUTDIR --subbatch_size 64
