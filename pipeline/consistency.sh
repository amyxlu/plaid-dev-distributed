COND_CODE="f1_o3617"
NPZ_TIMESTAMP="240911_2010"
PLAID_MODEL_ID="5j007z42"

MAX_SEQ_LEN=256

SAMPLE_ARTIFACTS_ROOT_DIR="/data/lux70/plaid/artifacts/samples"


python run_decode.py \
    ++npz_path=${SAMPLE_ARTIFACTS_ROOT_DIR}/${PLAID_MODEL_ID}/${COND_CODE}/${NPZ_TIMESTAMP}/latent.npz \
    ++output_root_dir=${SAMPLE_ARTIFACTS_ROOT_DIR}/${PLAID_MODEL_ID}/${COND_CODE}/${NPZ_TIMESTAMP}/generated \
    ++max_seq_len=${MAX_SEQ_LEN}

# generated structure to inverse-generated sequence
python run_inverse_fold.py \
    ++pdb_dir=${SAMPLE_ARTIFACTS_ROOT_DIR}/${PLAID_MODEL_ID}/${COND_CODE}/${NPZ_TIMESTAMP}/generated \
    ++outdir=${SAMPLE_ARTIFACTS_ROOT_DIR}/${PLAID_MODEL_ID}/${COND_CODE}/${NPZ_TIMESTAMP}/inverse_generate

# generated structure to inverse-generated sequence
# omegafold ${SAMPLE_ARTIFACTS_ROOT_DIR}/${PLAID_MODEL_ID}/${COND_CODE}/${NPZ_TIMESTAMP}/generated/sequences.fasta \
#     ${SAMPLE_ARTIFACTS_ROOT_DIR}/${PLAID_MODEL_ID}/${COND_CODE}/${NPZ_TIMESTAMP}/inverse_generate/structures \
#     --subbatch_size 64
