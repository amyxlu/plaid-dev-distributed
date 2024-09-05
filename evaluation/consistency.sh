COND_CODE=""
NPZ_TIMESTAMP=""
PLAID_MODEL_ID=""

SAMPLE_ARTIFACTS_ROOT_DIR=""


# latent to sequence and structure
python run_decode.py \
    ++npz_path=${SAMPLE_ARTIFACTS_ROOT_DIR}/${PLAID_MODEL_ID}/${COND_CODE}/${NPZ_TIMESTAMP}.npz \
    ++output_root_dir=${SAMPLE_ARTIFACTS_ROOT_DIR}/${PLAID_MODEL_ID}/${COND_CODE}/${NPZ_TIMESTAMP}/generated

# generated structure to inverse-generated sequence
python run_inverse_fold.py \
    ++pdb_dir=${SAMPLE_ARTIFACTS_ROOT_DIR}/${PLAID_MODEL_ID}/${COND_CODE}/${NPZ_TIMESTAMP}/generated \
    ++outdir=${SAMPLE_ARTIFACTS_ROOT_DIR}/${PLAID_MODEL_ID}/${COND_CODE}/${NPZ_TIMESTAMP}/inverse_generate

# generated structure to inverse-generated sequence
omegafold ${SAMPLE_ARTIFACTS_ROOT_DIR}/${PLAID_MODEL_ID}/${COND_CODE}/${NPZ_TIMESTAMP}/generated/sequences.fasta \
    ${SAMPLE_ARTIFACTS_ROOT_DIR}/${PLAID_MODEL_ID}/${COND_CODE}/${NPZ_TIMESTAMP}/inverse_generate/structures
