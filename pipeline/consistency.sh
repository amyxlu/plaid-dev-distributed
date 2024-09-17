PLAID_MODEL_ID="5j007z42"
SAMPLE_ARTIFACTS_ROOT_DIR="/data/lux70/plaid/artifacts/samples"

# f322_o3617 f362_o3617 f263_o3617 
# f319_o3617 f269_o3617 f327_o3617 
# f308_o3617 f349_o3617 f356_o3617 f307_o3617

# for COND_CODE in f322_o3617 f362_o3617 f263_o3617 f319_o3617 f269_o3617 f327_o3617 f308_o3617 f349_o3617 f356_o3617 f307_o3617; do
# for COND_CODE in f308_o3617 f349_o3617 f356_o3617 f307_o3617; do

cd ${SAMPLE_ARTIFACTS_ROOT_DIR}/${PLAID_MODEL_ID}/

condcodes=($(find . -maxdepth 1 -type d -name 'f*o*' -not -path .))

for COND_CODE in "${condcodes[@]}"; do
    COND_CODE=${COND_CODE#./}
    echo $COND_CODE

    cd ${SAMPLE_ARTIFACTS_ROOT_DIR}/${PLAID_MODEL_ID}/${COND_CODE}/

    # just do get some cleaning out of the way too
    rm -rf outputs

    # Get a list of subdirectories in the current directory
    subdirs=($(find . -maxdepth 1 -type d -name '24*' -not -path .))

    # Check the number of subdirectories
    if [ ${#subdirs[@]} -eq 1 ]; then
        # Only one subdirectory; set TIMESTAMP to its name
        NPZ_TIMESTAMP="${subdirs[0]#./}"
    else
        # More than one subdirectory; find the most recently created one
        NPZ_TIMESTAMP=$(ls -dt "${subdirs[@]}" | tail -n 1)
        NPZ_TIMESTAMP="${NPZ_TIMESTAMP#./}"
    fi

    echo NPZ_TIMESTAMP is set to: $NPZ_TIMESTAMP

    python /homefs/home/lux70/code/plaid/pipeline/run_decode.py \
        ++npz_path=${SAMPLE_ARTIFACTS_ROOT_DIR}/${PLAID_MODEL_ID}/${COND_CODE}/${NPZ_TIMESTAMP}/latent.npz \
        ++output_root_dir=${SAMPLE_ARTIFACTS_ROOT_DIR}/${PLAID_MODEL_ID}/${COND_CODE}/${NPZ_TIMESTAMP}/generated

    # generated structure to inverse-generated sequence
    python /homefs/home/lux70/code/plaid/pipeline/run_inverse_fold.py \
        ++pdb_dir=${SAMPLE_ARTIFACTS_ROOT_DIR}/${PLAID_MODEL_ID}/${COND_CODE}/${NPZ_TIMESTAMP}/generated \
        ++outdir=${SAMPLE_ARTIFACTS_ROOT_DIR}/${PLAID_MODEL_ID}/${COND_CODE}/${NPZ_TIMESTAMP}/inverse_generate

    # generated structure to inverse-generated sequence
    # omegafold ${SAMPLE_ARTIFACTS_ROOT_DIR}/${PLAID_MODEL_ID}/${COND_CODE}/${NPZ_TIMESTAMP}/generated/sequences.fasta \
    #     ${SAMPLE_ARTIFACTS_ROOT_DIR}/${PLAID_MODEL_ID}/${COND_CODE}/${NPZ_TIMESTAMP}/inverse_generate/structures \
    #     --subbatch_size 64

    python /homefs/home/lux70/code/plaid/pipeline/run_fold.py \
        ++fasta_file=${SAMPLE_ARTIFACTS_ROOT_DIR}/${PLAID_MODEL_ID}/${COND_CODE}/${NPZ_TIMESTAMP}/generated/sequences.fasta \
        ++outdir=${SAMPLE_ARTIFACTS_ROOT_DIR}/${PLAID_MODEL_ID}/${COND_CODE}/${NPZ_TIMESTAMP}/inverse_generate/structures \
        ++batch_size=32
done

