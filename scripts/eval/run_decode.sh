# for ((len=176; len<=256; len+=4)); do
for ((len=32; len<=175; len+=4)); do
    sbatch run_decode.slrm \
    ++npz_path=/data/lux70/plaid/artifacts/samples/by_length/$len/latent.npz \
    ++batch_size=8
done