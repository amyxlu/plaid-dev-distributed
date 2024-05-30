# for id in e2iwuio0 t1ji0n45 0u6sw28c 5006fhbi 6nahhz22 ftqrhze9 xppmuppx gzkug3bi; do
#     sbatch train_diffusion.slrm resume_from_model_id=$id
# done

sbatch train_diffusion_2gpu.slrm experiment=hdif/pfam_dit datamodule.h5_root_dir=/homefs/home/lux70/storage/data/pfam/compressed/subset_30K_with_clan ++datamodule.batch_size=256 ++datamodule.aprclan_version=True ++denoiser.depth=6