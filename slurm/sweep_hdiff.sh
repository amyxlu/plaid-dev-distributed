sbatch train_diffusion.slrm experiment=diffusion/dual_track/pfam_1M ++denoiser.num_blocks=3 ++datamodule.batch_size=4 ++logger.name="pfam_U-TSA_1M_d5_b16"

