# sbatch train_diffusion.slrm experiment=diffusion/dual_track/pfam_1M ++denoiser.num_blocks=3 ++datamodule.batch_size=4 ++logger.name="pfam_U-TSA_1M_d5_b16"
# sbatch train_diffusion.slrm experiment=diffusion/dit/pfam_udit ++denoiser.depth=7 ++logger.name="pfam_1M_udit_d7"
# sbatch train_diffusion.slrm experiment=diffusion/dit/pfam_udit_toy ++logger.name="pfam_s5000_udit_sanity"
# sbatch train_diffusion.slrm experiment=diffusion/dit/pfam_dit ++logger.name="pfam_s1M_small" ++datamodule.batch_size=256
sbatch train_diffusion.slrm experiment=diffusion/dit/pfam_dit_base ++logger.name="pfam_s1M_base" ++compression_model_id="jzlv54wl" ++datamodule.batch_size=128

