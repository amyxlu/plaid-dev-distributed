sbatch train_diffusion.slrm experiment=hdif/pfam_dit ++datamodule.num_workers=16 
sbatch train_diffusion.slrm experiment=hdif/pfam_dit ++datamodule.num_workers=16 ++diffusion.sequence_decoder_weight=1.0 
# sbatch train_diffusion.slrm experiment=hdif/pfam_dit ++datamodule.num_workers=16 ++diffusion.sequence_decoder_weight=1.0 ++denoiser.use_self_conditioning=True 
sbatch train_diffusion.slrm experiment=hdif/pfam_dit ++datamodule.num_workers=16 ++diffusion.sequence_decoder_weight=1.0 ++diffusion.objective="pred_noise" 