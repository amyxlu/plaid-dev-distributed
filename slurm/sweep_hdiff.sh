# sbatch train_diffusion.slrm experiment=hdif/pfam_dit ++diffusion.lr=1e-3 
# sbatch train_diffusion.slrm experiment=hdif/pfam_dit ++trainer.gradient_clip_val=5
# sbatch train_diffusion.slrm experiment=hdif/pfam_dit ++diffusion.sequence_decoder_weight=1.0
# sbatch train_diffusion.slrm experiment=hdif/pfam_dit ++diffusion.lr=1e-2 
# sbatch train_diffusion.slrm experiment=hdif/pfam_dit ++diffusion.lr=1e-5 
# sbatch train_diffusion.slrm experiment=hdif/pfam_dit ++diffusion.x_downscale_factor=0.01
# sbatch train_diffusion.slrm ++datamodule.max_seq_len=512 experiment=hdif/pfam_dit_toy ++trainer.limit_val_batches=0.0 
# sbatch train_diffusion.slrm ++datamodule.max_seq_len=512 experiment=hdif/pfam_dit_toy ++trainer.limit_val_batches=0.0 ++denoiser.depth=12 ++trainer.gradient_clip_val=5
sbatch train_diffusion.slrm experiment=hdif/pfam_udit_toy ++trainer.limit_val_batches=0.0 ++denoiser.depth=12 ++trainer.gradient_clip_val=5
