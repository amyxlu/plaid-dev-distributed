sbatch train_diffusion.slrm experiment=hdif/pfam_dit ++datamodule.num_workers=16 
sbatch train_diffusion.slrm experiment=hdif/pfam_clan ++datamodule.num_workers=16 ++callbacks.sample.batch_size=32 ++diffusion.lr=1e-5 
sbatch train_diffusion.slrm experiment=hdif/pfam_clan ++resume_from_model_id='2dj696qw'
sbatch train_diffusion.slrm experiment=hdif/pfam_clan ++resume_from_model_id='17lwgwcf'