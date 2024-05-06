# sbatch train_edm.slrm experiment=cfg/256x8_vd ++use_compile=False 
# sbatch train_edm.slrm experiment=cfg/256x8_vd ++denoiser.use_self_conditioning=True ++use_compile=False
# sbatch train_edm.slrm experiment=cfg/256x8_cosine ++use_compile=False
# sbatch train_edm.slrm experiment=cfg/256x8_cosine ++denoiser.use_self_conditioning=True ++use_compile=False
sbatch train_edm.slrm experiment=cfg/256x8_vd ++diffusion.lr=1e-5
sbatch train_edm.slrm experiment=cfg/256x8_vd ++diffusion.lr=3e-5 ++diffusion.lr_sched_type=inverse_sqrt
sbatch train_edm.slrm experiment=cfg/256x8_vd ++diffusion.lr=5e-5 ++diffusion.lr_sched_type=inverse_sqrt
sbatch train_edm.slrm experiment=cfg/256x8_vd ++diffusion.lr=3e-5 ++diffusion.lr_sched_type=linear
sbatch train_edm.slrm experiment=cfg/256x8_cosine ++diffusion.lr=1e-5