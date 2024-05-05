sbatch train_edm.slrm experiment=cfg/256x8_vd 
# sbatch train_edm.slrm experiment=cfg/256x8_vd ++denoiser.use_self_conditioning=True
sbatch train_edm.slrm experiment=cfg/256x8_cosine
# sbatch train_edm.slrm experiment=cfg/256x8_cosine ++denoiser.use_self_conditioning=True