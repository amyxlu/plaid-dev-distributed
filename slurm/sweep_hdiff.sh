sbatch train_diffusion.slrm experiment=hdif/dual_track/rocklin_tanh_512_8 
sbatch train_diffusion.slrm experiment=hdif/dual_track/rocklin_tanh_256_32
sbatch train_diffusion.slrm experiment=hdif/dual_track/rocklin_tanh_256_64

sbatch train_diffusion.slrm experiment=hdif/dual_track/rocklin_tanh_512_8 ++denoiser.use_self_conditioning=True
sbatch train_diffusion.slrm experiment=hdif/dual_track/rocklin_tanh_256_64 ++denoiser.use_self_conditioning=True

sbatch train_diffusion.slrm experiment=hdif/dual_track/rocklin_tanh_512_8 ++denoiser.use_self_conditioning=True ++diffusion.objective="pred_noise"
sbatch train_diffusion.slrm experiment=hdif/dual_track/rocklin_tanh_256_64 ++denoiser.use_self_conditioning=True ++diffusion.objective="pred_noise"

sbatch train_diffusion.slrm experiment=hdif/dual_track/rocklin_tanh_512_8 ++denoiser.use_self_conditioning=True ++diffusion.objective="pred_noise" ++diffusion.sequence_decoder_weight=1.0
sbatch train_diffusion.slrm experiment=hdif/dual_track/rocklin_tanh_256_64 ++denoiser.use_self_conditioning=True ++diffusion.objective="pred_noise" ++diffusion.sequence_decoder_weight=1.0