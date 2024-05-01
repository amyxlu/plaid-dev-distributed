# sbatch train_diffusion.slrm experiment=diffusion/dit/pfam_dit_base ++input_dim=8 ++logger.name="pfam_s30K_base_512x8" ++compression_model_id="jzlv54wl" ++datamodule.batch_size=128
# sbatch train_diffusion.slrm experiment=diffusion/dit/pfam_dit_small ++input_dim=8 ++logger.name="pfam_s30K_base_512x8" ++compression_model_id="jzlv54wl" ++datamodule.batch_size=128
# sbatch train_diffusion.slrm experiment=diffusion/dit/pfam_dit_base ++input_dim=32 ++logger.name="pfam_s30K_base_256x32" ++compression_model_id="j1v1wv6w" ++datamodule.batch_size=128
# sbatch train_diffusion.slrm experiment=diffusion/dit/pfam_dit_base ++input_dim=64 ++logger.name="pfam_s30K_base_256x64" ++compression_model_id="h9hzw1bp" ++datamodule.batch_size=64

sbatch train_diffusion.slrm experiment=diffusion/dit/5K/base ++input_dim=8 ++logger.name="pfam_s5K_base_512x8" 
sbatch train_diffusion.slrm experiment=diffusion/dit/5K/base ++input_dim=8 ++logger.name="pfam_s5K_base_512x8" ++denoiser.hidden_size=1024 ++denoiser.depth=4 ++denoiser.num_heads=16 ++trainer.gradient_clip_val=0.5 
sbatch train_diffusion.slrm experiment=diffusion/dit/5K/base ++input_dim=8 ++logger.name="pfam_s5K_base_512x8" ++trainer.gradient_clip_val=0.5 

# sbatch train_diffusion.slrm experiment=diffusion/dit/pfam_dit_base ++input_dim=8 ++logger.name="pfam_s5K_base_512x8" ++compression_model_id="jzlv54wl" ++datamodule.batch_size=128
# sbatch train_diffusion.slrm experiment=diffusion/udit/pfam_udit_base ++input_dim=8 ++logger.name="pfam_s30K_base_512x8" ++compression_model_id="jzlv54wl" ++datamodule.batch_size=128
# sbatch train_diffusion.slrm experiment=diffusion/udit/pfam_udit_base ++input_dim=32 ++logger.name="pfam_s30K_base_256x32" ++compression_model_id="j1v1wv6w" ++datamodule.batch_size=256
# sbatch train_diffusion.slrm experiment=diffusion/udit/pfam_udit_base ++input_dim=64 ++logger.name="pfam_s30K_base_256x64" ++compression_model_id="h9hzw1bp" ++datamodule.batch_size=128
# sbatch train_diffusion.slrm experiment=diffusion/udit/pfam_udit_base ++input_dim=8 ++logger.name="pfam_s5K_base_256x64" ++compression_model_id="jzlv54wl" 
# 256 x 64: h9hzw1bp
# 256 x 32: j1v1wv6w
# 512 x 8: jzlv54wl