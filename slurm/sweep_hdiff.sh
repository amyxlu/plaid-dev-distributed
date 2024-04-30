sbatch train_diffusion.slrm experiment=diffusion/dit/pfam_dit_base ++logger.name="pfam_s30K_base_512x8" ++compression_model_id="jzlv54wl" ++datamodule.batch_size=128
sbatch train_diffusion.slrm experiment=diffusion/dit/pfam_dit_base ++logger.name="pfam_s30K_base_256x32" ++compression_model_id="j1v1wv6w" ++datamodule.batch_size=128
sbatch train_diffusion.slrm experiment=diffusion/dit/pfam_dit_base ++logger.name="pfam_s30K_base_256x64" ++compression_model_id="h9hzw1bp" ++datamodule.batch_size=64

# 256 x 64: h9hzw1bp
# 256 x 32: j1v1wv6w
# 512 x 8: jzlv54wl