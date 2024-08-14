# A
# python run_diffusion_slrm.py --n_gpus 1 --flags "++logger.name='A' ++diffusion.objective=pred_noise ++diffusion.beta_scheduler_name=adm_cosine ++denoiser.use_self_conditioning=False ++min_snr_loss_weight=False"

# B = A + min_snr + pred_v
# python run_diffusion_slrm.py --n_gpus 1 --flags "++logger.name=B__A+predv ++diffusion.beta_scheduler_name=adm_cosine ++denoiser.use_self_conditioning=False"

# # C = B + sigmoid
# python run_diffusion_slrm.py --n_gpus 1 --flags "++logger.name=C__B+sigmoid ++denoiser.use_self_conditioning=False"

# # D = C + self_conditioning
# python run_diffusion_slrm.py --n_gpus 1 --flags "++logger.name=D__C+self_cond"

# # E = D + downscale
# python run_diffusion_slrm.py --n_gpus 1 --flags "++logger.name=E__D+downscale_0.5 ++diffusion.x_downscale_factor=0.5"
# python run_diffusion_slrm.py --n_gpus 1 --flags "++logger.name=E__D+downscale_0.3 ++diffusion.x_downscale_factor=0.3"
# python run_diffusion_slrm.py --n_gpus 1 --flags "++logger.name=E__D+downscale_0.1 ++diffusion.x_downscale_factor=0.1"

# # F = D + change y_cond_drop
# python run_diffusion_slrm.py --n_gpus 1 --flags "++logger.name=F__D+conddrop0.3 ++diffusion.function_y_cond_drop_prob=0.3 ++diffusion.organism_y_cond_drop_prob=0.3"
# python run_diffusion_slrm.py --n_gpus 1 --flags "++logger.name=F__D+conddrop0.05 ++diffusion.function_y_cond_drop_prob=0.05 ++diffusion.organism_y_cond_drop_prob=0.2"
# python run_diffusion_slrm.py --n_gpus 1 --flags "++logger.name=F__D+conddrop0.00 ++diffusion.function_y_cond_drop_prob=0. ++diffusion.organism_y_cond_drop_prob=0."

# # A + sigmoid + min_snr + downscale
# python run_diffusion_slrm.py --n_gpus 1 --flags "++logger.name=G__A+sigmoid+downscale_0.5 ++diffusion.x_downscale_factor=0.5 ++diffusion.objective=pred_noise ++denoiser.use_self_conditioning=False"
# python run_diffusion_slrm.py --n_gpus 1 --flags "++logger.name=G__A+sigmoid+downscale_0.3 ++diffusion.x_downscale_factor=0.3 ++diffusion.objective=pred_noise ++denoiser.use_self_conditioning=False"
# python run_diffusion_slrm.py --n_gpus 1 --flags "++logger.name=G__A+sigmoid+downscale_0.1 ++diffusion.x_downscale_factor=0.1 ++diffusion.objective=pred_noise ++denoiser.use_self_conditioning=False"

# python run_diffusion_slrm.py -g 4 -n 1 -c 8 --flags "experiment=compositional/XL ++logger.name=XL"
# python run_diffusion_slrm.py -g 8 -n 1 -c 8 --flags "experiment=compositional/XXL ++logger.name=XXL"
# python run_diffusion_slrm.py -g 8 -n 2 -c 8 --flags "experiment=compositional/XXL ++logger.name=XXL"

python run_diffusion_slrm.py -g 1 -n 1 -c 8 --flags "resume_from_model_id=o2cgf3eq"
python run_diffusion_slrm.py -g 1 -n 1 -c 8 --flags "resume_from_model_id=jgxkivbq"