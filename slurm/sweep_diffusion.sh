# A
# python run_diffusion_slrm.py --n_gpus 4 --flags "experiment=udit/B ++logger.name='AblationA' ++diffusion.objective=pred_noise ++diffusion.beta_scheduler_name=adm_cosine ++denoiser.use_self_conditioning=False ++diffusion.min_snr_loss_weight=False"

# B = A + pred_v
# python run_diffusion_slrm.py --n_gpus 4 --flags "experiment=udit/B ++logger.name='AblationB__A+predv' ++diffusion.beta_scheduler_name=adm_cosine ++denoiser.use_self_conditioning=False ++diffusion.min_snr_loss_weight=False"

# C = A + min_snr
# python run_diffusion_slrm.py --n_gpus 4 --flags "experiment=udit/B ++logger.name='AblationC__A+minsnr' ++diffusion.objective=pred_noise ++diffusion.beta_scheduler_name=adm_cosine ++denoiser.use_self_conditioning=False"

# D = A + pred_v + min_snr + sigmoid
# python run_diffusion_slrm.py --n_gpus 4 --flags "experiment=udit/B ++logger.name='AblationD__A+predv+minsnr+sigmoid' ++denoiser.use_self_conditioning=False"

# E = A + pred_v + min_snr + sigmoid + self_cond
# already trained

# F = no cond drop
# python run_diffusion_slrm.py --n_gpus 4 --flags "experiment=udit/B ++logger.name='AblationF__NoCondDrop' ++diffusion.function_y_cond_drop_prob=0.0 ++diffusion.organism_y_cond_drop_prob=0.0"

# python run_diffusion_slrm.py -g 4 -n 1 -c 8 --flags "++resume_from_model_id=lqp25b7g"
# python run_diffusion_slrm.py -g 8 -n 2 -c 8 --flags "++resume_from_model_id=4hdab8dn"
# python run_diffusion_slrm.py -g 8 -n 5 -c 8 --flags "++resume_from_model_id=5j007z42"
# python run_diffusion_slrm.py -g 8 -n 2 -c 8 --flags "++resume_from_model_id=4hdab8dn"
# python run_diffusion_slrm.py -g 4 -n 1 -c 8 --flags "++resume_from_model_id=ksme77o6"
# python run_diffusion_slrm.py -g 4 -n 1 -c 8 --flags "++resume_from_model_id=lqp25b7g"
# python run_diffusion_slrm.py -g 4 -n 1 -c 8 --flags "++resume_from_model_id=zlkurtdd"
python run_diffusion_slrm.py -g 8 -n 4 -c 8 --flags "++resume_from_model_id=oqdkajg3"

# python run_diffusion_slrm.py -g 8 -n 10 -c 8 --flags "experiment=udit/XXL ++logger.name=UDiT_XXL"

# python run_diffusion_slrm.py -g 8 -n 4 -c 8 --flags "++resume_from_model_id=5j007z42"

# python run_diffusion_slrm.py -g 8 -n 2 -c 8 --flags "experiment=udit/L ++logger.name=UDiT_L"
