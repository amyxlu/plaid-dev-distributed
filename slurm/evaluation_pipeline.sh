# sbatch evaluation_pipeline.slrm experiment=dpm ++sample.cond_scale=8. ++sample.sampling_timesteps=20 ++sample.function_idx=547

# start and end of lines to be evaluated
# sbatch conditional_eval_loop.slrm 0 4
sbatch conditional_eval_loop.slrm 5 55 
sbatch conditional_eval_loop.slrm 56 100 
sbatch conditional_eval_loop.slrm 101 155 
sbatch conditional_eval_loop.slrm 156 200 
sbatch conditional_eval_loop.slrm 201 250 
sbatch conditional_eval_loop.slrm 251 300
sbatch conditional_eval_loop.slrm 301 350
sbatch conditional_eval_loop.slrm 351 400
sbatch conditional_eval_loop.slrm 401 450
sbatch conditional_eval_loop.slrm 451 500
sbatch conditional_eval_loop.slrm 501 550
