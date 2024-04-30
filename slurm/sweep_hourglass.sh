# sbatch train_hourglass.slrm resume_from_model_id=jzlv54wl
# sbatch train_hourglass.slrm experiment=hvq/bounded/pfam_full.yaml ++hourglass.downproj_factor=128 ++hourglass.shorten_factor=4 ++logger.name="pfam_tanh_128_8" ++hourglass.lr=8e-5 ++hourglass.lr_num_training_steps=600000 ++hourglass.lr_num_cycles=4
# sbatch train_hourglass.slrm experiment=hvq/bounded/pfam_full.yaml ++hourglass.downproj_factor=128 ++hourglass.shorten_factor=2 ++logger.name="pfam_tanh_256_8" ++hourglass.lr=6e-5
# sbatch train_hourglass.slrm resume_from_model_id=jzlv54wl
sbatch train_hourglass.slrm resume_from_model_id=wiepwn5p