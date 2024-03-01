# sbatch train_hourglass.slrm experiment=hvq/recons_simple.yaml "++hourglass.lr_num_warmup_steps=5000 ++hourglass.seq_loss_weight=1" 
# sbatch train_hourglass.slrm experiment=hvq/recons_simple.yaml "++hourglass.lr_num_warmup_steps=5000"
# sbatch train_hourglass.slrm experiment=hvq/recons_simple.yaml "++hourglass.lr_num_warmup_steps=5000 ++hourglass.seq_loss_weight=1"
# sbatch train_hourglass.slrm experiment=hvq/recons_simple.yaml "++hourglass.lr_num_warmup_steps=5000"
# sbatch train_hourglass.slrm experiment=hvq/recons_simple.yaml "++hourglass.lr_num_warmup_steps=5000 ++hourglass.updown_sample_type='naive'"
sbatch train_hourglass.slrm experiment=hvq/recons_simple.yaml "++hourglass.downproj_factor=2"
sbatch train_hourglass.slrm experiment=hvq/recons_simple.yaml "++hourglass.downproj_factor=1"
sbatch train_hourglass.slrm experiment=hvq/recons_simple.yaml "++hourglass.downproj_factor=1 ++hourglass.shorten_factor=4"
# sbatch train_hourglass.slrm experiment=hvq/recons_simple_nonorm.yaml "++hourglass.lr_num_warmup_steps=5000"
# sbatch train_hourglass.slrm experiment=hvq/recons_simple_nonorm.yaml "++hourglass.lr_num_warmup_steps=5000 ++hourglass.seq_loss_weight=1"
