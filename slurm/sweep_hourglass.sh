# sbatch train_hourglass.slrm experiment=hvq/recons_simple.yaml "++hourglass.downproj_factor=1 ++hourglass.shorten_factor=1 ++hourglass.n_e=512 ++datamodule.seq_len=512 ++datamodule.batch_size=64"
# sbatch train_hourglass.slrm experiment=hvq/recons_simple.yaml "++hourglass.downproj_factor=1 ++hourglass.shorten_factor=1 ++hourglass.n_e=256 ++datamodule.seq_len=512 ++datamodule.batch_size=64"
sbatch train_hourglass.slrm experiment=hvq/recons_simple_uniref.yaml "++hourglass.downproj_factor=1 ++hourglass.n_e=128" 
sbatch train_hourglass.slrm experiment=hvq/recons_simple_uniref.yaml "++hourglass.downproj_factor=1 ++hourglass.n_e=256" 
sbatch train_hourglass.slrm experiment=hvq/recons_simple_uniref.yaml "++hourglass.downproj_factor=1 ++hourglass.n_e=512" 