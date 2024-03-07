sbatch train_hourglass.slrm "experiment=hvq/recons_simple.yaml ++hourglass.e_dim=512 ++hourglass.downproj_factor=2"
sbatch train_hourglass.slrm "experiment=hvq/recons_simple.yaml ++hourglass.e_dim=1024 ++hourglass.downproj_factor=1"

sbatch train_hourglass.slrm "experiment=hvq/recons_simple.yaml ++hourglass.n_e=1024 ++hourglass.e_dim=512 ++hourglass.downproj_factor=2"
sbatch train_hourglass.slrm "experiment=hvq/recons_simple.yaml ++hourglass.n_e=1024 ++hourglass.e_dim=1024 ++hourglass.downproj_factor=1"