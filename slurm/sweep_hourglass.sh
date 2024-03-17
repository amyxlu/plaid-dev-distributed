# sbatch train_hourglass.slrm "experiment=hvq/uniref_subset_small.yaml ++hourglass.seq_loss_weight=1.0" 
# sbatch train_hourglass.slrm "experiment=hvq/uniref_subset_small.yaml ++hourglass.seq_loss_weight=1.0 ++hourglass.lr=3e-5" 
# sbatch train_hourglass.slrm "experiment=hvq/uniref_subset_small.yaml ++hourglass.seq_loss_weight=0.0" 
# sbatch train_hourglass.slrm "experiment=hvq/uniref_subset_small.yaml ++hourglass.seq_loss_weight=0.0 ++hourglass.lr=3e-5" 

sbatch train_hourglass.slrm "experiment=hvq/fsq_uniref_subset_small.yaml ++hourglass.seq_loss_weight=1.0 ++hourglass.lr=1e-5" 
sbatch train_hourglass.slrm "experiment=hvq/fsq_uniref_subset_small.yaml ++hourglass.seq_loss_weight=1.0 ++hourglass.lr=3e-5" 
sbatch train_hourglass.slrm "experiment=hvq/fsq_uniref_subset_small.yaml ++hourglass.seq_loss_weight=0.0 ++hourglass.lr=1e-5" 
sbatch train_hourglass.slrm "experiment=hvq/fsq_uniref_subset_small.yaml ++hourglass.seq_loss_weight=0.0 ++hourglass.lr=3e-5" 