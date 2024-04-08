# sbatch train_hourglass.slrm experiment=hvq/fsq/cath_small.yaml ++hourglass.fsq_levels=[16,16,16,16] 
# sbatch train_hourglass.slrm experiment=hvq/fsq/cath_small.yaml 
# sbatch train_hourglass.slrm experiment=hvq/bounded/cath_small.yaml ++hourglass.downproj_factor=128
sbatch train_hourglass.slrm experiment=hvq/bounded/pfam_full.yaml ++hourglass.downproj_factor=32 ++hourglass.shorten_factor=2
sbatch train_hourglass.slrm experiment=hvq/bounded/pfam_full.yaml ++hourglass.downproj_factor=16 ++hourglass.shorten_factor=2