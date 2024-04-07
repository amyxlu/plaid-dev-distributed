sbatch train_hourglass.slrm experiment=hvq/bounded/pfam_full.yaml ++hourglass.downproj_factor=32 ++hourglass.shorten_factor=2
sbatch train_hourglass.slrm experiment=hvq/bounded/pfam_full.yaml ++hourglass.downproj_factor=128 ++hourglass.shorten_factor=1
sbatch train_hourglass.slrm experiment=hvq/bounded/pfam_full.yaml ++hourglass.downproj_factor=16 ++hourglass.shorten_factor=2

sbatch train_hourglass.slrm experiment=hvq/fsq/pfam_full.yaml 
sbatch train_hourglass.slrm experiment=hvq/fsq/pfam_full.yaml ++hourglass.fsq_levels=[16,16,16,16] 