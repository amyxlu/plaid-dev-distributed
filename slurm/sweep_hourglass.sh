# sbatch train_hourglass.slrm "experiment=hvq/no_quant/uniref_full.yaml ++hourglass.lr=1e-4 ++datamodule.num_workers=16 ++hourglass.downproj_factor=128 ++hourglass.shorten_factor=1"  # 512 x 8 
# sbatch train_hourglass.slrm "experiment=hvq/no_quant/uniref_full.yaml ++hourglass.lr=1e-4 ++datamodule.num_workers=16 ++hourglass.downproj_factor=256 ++hourglass.shorten_factor=1"  # 512 x 4 
# sbatch train_hourglass.slrm "experiment=hvq/no_quant/uniref_full.yaml ++hourglass.lr=1e-4 ++datamodule.num_workers=16 ++hourglass.downproj_factor=128 ++hourglass.shorten_factor=2"  # 256 x 8 

sbatch train_hourglass.slrm experiment=hvq/fsq/uniref_subset_small.yaml ++hourglass.lr_num_warmup_steps=3000 ++hourglass.lr=1e-4 ++hourglass.fsq_levels=[4,4,4,4,4,4,4,4] ++hourglass.downproj_factor=128
sbatch train_hourglass.slrm experiment=hvq/fsq/uniref_subset_small.yaml ++hourglass.lr_num_warmup_steps=3000 ++hourglass.lr=1e-4 ++hourglass.fsq_levels=[8,8,8,8] ++hourglass.downproj_factor=256  
