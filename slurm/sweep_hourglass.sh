# sbatch train_hourglass.slrm "experiment=hvq/no_quant/uniref_full.yaml ++hourglass.lr=1e-4 ++datamodule.num_workers=16 ++hourglass.downproj_factor=128 ++hourglass.shorten_factor=1"  # 512 x 8 
# sbatch train_hourglass.slrm "experiment=hvq/no_quant/uniref_full.yaml ++hourglass.lr=1e-4 ++datamodule.num_workers=16 ++hourglass.downproj_factor=256 ++hourglass.shorten_factor=1"  # 512 x 4 
# sbatch train_hourglass.slrm "experiment=hvq/no_quant/uniref_full.yaml ++hourglass.lr=1e-4 ++datamodule.num_workers=16 ++hourglass.downproj_factor=128 ++hourglass.shorten_factor=2"  # 256 x 8 

sbatch train_hourglass.slrm experiment=hvq/fsq/uniref_subset_small.yaml ++hourglass.lr_num_warmup_steps=3000 ++hourglass.lr=5e-5 ++hourglass.fsq_levels=[8,8,8,8,8,8] ++hourglass.downproj_factor=1  
sbatch train_hourglass.slrm experiment=hvq/fsq/uniref_subset_small.yaml ++hourglass.lr_num_warmup_steps=3000 ++hourglass.lr=5e-5 ++hourglass.fsq_levels=[8,8,8,5,5,5] ++hourglass.shorten_factor=1
sbatch train_hourglass.slrm experiment=hvq/fsq/uniref_subset_small.yaml ++hourglass.lr_num_warmup_steps=3000 ++hourglass.lr=5e-5 ++hourglass.fsq_levels=[8,8,8,8,8] ++hourglass.shorten_factor=1
sbatch train_hourglass.slrm experiment=hvq/fsq/uniref_subset_small.yaml ++hourglass.lr_num_warmup_steps=3000 ++hourglass.lr=5e-5 ++hourglass.fsq_levels=[8,8,8,6,5] ++hourglass.shorten_factor=1
# sbatch train_hourglass.slrm experiment=hvq/fsq/uniref_subset_small.yaml ++datamodule.batch_size=256 ++callbacks/compression.run_every_n_steps=5000 ++hourglass.lr_num_warmup_steps=3000 ++hourglass.lr=5e-5 ++hourglass.fsq_levels=[16,16,16,16] ++hourglass.shorten_factor=1
