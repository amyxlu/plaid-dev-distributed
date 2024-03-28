# sbatch train_hourglass.slrm "experiment=hvq/no_quant/uniref_subset.yaml ++hourglass.lr=1e-5 ++datamodule.num_workers=16 ++hourglass.downproj_factor=64 ++hourglass.shorten_factor=1"  # 512 x 16 
# sbatch train_hourglass.slrm "experiment=hvq/no_quant/uniref_subset.yaml ++hourglass.lr=1e-5 ++datamodule.num_workers=16 ++hourglass.downproj_factor=64 ++hourglass.shorten_factor=2"  # 256 x 16 
# sbatch train_hourglass.slrm "experiment=hvq/no_quant/uniref_subset.yaml ++hourglass.lr=1e-5 ++datamodule.num_workers=16 ++hourglass.downproj_factor=32 ++hourglass.shorten_factor=1"  # 512 x 32
# sbatch train_hourglass.slrm "experiment=hvq/no_quant/uniref_subset.yaml ++hourglass.lr=1e-5 ++datamodule.num_workers=16 ++hourglass.downproj_factor=32 ++hourglass.shorten_factor=2"  # 256 x 32
# sbatch train_hourglass.slrm "experiment=hvq/no_quant/uniref_subset.yaml ++hourglass.lr=1e-5 ++datamodule.num_workers=16 ++hourglass.downproj_factor=16 ++hourglass.shorten_factor=1"  # 512 x 64 
# sbatch train_hourglass.slrm "experiment=hvq/no_quant/uniref_subset.yaml ++hourglass.lr=1e-5 ++datamodule.num_workers=16 ++hourglass.downproj_factor=16 ++hourglass.shorten_factor=2"  # 256 x 64 

# sbatch train_hourglass.slrm "experiment=hvq/no_quant/uniref_subset.yaml ++hourglass.lr=1e-5 ++datamodule.num_workers=16 ++hourglass.downproj_factor=8 ++hourglass.shorten_factor=1"  # 256 x 64 
# sbatch train_hourglass.slrm "experiment=hvq/no_quant/uniref_subset.yaml ++hourglass.lr=1e-5 ++datamodule.num_workers=16 ++hourglass.downproj_factor=8 ++hourglass.shorten_factor=2"  # 256 x 64 
# sbatch train_hourglass.slrm "experiment=hvq/no_quant/uniref_subset.yaml ++hourglass.lr=1e-5 ++datamodule.num_workers=16 ++hourglass.downproj_factor=4 ++hourglass.shorten_factor=1"  # 256 x 64 
# sbatch train_hourglass.slrm "experiment=hvq/no_quant/uniref_subset.yaml ++hourglass.lr=1e-5 ++datamodule.num_workers=16 ++hourglass.downproj_factor=4 ++hourglass.shorten_factor=2"  # 256 x 64 

# sbatch train_hourglass.slrm "experiment=hvq/bounded/uniref_subset.yaml ++hourglass.lr=1e-5 ++datamodule.num_workers=16"
sbatch train_hourglass.slrm "experiment=hvq/bounded/uniref_subset.yaml ++hourglass.downproj_factor=64 ++hourglass.fsq_levels=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] ++hourglass.lr=5e-5 ++datamodule.num_workers=16"  # 16
sbatch train_hourglass.slrm "experiment=hvq/bounded/uniref_subset.yaml ++hourglass.downproj_factor=32 ++hourglass.fsq_levels=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] ++hourglass.lr=5e-5 ++datamodule.num_workers=16"  # 32
sbatch train_hourglass.slrm "experiment=hvq/bounded/uniref_subset.yaml ++hourglass.downproj_factor=22 ++hourglass.fsq_levels=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] ++hourglass.lr=5e-5 ++datamodule.num_workers=16"  # 48
sbatch train_hourglass.slrm "experiment=hvq/bounded/uniref_subset.yaml ++hourglass.downproj_factor=16 ++hourglass.fsq_levels=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] ++hourglass.lr=5e-5 ++datamodule.num_workers=16"  # 64

# sbatch train_hourglass.slrm experiment=hvq/fsq/uniref_subset_small.yaml ++hourglass.lr_num_warmup_steps=3000 ++hourglass.lr=5e-5 ++hourglass.fsq_levels=[8,8,8,8,8,8] ++hourglass.downproj_factor=1  
# sbatch train_hourglass.slrm experiment=hvq/fsq/uniref_subset_small.yaml ++hourglass.lr_num_warmup_steps=3000 ++hourglass.lr=5e-5 ++hourglass.fsq_levels=[8,8,8,5,5,5] ++hourglass.shorten_factor=1
# sbatch train_hourglass.slrm experiment=hvq/fsq/uniref_subset_small.yaml ++hourglass.lr_num_warmup_steps=3000 ++hourglass.lr=5e-5 ++hourglass.fsq_levels=[8,8,8,8,8] ++hourglass.shorten_factor=1
# sbatch train_hourglass.slrm experiment=hvq/fsq/uniref_subset_small.yaml ++hourglass.lr_num_warmup_steps=3000 ++hourglass.lr=5e-5 ++hourglass.fsq_levels=[8,8,8,6,5] ++hourglass.shorten_factor=1
# sbatch train_hourglass.slrm experiment=hvq/fsq/uniref_subset_small.yaml ++datamodule.batch_size=256 ++callbacks/compression.run_every_n_steps=5000 ++hourglass.lr_num_warmup_steps=3000 ++hourglass.lr=5e-5 ++hourglass.fsq_levels=[16,16,16,16] ++hourglass.shorten_factor=1
