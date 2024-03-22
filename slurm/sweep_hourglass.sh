# sbatch train_hourglass.slrm "experiment=hvq/uniref_subset_small.yaml ++hourglass.seq_loss_weight=1.0" 
# sbatch train_hourglass.slrm "experiment=hvq/uniref_subset_small.yaml ++hourglass.seq_loss_weight=1.0 ++hourglass.lr=3e-5" 
# sbatch train_hourglass.slrm "experiment=hvq/uniref_subset_small.yaml ++hourglass.seq_loss_weight=0.0" 
# sbatch train_hourglass.slrm "experiment=hvq/uniref_subset_small.yaml ++hourglass.seq_loss_weight=0.0 ++hourglass.lr=3e-5" 

# sbatch train_hourglass.slrm "experiment=hvq/fsq_uniref_subset_small.yaml ++hourglass.seq_loss_weight=1.0 ++hourglass.lr=1e-5" 
# sbatch train_hourglass.slrm "experiment=hvq/fsq_uniref_subset_small.yaml ++hourglass.seq_loss_weight=1.0 ++hourglass.lr=3e-5" 
# sbatch train_hourglass.slrm "experiment=hvq/fsq_uniref_subset_small.yaml ++hourglass.seq_loss_weight=0.0 ++hourglass.lr=1e-5" 
# sbatch train_hourglass.slrm "experiment=hvq/fsq_uniref_subset_small.yaml ++hourglass.seq_loss_weight=0.0 ++hourglass.lr=3e-5" 

# sbatch train_hourglass.slrm "experiment=hvq/no_quant_small.yaml ++hourglass.downproj_factor=4 ++hourglass.shorten_factor=1" 
# sbatch train_hourglass.slrm "experiment=hvq/no_quant_small.yaml ++hourglass.downproj_factor=8 ++hourglass.shorten_factor=1" 
# sbatch train_hourglass.slrm "experiment=hvq/no_quant_small.yaml ++hourglass.downproj_factor=4 ++hourglass.shorten_factor=2" 
# sbatch train_hourglass.slrm "experiment=hvq/no_quant_small.yaml ++hourglass.downproj_factor=8 ++hourglass.shorten_factor=2" 
#
# sbatch train_hourglass.slrm "experiment=hvq/no_quant/uniref_subset.yaml ++datamodule.num_workers=16 ++hourglass.downproj_factor=16 ++hourglass.shorten_factor=1"  # 64 x 512
# sbatch train_hourglass.slrm "experiment=hvq/no_quant/uniref_subset.yaml ++datamodule.num_workers=16 ++hourglass.downproj_factor=8 ++hourglass.shorten_factor=2"   # 128 x 256
# sbatch train_hourglass.slrm "experiment=hvq/no_quant/uniref_subset.yaml ++datamodule.num_workers=16 ++hourglass.downproj_factor=8 ++hourglass.shorten_factor=4"   # 128 x 128
# sbatch train_hourglass.slrm "experiment=hvq/no_quant/uniref_subset.yaml ++datamodule.num_workers=16 ++hourglass.downproj_factor=16 ++hourglass.shorten_factor=8"  # 64 x 64 

sbatch train_hourglass.slrm "experiment=hvq/no_quant/uniref_subset.yaml ++hourglass.lr=3e-5 ++hourglass.lr_adam_betas=[0.5,0.99] ++datamodule.num_workers=16 ++hourglass.downproj_factor=128 ++hourglass.shorten_factor=1"  # 512 x 8 
sbatch train_hourglass.slrm "experiment=hvq/no_quant/uniref_subset.yaml ++hourglass.lr=3e-5 ++hourglass.lr_adam_betas=[0.5,0.99] ++datamodule.num_workers=16 ++hourglass.downproj_factor=256 ++hourglass.shorten_factor=1"  # 512 x 4 
sbatch train_hourglass.slrm "experiment=hvq/no_quant/uniref_subset.yaml ++hourglass.lr=3e-5 ++hourglass.lr_adam_betas=[0.5,0.99] ++datamodule.num_workers=16 ++hourglass.downproj_factor=128 ++hourglass.shorten_factor=2"  # 512 x 8 
sbatch train_hourglass.slrm "experiment=hvq/no_quant/uniref_subset.yaml ++hourglass.lr=3e-5 ++hourglass.lr_adam_betas=[0.5,0.99] ++datamodule.num_workers=16 ++hourglass.downproj_factor=256 ++hourglass.shorten_factor=2"  # 512 x 4 

# sbatch train_hourglass.slrm "experiment=hvq/fsq_uniref_subset_small.yaml ++hourglass.fsq_levels=[4,4,4,4,4,4,4,4]" 
# sbatch train_hourglass.slrm "experiment=hvq/fsq_uniref_subset_small.yaml ++hourglass.fsq_levels=[8,8,8,8]"
