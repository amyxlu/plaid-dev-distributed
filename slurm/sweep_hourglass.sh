sbatch train_hourglass.slrm experiment=hourglass/nested_fape.yaml "++hourglass.downproj_factor=[4,4]"
sbatch train_hourglass.slrm experiment=hourglass/nested_recons_only.yaml "++hourglass.shorten_factor=[4,4]" 
sbatch train_hourglass.slrm experiment=hourglass/nested_recons_only.yaml "++hourglass.downproj_factor=[4,4] ++hourglass.shorten_factor=[4,4]"
sbatch train_hourglass.slrm experiment=hourglass/deep_nested_recons_only.yaml "++hourglass.downproj_factor=[4,4] ++hourglass.shorten_factor=[4,4] ++datamodule.batch_size=64"