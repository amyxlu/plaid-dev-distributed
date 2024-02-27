# sbatch train_hourglass.slrm experiment=hourglass/nested_fape.yaml "++hourglass.downproj_factor=[4,4]"
# sbatch train_hourglass.slrm experiment=hourglass/nested_recons_only.yaml "++hourglass.shorten_factor=[4,4]" 
# sbatch train_hourglass.slrm experiment=hourglass/nested_recons_only.yaml "++hourglass.downproj_factor=[4,4] ++hourglass.shorten_factor=[4,4]"
# sbatch train_hourglass.slrm experiment=hourglass/deep_nested_recons_only.yaml "++hourglass.downproj_factor=[4,4] ++hourglass.shorten_factor=[4,4] ++datamodule.batch_size=32 ++trainer.accumulate_grad_batches=4"

# sbatch train_hourglass.slrm experiment=hourglass/nested_recons_only.yaml "++hourglass.downproj_factor=[8,8] ++hourglass.shorten_factor=[8,8]"
# sbatch train_hourglass.slrm experiment=hourglass/nested_recons_only.yaml "++hourglass.downproj_factor=[16,16] ++hourglass.shorten_factor=[8,8]"
# sbatch train_hourglass.slrm experiment=hourglass/nested_recons_only.yaml "++hourglass.downproj_factor=[16,16] ++hourglass.shorten_factor=[16,16]"

# sbatch train_hourglass.slrm experiment=hourglass/nested_recons_only.yaml "++hourglass.downproj_factor=[8,8] ++hourglass.shorten_factor=[8,8] ++hourglass.updown_sample_type=linear"
# sbatch train_hourglass.slrm experiment=hourglass/nested_recons_only.yaml "++hourglass.downproj_factor=[16,16] ++hourglass.shorten_factor=[8,8] ++hourglass.updown_sample_type=linear"
# sbatch train_hourglass.slrm experiment=hourglass/nested_recons_only.yaml "++hourglass.downproj_factor=[16,16] ++hourglass.shorten_factor=[16,16] ++hourglass.updown_sample_type=linear"

sbatch train_hourglass.slrm experiment=hourglass/nested_recons_only.yaml "++hourglass.downproj_factor=[8,8] ++hourglass.shorten_factor=[4,4] ++hourglass.out_norm_type=layer ++hourglass.body_norm_type=channel"
# sbatch train_hourglass.slrm experiment=hourglass/nested_recons_only.yaml "++hourglass.downproj_factor=[16,16] ++hourglass.shorten_factor=[16,16] ++hourglass.out_norm_type=layer ++hourglass.body_norm_type=channel" 
# sbatch train_hourglass.slrm experiment=hourglass/nested_recons_only.yaml "++hourglass.downproj_factor=[8,16] ++hourglass.shorten_factor=[8,16] ++hourglass.norm_out=channel"
# sbatch train_hourglass.slrm experiment=hourglass/nested_recons_only.yaml "++hourglass.downproj_factor=[16,8] ++hourglass.shorten_factor=[16,8] ++hourglass.norm_out=channel"