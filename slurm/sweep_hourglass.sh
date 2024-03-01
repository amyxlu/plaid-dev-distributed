# sbatch train_hourglass.slrm experiment=hourglass/nested_recons_only.yaml "++datamodule.seq_len=128" "++datamodule.batch_size=512" "++hourglass.lr_num_warmup_steps=5000"
# sbatch train_hourglass.slrm experiment=hourglass/nested_recons_only.yaml "++datamodule.seq_len=128" "++datamodule.batch_size=512" "++hourglass.lr_num_warmup_steps=5000" "++hourglass.lr_sched_type=cosine" "++hourglass.lr_num_cycles=4"
# sbatch train_hourglass.slrm experiment=hourglass/nested_seq_recons.yaml "++datamodule.seq_len=128" "++datamodule.batch_size=512" "++hourglass.lr_num_warmup_steps=5000"
# sbatch train_hourglass.slrm experiment=hourglass/nested_seq_recons.yaml "++datamodule.seq_len=128" "++datamodule.batch_size=512" "++hourglass.lr_num_warmup_steps=5000" "++hourglass.lr_sched_type=cosine" "++hourglass.seq_loss_weight=5" "++hourglass.lr_num_cycles=4"
# sbatch train_hourglass.slrm experiment=hourglass/with_fape.yaml "++datamodule.seq_len=128" "++datamodule.batch_size=16" "++hourglass.lr_num_warmup_steps=5000" "++hourglass.seq_loss_weight=1" "++hourglass.struct_loss_weight=1" "++trainer.log_every_n_steps=100"
# sbatch train_hourglass.slrm experiment=hourglass/with_fape.yaml "++datamodule.seq_len=128" "++datamodule.batch_size=16" "++hourglass.lr_num_warmup_steps=5000" "++hourglass.seq_loss_weight=0.1" "++hourglass.struct_loss_weight=5" "++trainer.log_every_n_steps=100"
sbatch train_hourglass.slrm experiment=hourglass/recons_loss_only.yaml "++hourglass.lr_num_warmup_steps=5000 ++trainer.log_every_n_steps=100"
