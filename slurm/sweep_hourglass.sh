# sbatch train_hourglass.slrm "experiment=hvq/recons_small.yaml ++hourglass.n_e=65536 ++hourglass.lr=8e-5 ++hourglass.lr_num_warmup_steps=3000 ++hourglass.e_dim=64 ++hourglass.downproj_factor=16 ++trainer.gradient_clip_val=0.5"
# sbatch train_hourglass.slrm "experiment=hvq/recons_small.yaml ++hourglass.n_e=131072 ++hourglass.lr=8e-5 ++hourglass.lr_num_warmup_steps=3000 ++hourglass.e_dim=64 ++hourglass.downproj_factor=16 ++trainer.gradient_clip_val=0.5"
# sbatch train_hourglass.slrm "experiment=hvq/recons_small.yaml ++hourglass.n_e=262144 ++hourglass.lr=8e-5 ++hourglass.lr_num_warmup_steps=3000 ++hourglass.e_dim=64 ++hourglass.downproj_factor=16 ++trainer.gradient_clip_val=0.5"
# sbatch train_hourglass.slrm "experiment=hvq/recons_small.yaml ++hourglass.n_e=524288 ++hourglass.lr=8e-5 ++hourglass.lr_num_warmup_steps=3000 ++hourglass.e_dim=64 ++hourglass.downproj_factor=16 ++trainer.gradient_clip_val=0.5"

sbatch train_hourglass.slrm experiment=hvq/uniref_full_small.yaml ++hourglass.downproj_factor=1 ++hourglass.n_e=512

sbatch train_hourglass.slrm "experiment=hvq/recons_small.yaml ++logger.name=len64_seqloss ++hourglass.seq_loss_weight=1.0 ++hourglass.n_e=65536 ++hourglass.e_dim=64 ++hourglass.downproj_factor=16"
sbatch train_hourglass.slrm "experiment=hvq/recons_small.yaml ++logger.name=len64_seqloss_edim128 ++hourglass.seq_loss_weight=1.0 ++hourglass.n_e=65536 ++hourglass.e_dim=128 ++hourglass.downproj_factor=8"
sbatch train_hourglass.slrm "experiment=hvq/recons_small.yaml ++logger.name=len64_edim128 ++hourglass.n_e=65536 ++hourglass.e_dim=128 ++hourglass.downproj_factor=8"
sbatch train_hourglass.slrm "experiment=hvq/recons_small.yaml ++logger.name=len64_seqloss_structloss ++hourglass.seq_loss_weight=1.0 ++hourglass.struct_loss_weight=1.0 ++datamodule.batch_size 32 ++hourglass.n_e=65536 ++hourglass.downproj_factor=16"

# sbatch train_hourglass.slrm experiment=hvq/uniref_full_small.yaml 
# sbatch train_hourglass.slrm experiment=hvq/uniref_full_small.yaml ++hourglass.downproj_factor=1 ++hourglass.n_e=128