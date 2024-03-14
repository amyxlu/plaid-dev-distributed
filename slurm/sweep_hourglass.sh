# sbatch train_hourglass.slrm "experiment=hvq/recons_simple.yaml ++hourglass.lr=5e-5 ++hourglass.lr_num_warmup_steps=1000 ++hourglass.e_dim=128 ++hourglass.downproj_factor=8"
# sbatch train_hourglass.slrm "experiment=hvq/recons_simple.yaml ++hourglass.lr=5e-5 ++hourglass.lr_num_warmup_steps=1000 ++hourglass.e_dim=64 ++hourglass.downproj_factor=16"
# sbatch train_hourglass.slrm "experiment=hvq/recons_simple.yaml ++hourglass.lr=5e-5 ++hourglass.lr_num_warmup_steps=1000 ++hourglass.e_dim=128 ++hourglass.downproj_factor=8 ++trainer.gradient_clip_val=0.5"
# sbatch train_hourglass.slrm "experiment=hvq/recons_simple.yaml ++hourglass.lr=5e-5 ++hourglass.lr_num_warmup_steps=1000 ++hourglass.e_dim=64 ++hourglass.downproj_factor=16 ++trainer.gradient_clip_val=0.5"

# sbatch train_hourglass.slrm "experiment=hvq/recons_simple.yaml ++hourglass.e_dim=1024 ++hourglass.downproj_factor=1 ++hourglass.n_e=65536"
# sbatch train_hourglass.slrm "experiment=hvq/recons_simple.yaml ++hourglass.e_dim=64 ++hourglass.downproj_factor=16 ++hourglass.n_e=65536"
# sbatch train_hourglass.slrm "experiment=hvq/recons_simple.yaml ++hourglass.e_dim=1024 ++hourglass.downproj_factor=1 ++hourglass.n_e=131072"
# sbatch train_hourglass.slrm "experiment=hvq/recons_simple.yaml ++hourglass.e_dim=64 ++hourglass.downproj_factor=16 ++hourglass.n_e=131072"

# sbatch train_hourglass.slrm "experiment=hvq/recons_simple.yaml ++hourglass.n_e=1024 ++hourglass.e_dim=512 ++hourglass.downproj_factor=2"
# sbatch train_hourglass.slrm "experiment=hvq/recons_simple.yaml ++hourglass.n_e=1024 ++hourglass.e_dim=1024 ++hourglass.downproj_factor=1"

# sbatch train_hourglass.slrm "experiment=hvq/recons_simple.yaml"
# sbatch train_hourglass.slrm "experiment=hvq/recons_simple.yaml ++hourglass.use_quantizer=False"

# sbatch train_hourglass.slrm "experiment=hvq/recons_simple.yaml ++hourglass.lr=5e-5 ++hourglass.lr_num_warmup_steps=1000 ++hourglass.e_dim=64 ++hourglass.downproj_factor=1 ++trainer.gradient_clip_val=0.5"
# sbatch train_hourglass.slrm "experiment=hvq/recons_simple.yaml ++hourglass.lr=5e-5 ++hourglass.lr_num_warmup_steps=1000 ++hourglass.e_dim=64 ++hourglass.downproj_factor=2 ++trainer.gradient_clip_val=0.5"
# sbatch train_hourglass.slrm "experiment=hvq/recons_simple.yaml ++hourglass.lr=5e-5 ++hourglass.lr_num_warmup_steps=1000 ++hourglass.e_dim=64 ++hourglass.downproj_factor=4 ++trainer.gradient_clip_val=0.5"
# sbatch train_hourglass.slrm "experiment=hvq/recons_simple.yaml ++hourglass.lr=5e-5 ++hourglass.lr_num_warmup_steps=1000 ++hourglass.e_dim=64 ++hourglass.downproj_factor=8 ++trainer.gradient_clip_val=0.5"
# sbatch train_hourglass.slrm "experiment=hvq/recons_simple.yaml ++hourglass.lr=8e-5 ++hourglass.lr_num_warmup_steps=3000 ++hourglass.e_dim=64 ++hourglass.downproj_factor=16 ++trainer.gradient_clip_val=0.5"


# sbatch train_hourglass.slrm "experiment=hvq/recons_small.yaml ++hourglass.n_e=65536 ++hourglass.lr=8e-5 ++hourglass.lr_num_warmup_steps=3000 ++hourglass.e_dim=64 ++hourglass.downproj_factor=16 ++trainer.gradient_clip_val=0.5"
# sbatch train_hourglass.slrm "experiment=hvq/recons_small.yaml ++hourglass.n_e=131072 ++hourglass.lr=8e-5 ++hourglass.lr_num_warmup_steps=3000 ++hourglass.e_dim=64 ++hourglass.downproj_factor=16 ++trainer.gradient_clip_val=0.5"
# sbatch train_hourglass.slrm "experiment=hvq/recons_small.yaml ++hourglass.n_e=262144 ++hourglass.lr=8e-5 ++hourglass.lr_num_warmup_steps=3000 ++hourglass.e_dim=64 ++hourglass.downproj_factor=16 ++trainer.gradient_clip_val=0.5"
# sbatch train_hourglass.slrm "experiment=hvq/recons_small.yaml ++hourglass.n_e=524288 ++hourglass.lr=8e-5 ++hourglass.lr_num_warmup_steps=3000 ++hourglass.e_dim=64 ++hourglass.downproj_factor=16 ++trainer.gradient_clip_val=0.5"

sbatch train_hourglass.slrm "experiment=hvq/recons_small.yaml ++hourglass.seq_len=512 ++hourglass.n_e=65536 ++hourglass.lr=1e-4 ++hourglass.lr_num_warmup_steps=3000 ++hourglass.e_dim=64 ++hourglass.downproj_factor=16"
sbatch train_hourglass.slrm "experiment=hvq/recons_small.yaml ++hourglass.seq_len=512 ++hourglass.n_e=131072 ++hourglass.lr=1e-4 ++hourglass.lr_num_warmup_steps=3000 ++hourglass.e_dim=64 ++hourglass.downproj_factor=16"

sbatch train_hourglass.slrm "experiment=hvq/recons_small.yaml ++hourglass.seq_len=512 ++hourglass.n_e=65536 ++hourglass.lr=1e-4 ++hourglass.lr_num_warmup_steps=3000 ++hourglass.e_dim=128 ++hourglass.downproj_factor=8"
sbatch train_hourglass.slrm "experiment=hvq/recons_small.yaml ++hourglass.seq_len=512 ++hourglass.n_e=131072 ++hourglass.lr=1e-4 ++hourglass.lr_num_warmup_steps=3000 ++hourglass.e_dim=128 ++hourglass.downproj_factor=8"