# sbatch train_hourglass.slrm resume_from_model_id=jzlv54wl
# sbatch train_hourglass.slrm experiment=hvq/bounded/pfam_full.yaml ++hourglass.downproj_factor=128 ++hourglass.shorten_factor=2 ++logger.name="pfam_tanh_256_8" ++hourglass.lr=1e-4
# sbatch train_hourglass.slrm experiment=hvq/bounded/pfam_full.yaml ++hourglass.downproj_factor=256 ++hourglass.shorten_factor=1 ++logger.name="pfam_tanh_512_4" ++hourglass.lr=1e-4
# sbatch train_hourglass.slrm experiment=hvq/bounded/pfam_full.yaml ++hourglass.downproj_factor=256 ++hourglass.shorten_factor=2 ++logger.name="pfam_tanh_256_4" ++hourglass.lr=1e-4
sbatch train_hourglass.slrm experiment=hvq/bounded/pfam_full.yaml ++hourglass.downproj_factor=128 ++hourglass.shorten_factor=4 ++logger.name="pfam_tanh_128_4" ++hourglass.lr=1e-4

# sbatch train_hourglass.slrm experiment=hvq/bounded/rocklin.yaml
# sbatch train_hourglass.slrm experiment=hvq/bounded/pfam_full.yaml ++hourglass.downproj_factor=32 ++hourglass.shorten_factor=2 ++logger.name="pfam_tanh_256_32"
# sbatch train_hourglass.slrm experiment=hvq/bounded/pfam_full.yaml ++hourglass.downproj_factor=16 ++hourglass.shorten_factor=2 ++logger.name="pfam_tanh_256_64"

# sbatch train_hourglass.slrm experiment=hvq/fsq/pfam_full ++hourglass.fsq_levels=[16,16,16,16] ++logger.name="pfam_fsq_16_16_16_16" 
# sbatch train_hourglass.slrm experiment=hvq/fsq/pfam_full ++hourglass.fsq_levels=[8,8,8,6,5] ++logger.name="pfam_fsq_86"  

# sbatch train_hourglass.slrm experiment=hvq/bounded/cath_small.yaml ++hourglass.downproj_factor=128 ++logger.name="cath_tanh_512_8"
# sbatch train_hourglass.slrm experiment=hvq/bounded/cath_small.yaml ++hourglass.downproj_factor=32 ++hourglass.shorten_factor=2 ++logger.name="cath_tanh_256_32"
# sbatch train_hourglass.slrm experiment=hvq/bounded/cath_small.yaml ++hourglass.downproj_factor=16 ++hourglass.shorten_factor=2 ++logger.name="cath_tanh_256_64"

# sbatch train_hourglass.slrm experiment=hvq/bounded/pfam_full.yaml ++hourglass.downproj_factor=128 ++logger.name="pfam_tanh_512_8" ++hourglass.lr=6e-5
# sbatch train_hourglass.slrm experiment=hvq/bounded/pfam_full.yaml ++hourglass.downproj_factor=128 ++hourglass.shorten_factor=2 ++logger.name="pfam_tanh_256_8" ++hourglass.lr=6e-5
# sbatch train_hourglass.slrm experiment=hvq/bounded/uniref_full.yaml ++hourglass.downproj_factor=128 ++logger.name="uniref_tanh_512_8" ++hourglass.lr=6e-5