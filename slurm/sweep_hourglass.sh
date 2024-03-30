sbatch train_hourglass.slrm "experiment=hvq/bounded/uniref_subset.yaml ++hourglass.lr=5e-5 ++hourglass.downproj_factor=256"
sbatch train_hourglass.slrm "experiment=hvq/bounded/uniref_subset.yaml ++hourglass.lr=5e-5 ++hourglass.downproj_factor=256 ++hourglass.shorten_factor=2"

sbatch train_hourglass.slrm "experiment=hvq/bounded/uniref_subset.yaml ++hourglass.lr=5e-5 ++hourglass.downproj_factor=128"
sbatch train_hourglass.slrm "experiment=hvq/bounded/uniref_subset.yaml ++hourglass.lr=5e-5 ++hourglass.downproj_factor=128 ++hourglass.shorten_factor=2"

sbatch train_hourglass.slrm "experiment=hvq/bounded/uniref_subset.yaml ++hourglass.lr=5e-5 ++hourglass.downproj_factor=64"
sbatch train_hourglass.slrm "experiment=hvq/bounded/uniref_subset.yaml ++hourglass.lr=5e-5 ++hourglass.downproj_factor=64 ++hourglass.shorten_factor=2"

sbatch train_hourglass.slrm "experiment=hvq/bounded/uniref_subset.yaml ++hourglass.lr=5e-5 ++hourglass.downproj_factor=32"
sbatch train_hourglass.slrm "experiment=hvq/bounded/uniref_subset.yaml ++hourglass.lr=5e-5 ++hourglass.downproj_factor=32 ++hourglass.shorten_factor=2"

sbatch train_hourglass.slrm "experiment=hvq/bounded/uniref_subset.yaml ++hourglass.lr=5e-5 ++hourglass.downproj_factor=16"
sbatch train_hourglass.slrm "experiment=hvq/bounded/uniref_subset.yaml ++hourglass.lr=5e-5 ++hourglass.downproj_factor=16 ++hourglass.shorten_factor=2"