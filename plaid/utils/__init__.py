import os
from pathlib import Path

# hack - place just one level above
# CACHED_TENSORS_DIR = Path(os.path.dirname(__file__)) / "../../cached_tensors"
CACHED_TENSORS_DIR = Path(os.environ['HOME']) / "plaid_cached_tensors" 

from ._misc import *
from ._gns import DDPGradientStatsHook, GradientNoiseScale
from ._normalization import LatentScaler
from ._tmalign import run_tmalign, max_tm_across_refs
from ._lr_schedulers import get_lr_scheduler
from ._protein_properties import sequences_to_secondary_structure_fracs
from ._structure import StructureFeaturizer
