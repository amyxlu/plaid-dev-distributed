import os
from pathlib import Path


structure_module_c_s = 384
structure_module_c_z = 128
c_s = 1024
c_z = 128


# There are 659 labelled clans. During data preprocessing, any protein without a clan
# is assigned the "unknown" clan index (roughly 50% of the dataset), i.e.
# there are actually 660 unique labels in the cached dataset, but we will
# also use this "unknown" index as the unconditional index at sampling time,
# so the number of clans specified here is 659.
NUM_CLANS = 659
DECODER_CKPT_PATH = Path(os.environ["HOME"]) / "plaid_cached_tensors/decoder_mlp.ckpt"
DECODER_SINGLE_LAYER_CKPT_PATH = Path(os.environ["HOME"]) / "plaid_cached_tensors/decoder_single_layer.ckpt"
CACHED_TENSORS_DIR = Path(os.environ["HOME"]) / "plaid_cached_tensors"


ACCEPTED_LM_EMBEDDER_TYPES = [
    "esmfold",  # 1024 -- i.e. t36_3B with projection layers, used for final model
    "esmfold_pre_mlp",  # 2560
    "esm2_t48_15B_UR50D",  # 5120 
    "esm2_t36_3B_UR50D",  # 2560
    "esm2_t33_650M_UR50D",  # 1280
    "esm2_t30_150M_UR50D",  # 64e $EMBED
    "esm2_t12_35M_UR50D",  # dim=480
    "esm2_t6_8M_UR50D"  # dim=320
]


COMPRESSED_DATA_STDS = {
    "qjs33lme": 0.3685586,
    "jzlv54wl": 0.5509415,
    "wiepwn5p": 0.5558105,
    "h9hzw1bp": 0.2876465,
    "j1v1wv6w": 0.3381734
}