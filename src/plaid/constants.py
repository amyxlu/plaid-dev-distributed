import os
from pathlib import Path


structure_module_c_s = 384
structure_module_c_z = 128
c_s = 1024
c_z = 128

ICLR_MODEL_ID = "5j007z42"


# DEFAULT_CACHE = Path(os.environ['HOME']) / f".cache/cheap"
DEFAULT_CACHE = Path("/data/lux70/cheap")


# Weights to trained latent-to-sequence decoder
DECODER_CKPT_PATH = Path(DEFAULT_CACHE) / "sequence_decoder/mlp.ckpt"


# Directory to where per-channel statistics are stored
TENSOR_STATS_DIR = Path(DEFAULT_CACHE) / "statistics"


# Directory to where pre-trained models are stored
CHECKPOINT_DIR_PATH = Path(DEFAULT_CACHE) / "checkpoints"


ACCEPTED_LM_EMBEDDER_TYPES = [
    "esmfold",  # 1024 -- i.e. t36_3B with projection layers, used for final model
    "esmfold_pre_mlp",  # 2560
    "esm2_t48_15B_UR50D",  # 5120
    "esm2_t36_3B_UR50D",  # 2560
    "esm2_t33_650M_UR50D",  # 1280
    "esm2_t30_150M_UR50D",  # 64e $EMBED
    "esm2_t12_35M_UR50D",  # dim=480
    "esm2_t6_8M_UR50D",  # dim=320
]

COMPRESSED_DATA_STDS = {
    "qjs33lme": 0.3685586,
    "jzlv54wl": 0.5509415,
    "wiepwn5p": 0.5558105,
    "h9hzw1bp": 0.2876465,
    "j1v1wv6w": 0.3381734,
}


COMPRESSION_INPUT_DIMENSIONS = {
    "qjs33lme": 8,
    "jzlv54wl": 8,
    "wiepwn5p": 8,
    "h9hzw1bp": 64,
    "j1v1wv6w": 32,
    ## CATH, len 256 ##
    "8ebs7j9h": 4,
    "mm9fe6x9": 8,
    "kyytc8i9": 16,
    "fbbrfqzk": 32,
    "13lltqha": 64,
    "uhg29zk4": 128,
    "ich20c3q": 256,
    "7str7fhl": 512,
    "g8e83omk": 1024,
    ## CATH, len 512 ##
    "1b64t79h": 4,
    "1hr1x9r5": 8,
    "yfel5fnl": 16,
    "v2cer77t": 32,
    "2tjrgcde": 64,
    "3rs1hxky": 128,
    "5z4iaak9": 256,
    "q3m9fhii": 512,
}


COMPRESSION_SHORTEN_FACTORS = {
    "qjs33lme": 2,
    "jzlv54wl": 1,
    "wiepwn5p": 1,
    "h9hzw1bp": 2,
    "j1v1wv6w": 2,
    ## CATH, len 256 ##
    "8ebs7j9h": 2,
    "mm9fe6x9": 2,
    "kyytc8i9": 2,
    "fbbrfqzk": 2,
    "13lltqha": 2,
    "uhg29zk4": 2,
    "ich20c3q": 2,
    "7str7fhl": 2,
    "g8e83omk": 2,
    ## CATH, len 512 ##
    "q3m9fhii": 1,
    "5z4iaak9": 1,
    "3rs1hxky": 1,
    "2tjrgcde": 1,
    "v2cer77t": 1,
    "yfel5fnl": 1,
    "1hr1x9r5": 1,
    "1b64t79h": 1,
}

RESTYPES = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]


UNIREF_SS_BOUNDARIES = {
    "helix": [0.1647, 0.3294, 0.4941, 0.6589, 0.8236],
    "turn": [0.1628, 0.3255, 0.4883, 0.6510, 0.8138],
    "sheet": [0.1524, 0.3048, 0.4571, 0.6095, 0.7619],
}


CATH_SS_BOUNDARIES = {
    "helix": [0.2109375, 0.25, 0.2734375, 0.296875, 0.3203125],
    "turn": [0.171875, 0.2109375, 0.234375, 0.265625, 0.3046875],
    "sheet": [0.1875, 0.2265625, 0.2578125, 0.296875, 0.3359375],
}

# for quantizing secondary structure fraction for conditioning
NUM_SECONDARY_STRUCTURE_BINS = 6

# for foldseek
PDB_DATABASE_PATH = "/data/bucket/robins21/pdb"

# for mmseqs
UNIREF_DATABASE_PATH = "/data/lux70/data/uniref50db/uniref50db"