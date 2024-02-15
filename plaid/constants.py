import os
from pathlib import Path


structure_module_c_s = 384
structure_module_c_z = 128
c_s = 1024
c_z = 128

DECODER_CKPT_PATH = Path(os.environ["HOME"]) / "plaid_cached_tensors/decoder_mlp.ckpt"
CACHED_TENSORS_DIR = Path(os.environ["HOME"]) / "plaid_cached_tensors"
