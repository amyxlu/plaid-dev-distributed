from collections import namedtuple

DenoiserKwargs = namedtuple(
    "DenoiserKwargs", ["x", "t", "function_y", "organism_y", "mask", "x_self_cond"]
)

from .modules import BaseDenoiser, BaseBlock
from .utsa import PreinitializedTriSelfAttnDenoiser, UTriSelfAttnDenoiser
from .dit import FunctionOrganismDiT
