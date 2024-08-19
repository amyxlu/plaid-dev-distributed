"""
Inspiration:
- https://github.com/baofff/U-ViT/blob/main/libs/uvit.py

DiT-like blocks, with Flash Attention components:
* Fused MLP and multi-head attention kernels
* Rotary Embeddings
* Fourier feature timesteps
* AdaLN conditioning
* Skip connections
"""

import typing as T
from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from . import DenoiserKwargs
from .modules import BaseDenoiser, Mlp


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Block(nn.Module):
    """
    Block using Flash Attention components, optionally with:
    - ada-LN conditioning (https://arxiv.org/abs/2212.09748)
    - skip connections (https://arxiv.org/abs/2209.12152)
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads, qkv_bias=False, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, mask):
        """Apply multi-head attention (with mask) and adaLN conditioning (mask agnostic)."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), mask
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FunctionOrganismDenoiser(BaseDenoiser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def make_denoising_blocks(self):
        return nn.ModuleList(
            [
                Block(hidden_size=self.hidden_size, num_heads=self.num_heads)
                for _ in range(self.depth)
            ]
        )

