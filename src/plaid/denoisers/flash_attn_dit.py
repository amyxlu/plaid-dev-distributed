"""
Inspiration:
- https://github.com/baofff/U-ViT/blob/main/libs/uvit.py

DiT-like blocks, with Flash Attention components:
* Fused multi-head attention kernels
* Rotary or fixed sinusoidal positional embeddings
* Fourier or sinusoidal timestep transforms (with MLP projection)
* Optional: AdaLN conditioning
* Optional: Skip connections
"""

import typing as T
from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from . import DenoiserKwargs
from .modules import BaseDenoiser, Mlp, Attention


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

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_skip_connections=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        #####################
        # TODO: replace with flash attention
        self.attn = Attention(hidden_size, num_heads, qkv_bias=False, **block_kwargs)
        #####################

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

    def forward(self, x, c, mask, skip=False):
        """Apply multi-head attention (with mask) and adaLN conditioning (mask agnostic)."""
        if skip:
            # TODO: implement skip connections
            pass

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


class FunctionOrganismDiT(BaseDenoiser):
    def __init__(
        self,
        input_dim=8,
        hidden_size=512,
        max_seq_len=512,
        depth=6,
        num_heads=8,
        mlp_ratio=2.0,
        use_self_conditioning=True,
        timestep_embedding_strategy: T.Optional[str] = "fourier", 
        pos_embedding_strategy: T.Optional[str] = "rotary",
        use_skip_connections: bool = True,
    ):
        kwargs = dict(
            input_dim=input_dim,
            hidden_size=hidden_size,
            max_seq_len=max_seq_len,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            use_self_conditioning=use_self_conditioning,
            timestep_embedding_strategy=timestep_embedding_strategy,
            pos_embedding_strategy=pos_embedding_strategy,
        )
        self.use_skip_connections = use_skip_connections
        super().__init__(**kwargs)

    def make_denoising_blocks(self):
        return nn.ModuleList(
            [
                Block(hidden_size=self.hidden_size, num_heads=self.num_heads)
                for _ in range(self.depth)
            ]
        )
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize label embedding table:
        nn.init.normal_(self.organism_y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.function_y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

