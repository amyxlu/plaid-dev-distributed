import typing as T
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from . import DenoiserKwargs
from .modules import (
    BaseDenoiser,
    Mlp,
    Attention,
    LabelEmbedder,
    get_1d_sincos_pos_embed,
)

# from .modules._embedders import SinusoidalTimestepEmbedder as TimestepEmbedder
from ..datasets import NUM_FUNCTION_CLASSES, NUM_ORGANISM_CLASSES


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class InputProj(nn.Module):
    def __init__(self, input_dim, hidden_size, bias=True):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_size, bias=bias)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        return self.norm(self.proj(x))


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiTBlock(nn.Module):
    """
    Block using Flash Attention components, optionally with:
    - ada-LN conditioning (https://arxiv.org/abs/2212.09748)
    - skip connections (https://arxiv.org/abs/2209.12152)
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_skip_connect=False,
        use_xformers=False,
        **block_kwargs
    ):
        super().__init__()
        self.use_skip_connect = use_skip_connect
        self.attn = Attention(
            hidden_size,
            num_heads,
            qkv_bias=False,
            use_xformers=use_xformers,
            **block_kwargs
        )
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
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
        timestep_embedding_strategy="fourier",
        use_skip_connect=False,
        use_xformers=False,
    ):
        self.use_xformers = use_xformers
        self.use_skip_connect = use_skip_connect

        super().__init__(
            input_dim=input_dim,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            use_self_conditioning=use_self_conditioning,
            timestep_embedding_strategy=timestep_embedding_strategy,
            max_seq_len=max_seq_len,
        )

        self.initialize_weights()
        self.initialize_adaln_weights()

    def make_output_projection(self):
        return FinalLayer(self.hidden_size, self.input_dim)

    def make_denoising_blocks(self):
        return nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    use_skip_connect=self.use_skip_connect,
                    use_xformers=self.use_xformers,
                )
                for _ in range(self.depth)
            ]
        )

    def initialize_adaln_weights(self):
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
