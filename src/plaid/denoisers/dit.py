# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
from functools import partial
from einops import rearrange

from .modules.helpers import to_2tuple
from .modules.embedders import TimestepEmbedder, get_1d_sincos_pos_embed


def exists(val):
    return val is not None


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        qkv_bias = False,
        dropout = 0.,
    ):
        super().__init__()
        self.heads = heads
        dim_head = dim // heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = qkv_bias)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None):
        h, device = self.heads, x.device
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b l (h d) -> b h l d', h = h), (q, k, v))

        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        mask_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = rearrange(mask, 'b j -> b () () j')
            sim = sim.masked_fill(~mask, mask_value)

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h l d -> b l (h d)', h = h)
        return self.to_out(out)


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/mlp.py
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = nn.LayerNorm(hidden_features, elementwise_affine=False, eps=1e-6)
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads, qkv_bias=False, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, mask):
        """Apply multi-head attention (with mask) and adaLN conditioning (mask agnostic)."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, max_seq_len, out_channels):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class InputProj(nn.Module):
    """
    Initial x projection, serves a similar purpose as the patch embed
    """
    def __init__(self, input_dim, hidden_size, bias=True):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_size, bias=bias)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    
    def forward(self, x):
        return self.norm(self.proj(x))
    

class BaseDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    Adapted to remove y_embedder and patch embeddings
    """
    def __init__(
        self,
        input_dim=8,
        hidden_size=1024,
        max_seq_len=512, 
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        use_self_conditioning=False
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.use_self_conditioning = use_self_conditioning

        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio
        self.depth = depth

        self.x_proj = InputProj(input_dim, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size), requires_grad=False)
        if self.use_self_conditioning:
            self.self_conditioning_mlp = Mlp(input_dim * 2, input_dim)
        self.final_layer = FinalLayer(hidden_size, max_seq_len, input_dim)
        self.make_blocks()
        self.initialize_base_weights()
        self.initialize_block_weights()

    def initialize_base_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], np.arange(self.max_seq_len))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def make_blocks(self):
        raise NotImplementedError
    
    def initialize_block_weights(self):
        raise NotImplementedError

    def forward(*args, **kwargs):
        raise NotImplementedError
    

class SimpleDiT(BaseDiT):
    def __init__(
        self,
        input_dim=8,
        hidden_size=1024,
        max_seq_len=512, 
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        use_self_conditioning=False
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_size=hidden_size,
            max_seq_len=max_seq_len,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            use_self_conditioning=use_self_conditioning,
        )
    
    def make_blocks(self):
        self.blocks = nn.ModuleList([
            DiTBlock(self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio) for _ in range(self.depth)
        ])

    def initialize_block_weights(self):
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(self, x, t, mask=None, x_self_cond=None):
        """
        Forward pass of DiT.
        x: (N, L, C_compressed) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        mask: (N, L) mask where True means to attend and False denotes padding
        x_self_cond: (N, L, C_compressed) Optional tensor for self-condition
        """
        if x_self_cond is not None:
            x = self.self_conditioning_mlp(torch.cat([x, x_self_cond], dim=-1))
        x = self.x_proj(x)
        x += self.pos_embed[:, :x.shape[1], :]
        t = self.t_embedder(t)                   # (N, D)
        c = t  # TODO: add y embedding and clf guidance
        
        if mask is None:
            mask = torch.ones(x.shape[:2], device=x.device).bool()

        for block in self.blocks:
            x = block(x, c, mask)                # (N, L, D)
        x = self.final_layer(x, c)               # (N, L, out_channels)
        return x
    