"""k-diffusion transformer diffusion models, version 2."""

from dataclasses import dataclass
import einops
from functools import lru_cache, reduce
import math
from typing import Union

from einops import rearrange
import torch
from torch import nn
import torch._dynamo
from torch.nn import functional as F

from .. import flags
from .. import flops
from .modules.embedders import LabelEmbedder, TimestepEmbedder

from xformers.components.positional_embedding import RotaryEmbedding
from xformers.ops import unbind


# To use compile:
# export DIFFUSION_USE_COMPILE=1
# see flags.py


if flags.get_use_compile():
    torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
    torch._dynamo.config.suppress_errors = True


# Helpers

def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


# trades off computation time for memory efficiency
# https://pytorch.org/docs/stable/checkpoint.html
def checkpoint(function, *args, **kwargs):
    if flags.get_checkpointing():
        kwargs.setdefault("use_reentrant", True)
        return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)
    else:
        return function(*args, **kwargs)

def downscale_pos(pos):
    pos = rearrange(pos, "... (l nl) e -> ... l nl e", nl=2)
    return torch.mean(pos, dim=-1)


# Param tags

def tag_param(param, tag):
    if not hasattr(param, "_tags"):
        param._tags = set([tag])
    else:
        param._tags.add(tag)
    return param


def tag_module(module, tag):
    for param in module.parameters():
        tag_param(param, tag)
    return module


def apply_wd(module):
    for name, param in module.named_parameters():
        if name.endswith("weight"):
            tag_param(param, "wd")
    return module


def filter_params(function, module):
    for param in module.parameters():
        tags = getattr(param, "_tags", set())
        if function(tags):
            yield param


# Kernels

@flags.compile_wrap
def linear_geglu(x, weight, bias=None):
    x = x @ weight.mT
    if bias is not None:
        x = x + bias
    x, gate = x.chunk(2, dim=-1)
    return x * F.gelu(gate)


@flags.compile_wrap
def rms_norm(x, scale, eps):
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype)**2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)


@flags.compile_wrap
def scale_for_cosine_sim(q, k, scale, eps):
    dtype = reduce(torch.promote_types, (q.dtype, k.dtype, scale.dtype, torch.float32))
    sum_sq_q = torch.sum(q.to(dtype)**2, dim=-1, keepdim=True)
    sum_sq_k = torch.sum(k.to(dtype)**2, dim=-1, keepdim=True)
    sqrt_scale = torch.sqrt(scale.to(dtype))
    scale_q = sqrt_scale * torch.rsqrt(sum_sq_q + eps)
    scale_k = sqrt_scale * torch.rsqrt(sum_sq_k + eps)
    return q * scale_q.to(q.dtype), k * scale_k.to(k.dtype)


@flags.compile_wrap
def scale_for_cosine_sim_qkv(qkv, scale, eps):
    q, k, v = unbind(qkv, 2)
    q, k = scale_for_cosine_sim(q, k, scale[:, None], eps)
    return torch.stack((q, k, v), dim=2)


# Layers

class Linear(nn.Linear):
    def forward(self, x):
        flops.op(flops.op_linear, x.shape, self.weight.shape)
        return super().forward(x)


class LinearGEGLU(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features * 2, bias=bias)
        self.out_features = out_features

    def forward(self, x):
        flops.op(flops.op_linear, x.shape, self.weight.shape)
        return linear_geglu(x, self.weight, self.bias)


class RMSNorm(nn.Module):
    def __init__(self, shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(shape))

    def extra_repr(self):
        return f"shape={tuple(self.scale.shape)}, eps={self.eps}"

    def forward(self, x):
        return rms_norm(x, self.scale, self.eps)


class AdaRMSNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.linear = apply_wd(zero_init(Linear(features, features, bias=False)))
        tag_module(self.linear, "mapping")

    def extra_repr(self):
        return f"eps={self.eps},"

    def forward(self, x, cond):
        return rms_norm(x, self.linear(cond)[:, None, :] + 1, self.eps)


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, d_head, dropout=0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.norm = AdaRMSNorm(d_model)
        self.qkv_proj = apply_wd(Linear(d_model, d_model * 3, bias=False))
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.pos_emb = RotaryEmbedding(d_head)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = apply_wd(zero_init(Linear(d_model, d_model, bias=False)))

    def extra_repr(self):
        return f"d_head={self.d_head},"

    def forward(self, x, pos, mask, cond):
        # x: (N, L, C)
        # cond: (N, ...) 
        skip = x
        x = self.norm(x, cond)
        qkv = self.qkv_proj(x)

        q, k, v = rearrange(qkv, "n l (t nh e) -> t n nh l e", t=3, e=self.d_head)
        q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None], 1e-6)
        q, k = self.pos_emb(q, k)
        flops.op(flops.op_attention, q.shape, k.shape, v.shape)
        
        x = F.scaled_dot_product_attention(q, k, v, scale=1.0, attn_mask=mask)
        x = rearrange(x, "n nh l e -> n l (nh e)", l=skip.shape[1])

        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.norm = AdaRMSNorm(d_model)
        self.up_proj = apply_wd(LinearGEGLU(d_model, d_ff, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.down_proj = apply_wd(zero_init(Linear(d_ff, d_model, bias=False)))

    def forward(self, x, cond):
        skip = x
        x = self.norm(x, cond)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip


class GlobalTransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_head, dropout=0.0, *args, **kwargs):
        super().__init__()
        self.self_attn = SelfAttentionBlock(d_model, d_head, dropout=dropout)
        self.ff = FeedForwardBlock(d_model, d_ff, dropout=dropout)

    def forward(self, x, pos, mask, cond):
        x = checkpoint(self.self_attn, x, pos, mask, cond)
        x = checkpoint(self.ff, x, cond)
        return x


# Mapping network

class MappingFeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.up_proj = apply_wd(LinearGEGLU(d_model, d_ff, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.down_proj = apply_wd(zero_init(Linear(d_ff, d_model, bias=False)))

    def forward(self, x):
        skip = x
        x = self.norm(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip


class MappingNetwork(nn.Module):
    def __init__(self, n_layers, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.in_norm = RMSNorm(d_model)
        self.blocks = nn.ModuleList([MappingFeedForwardBlock(d_model, d_ff, dropout=dropout) for _ in range(n_layers)])
        self.out_norm = RMSNorm(d_model)

    def forward(self, x):
        x = self.in_norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        return x


# Token merging and splitting

class TokenMerge(nn.Module):
    def __init__(self, in_features, out_features, fold_length=2):
        super().__init__()
        self.nl = fold_length
        self.proj = apply_wd(Linear(in_features * self.nl, out_features, bias=False))

    def forward(self, x):
        x = rearrange(x, "... (l nl) e -> ... l (nl e)", nl=self.nl)
        return self.proj(x)


class TokenSplitWithoutSkip(nn.Module):
    def __init__(self, in_features, out_features, fold_length=2):
        super().__init__()
        self.nl = fold_length
        self.proj = apply_wd(Linear(in_features, out_features * self.nl, bias=False))

    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, "... l (nl e) -> ... (l nl) e", nl=self.nl)


class TokenSplit(nn.Module):
    def __init__(self, in_features, out_features, fold_length=2):
        super().__init__()
        self.nl = fold_length
        self.proj = apply_wd(Linear(in_features, out_features * self.nl, bias=False))
        self.fac = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x, skip):
        x = self.proj(x)
        x = rearrange(x, "... l (nl e) -> ... (l nl) e", nl=self.nl)
        return torch.lerp(skip, x, self.fac.to(x.dtype))


## Model class


class HDiT(nn.Module):
    """Modified from HDiT and original DiT for 1D joint embeddings of protein sequence and structure."""

    def __init__(
        self,
        input_dim=8,
        hidden_size=1024,
        max_seq_len=512, 
        fold_length=2,
        depth=15,
        num_heads=16,
        mlp_ratio=4.0,
        use_self_conditioning=False,
        class_dropout_prob=0.1,
        num_classes=660,
        mapping_depth=8,
        d_ff=512,
        d_head=64,
    ):
        super().__init__()

        # dimensions for 1D diffusion
        self.max_seq_len = max_seq_len
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.use_self_conditioning = use_self_conditioning

        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio
        self.depth = depth

        assert depth % 2 == 1, "Depth must an odd number."

        # Simple projection layer from input latent dimension up to higher dimension
        self.x_proj = Linear(input_dim, hidden_size, bias=False)
        if self.use_self_conditioning:
            self.x_cond_proj = Linear(input_dim * 2, input_dim)

        # embedder modules inspired by DiT
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        # mapping network for combining conditioning information
        self.mapping = tag_module(MappingNetwork(mapping_depth, hidden_size, hidden_size), "mapping")

        # hierarchical levels
        self.down_levels, self.up_levels = nn.ModuleList(), nn.ModuleList()

        for i in range(depth // 2):
            self.down_levels.append(GlobalTransformerLayer(hidden_size, d_ff, d_head))
            self.up_levels.append(GlobalTransformerLayer(hidden_size, d_ff, d_head))
        self.mid_level = GlobalTransformerLayer(hidden_size, d_ff, d_head)

        self.merges = nn.ModuleList([TokenMerge(hidden_size, hidden_size) for _ in range(depth - 1)])
        self.splits = nn.ModuleList([TokenSplit(hidden_size, hidden_size) for _ in range(depth - 1)])

        self.out_norm = RMSNorm(hidden_size)
        self.patch_out = TokenSplitWithoutSkip(hidden_size, input_dim, fold_length)
        nn.init.zeros_(self.patch_out.proj.weight)

    def forward(self, x, t, mask=None, y=None, x_self_cond=None, train_mode=True):
        if x_self_cond is not None:
            x = torch.cat([x, x_self_cond], dim=-1)
            x_self_cond = self.x_cond_proj(x)
        x = self.x_proj(x)

        pos = einops.repeat(torch.arange(x.shape[1], device=x.device), "l -> b l", b=x.shape[0])

        t = self.t_embedder(t)

        if mask is None:
            mask = torch.ones_like(x).bool

        if not y is None:
            y = self.y_embedder(y, train=train_mode)
            c = self.mapping(t + y)
        else:
            c = self.mapping(t)

        # Hourglass transformer
        skips, poses = [], []
        for down_level, merge in zip(self.down_levels, self.merges):
            x = down_level(x, pos, mask, c)
            skips.append(x)
            poses.append(pos)
            x = merge(x)
            pos = downscale_pos(pos)

        x = self.mid_level(x, pos, mask, c)

        for up_level, split, skip, pos in reversed(list(zip(self.up_levels, self.splits, skips, poses))):
            x = split(x, skip)
            x = up_level(x, pos, mask, c)

        x = self.out_norm(x)
        x = self.patch_out(x)
        return x
    