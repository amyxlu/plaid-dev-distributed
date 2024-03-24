"""Hourglass Diffusion Models
https://arxiv.org/pdf/2401.11605.pdf

Adapted for 1D protein latent diffusion from
https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/models/image_transformer_v2.py
"""


from dataclasses import dataclass
from functools import lru_cache, reduce
import math
from typing import Union

from einops import rearrange
import torch
from torch import nn
import torch._dynamo
from torch.nn import functional as F

from plaid.denoisers.modules import flags, flops


try:
    import flash_attn
except ImportError:
    flash_attn = None


if flags.get_use_compile():
    torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
    torch._dynamo.config.suppress_errors = True


# Helpers

def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def checkpoint(function, *args, **kwargs):
    if flags.get_checkpointing():
        kwargs.setdefault("use_reentrant", True)
        return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)
    else:
        return function(*args, **kwargs)


def downscale_pos(pos):
    # ?????? TODO
    pos = rearrange(pos, "... (l nl) e -> ... l nl e", nl=2)
    # return torch.mean(pos, dim=-1)
    return pos


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
    q, k, v = qkv.unbind(2)
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
    def __init__(self, features, cond_features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.linear = apply_wd(zero_init(Linear(cond_features, features, bias=False)))
        tag_module(self.linear, "mapping")

    def extra_repr(self):
        return f"eps={self.eps},"

    def forward(self, x, cond):
        return rms_norm(x, self.linear(cond)[:, None, None, :] + 1, self.eps)


# Rotary position embeddings

@flags.compile_wrap
def apply_rotary_emb(x, theta, conj=False):
    out_dtype = x.dtype
    dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1]
    x1, x2, x3 = x[..., :d], x[..., d : d * 2], x[..., d * 2 :]
    x1, x2, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    y1, y2 = y1.to(out_dtype), y2.to(out_dtype)
    return torch.cat((y1, y2, x3), dim=-1)


@flags.compile_wrap
def _apply_rotary_emb_inplace(x, theta, conj):
    dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1]
    x1, x2 = x[..., :d], x[..., d : d * 2]
    x1_, x2_, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1_ * cos - x2_ * sin
    y2 = x2_ * cos + x1_ * sin
    x1.copy_(y1)
    x2.copy_(y2)


class ApplyRotaryEmbeddingInplace(torch.autograd.Function):
    @staticmethod
    def forward(x, theta, conj):
        _apply_rotary_emb_inplace(x, theta, conj=conj)
        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, theta, conj = inputs
        ctx.save_for_backward(theta)
        ctx.conj = conj

    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        _apply_rotary_emb_inplace(grad_output, theta, conj=not ctx.conj)
        return grad_output, None, None


def apply_rotary_emb_(x, theta):
    return ApplyRotaryEmbeddingInplace.apply(x, theta, False)


class RoPE(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        log_min = math.log(math.pi)
        log_max = math.log(10.0 * math.pi)
        freqs = torch.linspace(log_min, log_max, n_heads * dim // 4 + 1)[:-1].exp()
        self.register_buffer("freqs", freqs.view(dim // 4, n_heads).T.contiguous())

    def extra_repr(self):
        return f"dim={self.freqs.shape[1] * 4}, n_heads={self.freqs.shape[0]}"

    def forward(self, pos):
        print(pos.shape)
        theta = pos[..., None] * self.freqs.to(pos.dtype)
        print(theta.shape)
        return theta



# Transformer layers

def use_flash_2(x):
    if not flags.get_use_flash_attention_2():
        return False
    if flash_attn is None:
        return False
    if x.device.type != "cuda":
        return False
    if x.dtype not in (torch.float16, torch.bfloat16):
        return False
    return True

class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, d_head, cond_features, dropout=0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.norm = AdaRMSNorm(d_model, cond_features)
        self.qkv_proj = apply_wd(Linear(d_model, d_model * 3, bias=False))
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.pos_emb = RoPE(d_head // 2, self.n_heads) 
        self.dropout = nn.Dropout(dropout)
        self.out_proj = apply_wd(zero_init(Linear(d_model, d_model, bias=False)))

    def extra_repr(self):
        return f"d_head={self.d_head},"

    def forward(self, x, pos, cond):
        skip = x
        x = self.norm(x, cond)
        qkv = self.qkv_proj(x)
        pos = pos.to(qkv.dtype)
        theta = self.pos_emb(pos)
        if use_flash_2(qkv):
            qkv = rearrange(qkv, "n l (t nh e) -> n l t nh e", t=3, e=self.d_head)
            qkv = scale_for_cosine_sim_qkv(qkv, self.scale, 1e-6)
            theta = torch.stack((theta, theta, torch.zeros_like(theta)), dim=-3)
            qkv = apply_rotary_emb_(qkv, theta)
            flops_shape = qkv.shape[-5], qkv.shape[-2], qkv.shape[-4], qkv.shape[-1]
            flops.op(flops.op_attention, flops_shape, flops_shape, flops_shape)
            x = flash_attn.flash_attn_qkvpacked_func(qkv, softmax_scale=1.0)
            x = rearrange(x, "n l nh e -> n l (nh e)", l=skip.shape[1])
        else:
            q, k, v = rearrange(qkv, "n l (t nh e) -> t n nh l e", t=3, e=self.d_head)
            q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None], 1e-6)
            theta = theta.movedim(-2, -3)
            q = apply_rotary_emb_(q, theta)
            k = apply_rotary_emb_(k, theta)
            flops.op(flops.op_attention, q.shape, k.shape, v.shape)
            x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
            x = rearrange(x, "n nh l e -> n l (nh e)", l=skip.shape[1])
        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, cond_features, dropout=0.0):
        super().__init__()
        self.norm = AdaRMSNorm(d_model, cond_features)
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
    def __init__(self, d_model, d_ff, d_head, cond_features, dropout=0.0):
        super().__init__()
        self.self_attn = SelfAttentionBlock(d_model, d_head, cond_features, dropout=dropout)
        self.ff = FeedForwardBlock(d_model, d_ff, cond_features, dropout=dropout)

    def forward(self, x, pos, cond):
        x = checkpoint(self.self_attn, x, pos, cond)
        x = checkpoint(self.ff, x, cond)
        return x


class Level(nn.ModuleList):
    def forward(self, x, *args, **kwargs):
        for layer in self:
            x = layer(x, *args, **kwargs)
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
    def __init__(self, in_features, out_features, patch_len=2):
        super().__init__()
        self.l = patch_len
        self.proj = apply_wd(Linear(in_features * self.l, bias=False))

    def forward(self, x):
        x = rearrange(x, "... (l nl) e -> ... l (nl e)", nl=self.l)
        return self.proj(x)


class TokenSplitWithoutSkip(nn.Module):
    def __init__(self, in_features, out_features, patch_len=2):
        super().__init__()
        self.l = patch_len
        self.proj = apply_wd(Linear(in_features, out_features * self.l, bias=False))

    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, "... l (nl e) -> ... (l nl) e", nl=self.l)


class TokenSplit(nn.Module):
    def __init__(self, in_features, out_features, patch_len=2):
        super().__init__()
        self.l = patch_len 
        self.proj = apply_wd(Linear(in_features, out_features * self.l, bias=False))
        self.fac = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x, skip):
        x = self.proj(x)
        x = rearrange(x, "... l (nl e) -> ... (l nl) e", nl=self.l)
        return torch.lerp(skip, x, self.fac.to(x.dtype))


# Configuration

@dataclass
class GlobalAttentionSpec:
    d_head: int

@dataclass
class LevelSpec:
    depth: int
    width: int
    d_ff: int
    self_attn: GlobalAttentionSpec
    dropout: float


@dataclass
class MappingSpec:
    depth: int
    width: int
    d_ff: int
    dropout: float


# Model class

class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer('weight', torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


def make_levels(depths: int, widths: int, d_ffs: int, d_head: int, dropout_rate=0.0):
    levels = []
    for depth, width, d_ff, self_attn,dropout in zip(depths, widths, d_ffs, dropout_rate):
        self_attn = GlobalAttentionSpec(d_head)
        levels.append(LevelSpec(depth, width, d_ff, self_attn, dropout))
    return levels


class HDIT(nn.Module):
    def __init__(
        self,
        depths,
        widths,
        d_ffs,
        d_head,
        dropout_rate,
        mapping,
        num_classes=0,
        mapping_cond_dim=0
    ):
        super().__init__()
        self.num_classes = num_classes
        levels = make_levels(depths, widths, d_ffs, d_head, dropout_rate)

        self.time_emb = FourierFeatures(1, mapping.width)
        self.time_in_proj = Linear(mapping.width, mapping.width, bias=False)
        self.class_emb = nn.Embedding(num_classes, mapping.width) if num_classes else None
        self.mapping_cond_in_proj = Linear(mapping_cond_dim, mapping.width, bias=False) if mapping_cond_dim else None
        self.mapping = tag_module(MappingNetwork(mapping.depth, mapping.width, mapping.d_ff, dropout=mapping.dropout), "mapping")

        self.down_levels, self.up_levels = nn.ModuleList(), nn.ModuleList()
        for i, spec in enumerate(levels):
            # TODO: add other factories if trying other attentions too, e.g. SWIN or NATTEN
            layer_factory = lambda _: GlobalTransformerLayer(spec.width, spec.d_ff, spec.self_attn.d_head, mapping.width, dropout=spec.dropout)

            if i < len(levels) - 1:
                self.down_levels.append(Level([layer_factory(i) for i in range(spec.depth)]))
                self.up_levels.append(Level([layer_factory(i + spec.depth) for i in range(spec.depth)]))
            else:
                self.mid_level = Level([layer_factory(i) for i in range(spec.depth)])

        self.merges = nn.ModuleList([TokenMerge(spec_1.width, spec_2.width) for spec_1, spec_2 in zip(levels[:-1], levels[1:])])
        self.splits = nn.ModuleList([TokenSplit(spec_2.width, spec_1.width) for spec_1, spec_2 in zip(levels[:-1], levels[1:])])
        self.out_norm = RMSNorm(levels[0].width)

    def param_groups(self, base_lr=5e-4, mapping_lr_scale=1 / 3):
        wd = filter_params(lambda tags: "wd" in tags and "mapping" not in tags, self)
        no_wd = filter_params(lambda tags: "wd" not in tags and "mapping" not in tags, self)
        mapping_wd = filter_params(lambda tags: "wd" in tags and "mapping" in tags, self)
        mapping_no_wd = filter_params(lambda tags: "wd" not in tags and "mapping" in tags, self)
        groups = [
            {"params": list(wd), "lr": base_lr},
            {"params": list(no_wd), "lr": base_lr, "weight_decay": 0.0},
            {"params": list(mapping_wd), "lr": base_lr * mapping_lr_scale},
            {"params": list(mapping_no_wd), "lr": base_lr * mapping_lr_scale, "weight_decay": 0.0}
        ]
        return groups

    def forward(self, x, sigma, aug_cond=None, class_cond=None, mapping_cond=None):
        pos = torch.arange(x.shape[1]).to(x.device)

        # Mapping network
        if class_cond is None and self.class_emb is not None:
            raise ValueError("class_cond must be specified if num_classes > 0")
        if mapping_cond is None and self.mapping_cond_in_proj is not None:
            raise ValueError("mapping_cond must be specified if mapping_cond_dim > 0")

        c_noise = torch.log(sigma) / 4
        time_emb = self.time_in_proj(self.time_emb(c_noise[..., None]))
        aug_cond = x.new_zeros([x.shape[0], 9]) if aug_cond is None else aug_cond
        class_emb = self.class_emb(class_cond) if self.class_emb is not None else 0
        mapping_emb = self.mapping_cond_in_proj(mapping_cond) if self.mapping_cond_in_proj is not None else 0
        cond = self.mapping(time_emb + class_emb + mapping_emb)

        # Hourglass transformer
        skips, poses = [], []
        for down_level, merge in zip(self.down_levels, self.merges):
            x = down_level(x, pos, cond)
            skips.append(x)
            poses.append(pos)
            x = merge(x)
            pos = downscale_pos(pos)

        x = self.mid_level(x, pos, cond)

        for up_level, split, skip, pos in reversed(list(zip(self.up_levels, self.splits, skips, poses))):
            x = split(x, skip)
            x = up_level(x, pos, cond)

        x = self.out_norm(x)
        return x

import IPython;IPython.embed()
model = HDIT()