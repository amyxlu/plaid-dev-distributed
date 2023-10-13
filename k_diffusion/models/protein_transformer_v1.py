"""k-diffusion transformer diffusion models, version 1."""

import math
import typing as T

from einops import rearrange, repeat
import torch
from torch import nn
import torch._dynamo
from torch.nn import functional as F

from . import flags
from .. import layers

# from .axial_rope import AxialRoPE, make_axial_pos
from .rope import RotaryEmbedding
from .esmfold import ESMFold, ESMFoldConfig


ESMFOLD_S_DIM = 1024  # dimension of the s_s_0 tensor input to ESMFold folding trunk


if flags.get_use_compile():
    torch._dynamo.config.suppress_errors = True

def maybe_unsqueeze(x):
    if len(x.shape) == 1:
        return x.unsqueeze(1)
    return x  

def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def checkpoint_helper(function, *args, **kwargs):
    if flags.get_checkpointing():
        kwargs.setdefault("use_reentrant", True)
        return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)
    else:
        return function(*args, **kwargs)


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


def scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0):
    # removed transpose operations to make 1D
    if flags.get_use_flash_attention_2() and attn_mask is None:
        try:
            from flash_attn import flash_attn_func
            return flash_attn_func(q, k, v, dropout_p=dropout_p)
        except (ImportError, RuntimeError):
            pass
    return F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p=dropout_p)


@flags.compile_wrap
def geglu(x):
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


@flags.compile_wrap
def rms_norm(x, scale, eps):
    dtype = torch.promote_types(x.dtype, torch.float32)
    mean_sq = torch.mean(x.to(dtype) ** 2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)


class GEGLU(nn.Module):
    def forward(self, x):
        return geglu(x)


class RMSNorm(nn.Module):
    def __init__(self, param_shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(param_shape))

    def extra_repr(self):
        return f"shape={tuple(self.scale.shape)}, eps={self.eps}"

    def forward(self, x):
        return rms_norm(x, self.scale, self.eps)


class QKNorm(nn.Module):
    def __init__(self, n_heads, eps=1e-6, max_scale=100.0):
        super().__init__()
        self.eps = eps
        self.max_scale = math.log(max_scale)
        self.scale = nn.Parameter(torch.full((n_heads,), math.log(10.0)))
        self.proj_()

    def extra_repr(self):
        return f"n_heads={self.scale.shape[0]}, eps={self.eps}"

    @torch.no_grad()
    def proj_(self):
        """Modify the scale in-place so it doesn't get "stuck" with zero gradient if it's clamped
        to the max value."""
        self.scale.clamp_(max=self.max_scale)

    def forward(self, x):
        self.proj_()
        scale = torch.exp(0.5 * self.scale - 0.25 * math.log(x.shape[-1]))
        return rms_norm(x, scale[:, None, None], self.eps)


class AdaRMSNorm(nn.Module):
    def __init__(self, features: int, cond_features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.linear = apply_wd(
            zero_init(nn.Linear(cond_features, features, bias=False))
        )
        tag_module(self.linear, "mapping")

    def extra_repr(self):
        return f"eps={self.eps},"

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        return rms_norm(x, self.linear(cond) + 1, self.eps)


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, d_head, dropout=0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.norm = AdaRMSNorm(d_model, d_model)
        self.qkv_proj = apply_wd(nn.Linear(d_model, d_model * 3, bias=False))
        self.qk_norm = QKNorm(self.n_heads)
        # self.pos_emb = AxialRoPE(d_head, self.n_heads)
        self.rotary_emb = RotaryEmbedding(d_head)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = apply_wd(zero_init(nn.Linear(d_model, d_model, bias=False)))

    def extra_repr(self):
        return f"d_head={self.d_head},"

    def forward(self, x, attn_mask, cond):
        skip = x
        x = self.norm(x, cond)
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = rearrange(q, "n l (h e) -> n h l e", e=self.d_head)
        k = rearrange(k, "n l (h e) -> n h l e", e=self.d_head)
        v = rearrange(v, "n l (h e) -> n h l e", e=self.d_head)
        q, k = self.rotary_emb(self.qk_norm(q), self.qk_norm(k))
        attn_mask = attn_mask.to(torch.bool)
        attn_mask = repeat(attn_mask, "n l -> n h l s", h=self.n_heads, s=attn_mask.shape[-1])
        x = scaled_dot_product_attention(q, k, v, attn_mask)
        x = rearrange(x, "n h l e -> n l (h e)")
        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.norm = AdaRMSNorm(d_model, d_model)
        self.up_proj = apply_wd(nn.Linear(d_model, d_ff * 2, bias=False))
        self.act = GEGLU()
        self.dropout = nn.Dropout(dropout)
        self.down_proj = apply_wd(zero_init(nn.Linear(d_ff, d_model, bias=False)))

    def forward(self, x, cond):
        skip = x
        x = self.norm(x, cond)
        x = self.up_proj(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip


class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, d_head, dropout=0.0):
        super().__init__()
        self.self_attn = SelfAttentionBlock(d_model, d_head, dropout=dropout)
        self.ff = FeedForwardBlock(d_model, d_ff, dropout=dropout)

    # def forward(self, x, pos, attn_mask, cond):
    def forward(self, x, attn_mask, cond):
        x = checkpoint_helper(self.self_attn, x, attn_mask, cond)
        x = checkpoint_helper(self.ff, x, cond)
        return x


class MappingFeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.up_proj = apply_wd(nn.Linear(d_model, d_ff * 2, bias=False))
        self.act = GEGLU()
        self.dropout = nn.Dropout(dropout)
        self.down_proj = apply_wd(zero_init(nn.Linear(d_ff, d_model, bias=False)))

    def forward(self, x):
        skip = x
        x = self.norm(x)
        x = self.up_proj(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip


class MappingNetwork(nn.Module):
    def __init__(self, n_layers, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.in_norm = RMSNorm(d_model)
        self.blocks = nn.ModuleList(
            [
                MappingFeedForwardBlock(d_model, d_ff, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.out_norm = RMSNorm(d_model)

    def forward(self, x):
        x = self.in_norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        return x


class TransformerDenoiserModelV1(nn.Module):
    def __init__(
        self,
        n_layers,
        d_model,
        d_ff,
        d_head,
        num_classes=0,
        dropout=0.0,
        sigma_data=1.0,
    ):
        super().__init__()
        self.sigma_data = sigma_data
        self.num_classes = num_classes

        assert (
            d_model == ESMFOLD_S_DIM
        ), "d_model must match ESMFold embedding dimension"

        self.esmfold_embedder = ESMFold(make_trunk=False)
        self.esmfold_embedder.eval()

        # TODO: what are these constants in the architecture for the initial codebase?
        self.time_emb = layers.FourierFeatures(1, d_model)
        self.time_in_proj = nn.Linear(d_model, d_model, bias=False)
        self.aug_emb = layers.FourierFeatures(9, d_model)
        self.aug_in_proj = nn.Linear(d_model, d_model, bias=False)
        self.class_emb = nn.Embedding(num_classes, d_model) if num_classes else None
        self.mapping = tag_module(
            MappingNetwork(2, d_model, d_ff, dropout=dropout), "mapping"
        )
        # self.mapping = tag_module(
        #     MappingNetwork(1, d_model, d_ff, dropout=dropout), "mapping"
        # )

        self.in_proj = nn.Linear(d_model, d_model, bias=False)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, d_ff, d_head, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.out_norm = RMSNorm(d_model)
        self.out_proj = zero_init(nn.Linear(d_model, d_model, bias=False))

    def proj_(self):
        for block in self.blocks:
            block.self_attn.qk_norm.proj_()

    def param_groups(self, base_lr=5e-4, mapping_lr_scale=1 / 3):
        wd = filter_params(lambda tags: "wd" in tags and "mapping" not in tags, self)
        no_wd = filter_params(
            lambda tags: "wd" not in tags and "mapping" not in tags, self
        )
        mapping_wd = filter_params(
            lambda tags: "wd" in tags and "mapping" in tags, self
        )
        mapping_no_wd = filter_params(
            lambda tags: "wd" not in tags and "mapping" in tags, self
        )
        groups = [
            {"params": list(wd), "lr": base_lr},
            {"params": list(no_wd), "lr": base_lr, "weight_decay": 0.0},
            {"params": list(mapping_wd), "lr": base_lr * mapping_lr_scale},
            {
                "params": list(mapping_no_wd),
                "lr": base_lr * mapping_lr_scale,
                "weight_decay": 0.0,
            },
        ]
        return groups

    def embed_from_sequences(self, sequences: T.List[str]):
        """
        Create the ESMFold intermediate representation from strings.
        Used for training only.
        """
        with torch.no_grad():
            embeddings_dict = self.esmfold_embedder.infer_embedding(sequences)
        return embeddings_dict["s"], embeddings_dict["mask"]

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        sigma: torch.Tensor,
        aug_cond: T.Optional[torch.Tensor] = None,
        class_cond: T.Optional[torch.Tensor] = None,
    ):
        # Mapping network
        if class_cond is None and self.class_emb is not None:
            raise ValueError("class_cond must be specified if num_classes > 0")

        N, L, C = x.shape
        c_noise = torch.log(sigma) / 4
        time_emb = self.time_in_proj(self.time_emb(c_noise))
        aug_cond = x.new_zeros([x.shape[0], 9]) if aug_cond is None else aug_cond
        aug_emb = self.aug_in_proj(self.aug_emb(aug_cond))
        class_emb = self.class_emb(class_cond) if self.class_emb is not None else 0
        cond = self.mapping(time_emb + aug_emb + class_emb).unsqueeze(1)

        # Transformer
        for i, block in enumerate(self.blocks):
            x = block(x, mask, cond)

        return x