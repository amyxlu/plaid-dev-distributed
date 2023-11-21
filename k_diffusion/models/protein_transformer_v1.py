"""Adapted from the image version, but:
* Model inputs are 1D instead of 2D. Attention shapes are mostly the same.
* Uses an embedder to create the ESMFold intermediate representation from strings.
* Uses Rotary Embeddings instead of the axial version
* Removes augmentation conditioningk-diffusion transformer diffusion models, version 1.

Model inputs are 1D instead of 2D.
Attention shapes are mostly the same, except that an attention mask is used for variable lengths. 
"""

import math
import typing as T
import random

from einops import rearrange, repeat
import torch
from torch import nn
import torch._dynamo
from torch.nn import functional as F

from . import flags
from .. import layers, utils, normalization 

# from .axial_rope import AxialRoPE, make_axial_pos
from .rope import RotaryEmbedding
from .esmfold import ESMFold, ESMFOLD_S_DIM


if flags.get_use_compile():
    torch._dynamo.config.suppress_errors = True

def maybe_unsqueeze(x):
    if len(x.shape) == 1:
        return x.unsqueeze(-1)
    return x



def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def xavier_init(module):
    for p in module.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)
    return module


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
        attn_mask = (attn_mask != 1).to(q.dtype) * -1e9
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
    def __init__(self, d_model, d_ff, d_head, skip=False, dropout=0.0):
        super().__init__()
        self.self_attn = SelfAttentionBlock(d_model, d_head, dropout=dropout)
        self.ff = FeedForwardBlock(d_model, d_ff, dropout=dropout)
        self.skip_linear = nn.Linear(d_model * 2, d_model) if skip else None

    # def forward(self, x, pos, attn_mask, cond):
    def forward(self, x, attn_mask, cond, skip=None):
        if self.skip_linear is not None:
            x = checkpoint_helper(self.skip_linear, torch.cat([x, skip], dim=-1))
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


class ProteinTransformerDenoiserModelV1(nn.Module):
    def __init__(
        self,
        n_layers,
        d_model,
        d_ff,
        d_head,
        skip_connect: bool = False,
        input_size: int = 512,
        input_dim: int = 1024, 
        min_len: int = 32,
        num_classes: int = 0,
        dropout: float = 0.0,
        sigma_data: float = 1.0,
    ):
        """Transformer denoiser model for proteins.
        input_dim: the dimension of the actual embedding needed for the frozen decoders (i.e. for ESMFold, this is 1024)
        d_model: the dimension of the latent embedding we will generate; might be a downprojection.
        """
        super().__init__()
        self.sigma_data = sigma_data  # i.e. the noise scale
        self.num_classes = num_classes # not used
        self.input_size = input_size  # i.e. length of protein
        self.input_dim = input_dim # i.e. the actual latent dimension for ESMFold
        assert input_dim == ESMFOLD_S_DIM
        self.d_model = d_model
        self.min_len = min_len

        # project from the ESMFold latent to the actual dimension for latent diffusion
        self.latent_in_proj = nn.Linear(input_dim, d_model, bias=False)

        self.esmfold_embedder = ESMFold(make_trunk=False)
        self.esmfold_embedder.eval()
        if not self.esmfold_embedder.trunk is None:
            self.esmfold_embedder.set_chunk_size(128)

        # TODO: what are these constants in the architecture for the initial codebase?
        self.time_emb = layers.FourierFeatures(1, d_model)
        self.time_in_proj = nn.Linear(d_model, d_model, bias=False)
        xavier_init(self.time_in_proj)

        # self.aug_emb = layers.FourierFeatures(9, d_model)
        # self.aug_in_proj = nn.Linear(d_model, d_model, bias=False)
        self.class_emb = nn.Embedding(num_classes, d_model) if num_classes else None

        self.mapping = tag_module(
            MappingNetwork(2, d_model, d_ff, dropout=dropout), "mapping"
        )

        self.in_proj = nn.Linear(d_model, d_model, bias=False)
        xavier_init(self.in_proj)

        if skip_connect:
            self.in_blocks = nn.ModuleList(
                [
                    TransformerBlock(d_model, d_ff, d_head, skip=False, dropout=dropout)
                    for _ in range(n_layers // 2)
                ]
            )
            self.mid_block = TransformerBlock(d_model, d_ff, d_head, skip=False, dropout=dropout)
            self.out_blocks = nn.ModuleList(
                [
                    TransformerBlock(d_model, d_ff, d_head, skip=skip_connect, dropout=dropout)
                    for _ in range(n_layers // 2)
                ]
            )
            xavier_init(self.in_blocks)
            xavier_init(self.mid_block)
            xavier_init(self.out_blocks)
            self.blocks = None
        else:
            # for backwards compatibility w/ checkpoint loading
            self.blocks = nn.ModuleList(
                [
                    TransformerBlock(d_model, d_ff, d_head, skip=False, dropout=dropout)
                    for _ in range(n_layers)
                ]
            )
            xavier_init(self.blocks)
            self.in_blocks, self.mid_block, self.out_blocks = None, None, None
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
        sequences = utils.get_random_sequence_crop_batch(sequences, self.input_size, self.min_len)

        with torch.no_grad():
            embeddings_dict = self.esmfold_embedder.infer_embedding(sequences)
        return embeddings_dict["s"], embeddings_dict["mask"]
    
    def project_to_d_model(self, x: torch.Tensor):
        assert x.shape[-1] == self.input_dim, "x must have the same last dimension as input_dim"
        return self.latent_in_proj(x)
    
    def project_to_input_dim(self, x: torch.Tensor):
        assert x.shape[-1] == self.d_model, "x must have the same last dimension as d_model"
        return x @ self.latent_in_proj.weight.data  # (batch_size, d_model) @ (d_model, input_dim)

    def raw_to_latent(self, x: torch.Tensor, normalization_mode: str = "channel_minmax"):
        # 2) Upproject latent space, maybe
        if self.model_config.d_model != self.model_config.input_dim:
            x = self.model.inner_model.project_to_input_dim(x)
        # 1) Normalize, maybe
        x = normalization.undo_scale_embedding(x, normalization_mode) 
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        mask: torch.Tensor,
        class_cond: T.Optional[torch.Tensor] = None,
    ):
        assert x.shape[-1] == self.d_model, "x must have the same dim as d_model; call self.project_to_latent(x) first."
        # Mapping network
        if class_cond is None and self.class_emb is not None:
            raise ValueError("class_cond must be specified if num_classes > 0")
        c_noise = torch.log(sigma) / 4
        time_emb = self.time_in_proj(self.time_emb(maybe_unsqueeze(c_noise)))
        # aug_cond = x.new_zeros([x.shape[0], 9]) if aug_cond is None else aug_cond
        # aug_emb = self.aug_in_proj(self.aug_emb(aug_cond))
        class_emb = self.class_emb(class_cond) if self.class_emb is not None else 0
        # cond = self.mapping(time_emb + aug_emb + class_emb).unsqueeze(1)
        cond = self.mapping(time_emb + class_emb).unsqueeze(1)
        
        # Transformer
        if self.in_blocks is None:
            for block in self.blocks:
                x = block(x, mask, cond)
            return x
        else:
            skips = []
            for i, block in enumerate(self.in_blocks):
                x = block(x, mask, cond)
                skips.append(x)
            x = self.mid_block(x, mask, cond)
            for i, block in enumerate(self.out_blocks):
                x = block(x, mask, cond, skip=skips.pop())
            return x