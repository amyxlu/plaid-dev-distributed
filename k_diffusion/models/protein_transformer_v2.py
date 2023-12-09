"""Adapted from the image version, but:
* Model inputs are 1D instead of 2D. Attention shapes are mostly the same.
* Uses an embedder to create the ESMFold intermediate representation from strings.
* Uses Rotary Embeddings instead of the axial version
* Removes augmentation conditioningk-diffusion transformer diffusion models, version 1.

Model inputs are 1D instead of 2D.
Attention shapes are mostly the same, except that an attention mask is used for variable lengths. 
"""
from pathlib import Path

import math
import typing as T

from einops import rearrange, repeat
import torch
from torch import nn
import torch._dynamo
from torch.nn import functional as F

from . import flags
from .. import layers, utils, normalization, config

# from .axial_rope import AxialRoPE, make_axial_pos
from .rope import RotaryEmbedding
from .esmfold import ESMFold, ESMFOLD_S_DIM


ACCEPTED_LM_EMBEDDER_TYPES = [
    "esmfold",  # 1024 -- i.e. t36_3B with projection layers, used for final model
    "esm2_t48_15B_UR50D",  # 5120
    "esm2_t36_3B_UR50D",  # 2560
    "esm2_t33_650M_UR50D",  # 1280
    "esm2_t30_150M_UR50D",  # 640
    "esm2_t12_35M_UR50D",  # dim=480
    "esm2_t6_8M_UR50D",  # dim=320
]


if flags.get_use_compile():
    torch._dynamo.config.suppress_errors = True


def maybe_unsqueeze(x):
    if x.ndim == 1:
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
    mean_sq = torch.mean(x**2, dim=-1, keepdim=True)
    scale = scale.to(x.dtype) * torch.rsqrt(mean_sq + eps)
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


class ProjectionFeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.act = nn.SiLU()
        self.down_proj = apply_wd(nn.Linear(d_model * 2, d_ff, bias=True))
        self.dropout = nn.Dropout(dropout)
        self.linear = apply_wd(nn.Linear(d_ff, d_ff))
        self.norm = RMSNorm(d_ff)
        self.out_proj = apply_wd(zero_init(nn.Linear(d_ff, d_model, bias=True)))

    def forward(self, x, cond):
        cond = repeat(cond, "n c -> n l c", l=x.shape[1])
        x = torch.cat([x, cond], dim=-1)
        x = self.down_proj(x)
        x = self.dropout(self.act(x))
        x = self.linear(self.norm(x))
        x = self.dropout(self.act(x))
        return self.out_proj(x)


class ProjectionNetwork(nn.Module):
    def __init__(self, n_layers, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ProjectionFeedForwardBlock(d_model, d_ff, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, cond):
        for block in self.blocks:
            x = block(x, cond)
        return x


class ConditioningNetwork(nn.Module):
    def __init__(self, strategy, n_layers, d_model, d_ff, dropout=0.0):
        super().__init__()
        assert strategy in ["add", "concat", "project"]
        self.strategy = strategy
        if strategy == "project":
            self.projection_network = tag_module(
                ProjectionNetwork(n_layers, d_model, d_ff, dropout=dropout), "mapping"
            )
        self.extras = 1 if strategy == "concat" else 0

    def _add(self, x, cond):
        return x + cond

    def _concat(self, x, cond):
        return torch.cat([x, cond], dim=1)

    def _project(self, x, cond):
        return self.projection_network(x, cond)

    def forward(self, x, cond):
        if self.strategy == "add":
            return self._add(x, cond)
        elif self.strategy == "concat":
            return self._concat(x, cond)
        elif self.strategy == "project":
            return self._project(x, cond)
        else:
            raise ValueError("strategy must be one of: add, concat, project")


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
        attn_mask = repeat(
            attn_mask, "n l -> n h l s", h=self.n_heads, s=attn_mask.shape[-1]
        )
        x = scaled_dot_product_attention(q, k, v, attn_mask)
        x = rearrange(x, "n h l e -> n l (h e)")
        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip


class FeedForwardLayer(nn.Module):
    """From PyTorch Transformer implementation.
    https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/transformer.py#L563
    """
    def __init__(self, d_model, d_ff, dropout=0.0, bias=True):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model, bias=bias)
        self.activation = nn.GELU()

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
        d_head,
        n_mapping_layers,
        norm_first=False,
        cond_strategy="project",
        skip=False,
        dropout=0.0,
    ):
        super().__init__()
        self.self_attn = SelfAttentionBlock(d_model, d_head, dropout=dropout)
        self.ff = FeedForwardLayer(d_model, d_ff, dropout=dropout)
        self.skip_linear = nn.Linear(d_model * 2, d_model) if skip else None
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.cond_mapping = ConditioningNetwork(
            cond_strategy, n_mapping_layers, d_model, d_ff, dropout=dropout
        )

    def maybe_concat_padding(self, x, attn_mask, skip=None):
        # if concatenating conditioning lengthwise, also pad up the auxiliary inputs
        if self.cond_mapping.extras != 0:
            padding = torch.zeros_like(x[:, : self.cond_mapping.extras, :])
            attn_mask = torch.cat([attn_mask, padding], dim=1)
            if skip is not None:
                skip = torch.cat([skip, padding], dim=1)
        return x, attn_mask, skip

    def forward(self, x, attn_mask, cond, skip=None):
        x = checkpoint_helper(self.cond_mapping, x, cond)
        x, attn_mask, skip = self.maybe_concat_padding(x, attn_mask, skip)
        if self.skip_linear is not None:
            x = checkpoint_helper(self.skip_linear, torch.cat([x, skip], dim=-1))

        if self.norm_first:
            x = checkpoint_helper(self.self_attn, self.norm1(x), attn_mask, cond)
            x = checkpoint_helper(self.ff, self.norm2(x))
        else:
            x = checkpoint_helper(self.self_attn, x, attn_mask, cond)
            x = self.norm1(x)
            x = checkpoint_helper(self.ff, x)
            x = self.norm2(x)
        return x


class ProteinTransformerDenoiserModelV2(nn.Module):
    def __init__(
        self,
        n_attention_layers,
        n_mapping_layers,
        d_model,
        d_ff,
        d_head,
        skip_connect: bool = True,
        lm_embedder_type: str = "esmfold",
        cond_strategy: str = "project",
        input_dim: int = 1024,
        min_len: int = 32,
        num_classes: int = 0,
        dropout: float = 0.0,
    ):
        """Transformer denoiser model for proteins.
        input_dim: the dimension of the actual embedding needed for the frozen decoders (i.e. for ESMFold, this is 1024)
        d_model: the dimension of the latent embedding we will generate; might be a downprojection.
        """
        super().__init__()
        self.num_classes = num_classes  # not used
        self.input_dim = input_dim  # i.e. the actual latent dimension for ESMFold
        assert input_dim == ESMFOLD_S_DIM
        self.d_model = d_model
        self.min_len = min_len

        assert lm_embedder_type in ACCEPTED_LM_EMBEDDER_TYPES
        self.lm_embedder_type = lm_embedder_type
        if lm_embedder_type == "esmfold":
            self.repr_layer = None
        else:
            self.repr_layer = int(lm_embedder_type.split("_")[1][1:])

        self.esmfold_embedder, self.lm_alphabet = self._make_lm_embedder(
            lm_embedder_type
        )
        self.esmfold_embedder.eval()
        for param in self.esmfold_embedder.parameters():
            param.requires_grad = False

        self.time_emb = layers.FourierFeatures(1, d_model)
        self.time_in_proj = nn.Linear(d_model, d_model, bias=True)

        # project from the ESMFold latent to the actual dimension for latent diffusion
        self.latent_in_proj = nn.Linear(input_dim, d_model, bias=True)
        self.latent_out_proj = nn.Linear(d_model, input_dim, bias=True)

        if skip_connect:
            self.in_blocks = nn.ModuleList(
                [
                    TransformerBlock(
                        d_model,
                        d_ff,
                        d_head,
                        n_mapping_layers=n_mapping_layers,
                        skip=False,
                        cond_strategy=cond_strategy,
                        dropout=dropout,
                    )
                    for _ in range(n_attention_layers // 2)
                ]
            )
            self.mid_block = TransformerBlock(
                d_model,
                d_ff,
                d_head,
                n_mapping_layers=n_mapping_layers,
                skip=False,
                cond_strategy=cond_strategy,
                dropout=dropout,
            )
            self.out_blocks = nn.ModuleList(
                [
                    TransformerBlock(
                        d_model,
                        d_ff,
                        d_head,
                        n_mapping_layers=n_mapping_layers,
                        skip=skip_connect,
                        cond_strategy=cond_strategy,
                        dropout=dropout,
                    )
                    for _ in range(n_attention_layers // 2)
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
                    TransformerBlock(
                        d_model,
                        d_ff,
                        d_head,
                        n_mapping_layers=n_mapping_layers,
                        skip=False,
                        cond_strategy=cond_strategy,
                        dropout=dropout,
                    )
                    for _ in range(n_attention_layers)
                ]
            )
            xavier_init(self.blocks)
            self.in_blocks, self.mid_block, self.out_blocks = None, None, None

    def _make_lm_embedder(self, lm_embedder_type: str):
        if lm_embedder_type == "esmfold":
            embedder = ESMFold(make_trunk=False)
            embedder.eval()
            if not embedder.trunk is None:
                embedder.set_chunk_size(128)
            alphabet = None
        else:
            try:
                embedder, alphabet = torch.hub.load(
                    "facebookresearch/esm:main", lm_embedder_type
                )
            except:
                raise ValueError(
                    "Expected lm_embedder_type to be one of: ",
                    ACCEPTED_LM_EMBEDDER_TYPES,
                    " but got ",
                    lm_embedder_type,
                )
        return embedder, alphabet

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

    def embed_from_sequences(self, sequences: T.List[str], max_seq_len: int, min_seq_len: int):
        """
        Create the ESMFold intermediate representation from strings.
        Used for training only.
        """
        sequences = utils.get_random_sequence_crop_batch(sequences, max_seq_len, min_seq_len)
        with torch.no_grad():
            if self.lm_embedder_type == "esmfold":
                embeddings_dict = self.esmfold_embedder.infer_embedding(sequences)
                return embeddings_dict["s"], embeddings_dict["mask"]
            else:
                batch_converter = self.lm_alphabet.get_batch_converter()
                batch = [("", seq) for seq in sequences]
                _, _, tokens = batch_converter(batch)
                device = utils.get_model_device(self.esmfold_embedder)
                tokens = tokens.to(device)
                mask = tokens != self.lm_alphabet.padding_idx
                with torch.no_grad():
                    results = self.esmfold_embedder(
                        tokens, repr_layers=[self.repr_layer], return_contacts=False
                    )
                return results["representations"][self.repr_layer], mask

    def project_to_d_model(self, x: torch.Tensor):
        assert (
            x.shape[-1] == self.input_dim
        ), "x must have the same last dimension as input_dim"
        return self.latent_in_proj(x)

    def project_to_input_dim(self, x: torch.Tensor):
        assert (
            x.shape[-1] == self.d_model
        ), "x must have the same last dimension as d_model"
        return self.latent_out_proj(x)

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        mask: torch.Tensor,
    ):
        assert (
            x.shape[-1] == self.d_model
        ), "x must have the same dim as d_model; call self.project_to_latent(x) first."
        time_cond = self.time_in_proj(self.time_emb(maybe_unsqueeze(sigma)))

        # Transformer
        if self.in_blocks is None:
            for block in self.blocks:
                x = block(x, mask, time_cond)
            return x
        else:
            skips = []
            for i, block in enumerate(self.in_blocks):
                x = block(x, mask, time_cond)
                skips.append(x)
            x = self.mid_block(x, mask, time_cond)
            for i, block in enumerate(self.out_blocks):
                x = block(x, mask, time_cond, skip=skips.pop())
            return x


if __name__ == "__main__":
    import k_diffusion as K

    ds = K.datasets.ShardedTensorDataset(
        "/shared/amyxlu/data/cath/shards/seqlen_64/toy"
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=16)
    batch = next(iter(dl))
    model = ProteinTransformerDenoiserModelV2(
        n_attention_layers=7,
        n_mapping_layers=2,
        d_model=1024,
        d_ff=1536,
        d_head=64,
        skip_connect=True,
        cond_strategy="project"
    )
