import torch
from torch import nn, einsum
from einops import rearrange, repeat
import torch.nn.functional as F


xformers_installed = True
try:
    from xformers.ops import memory_efficient_attention
except ImportError:
    xformers_installed = False 


def exists(val):
    return val is not None


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        qkv_bias=False,
        dropout=0.0,
        use_xformers=False
    ):
        super().__init__()
        self.use_xformers = xformers_installed and use_xformers

        self.heads = heads
        dim_head = dim // heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def xformers_attention(self, x, mask=None):
        if not xformers_installed:
            raise ImportError("xformers is not installed, cannot use xformer attention")

        dtype, device = x.dtype, x.device
        b, l, _, h = *x.shape, self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # expects query/key/value tensors to have shape [B, L, H, D] 
        q, k, v = map(lambda t: rearrange(t, "b l (h d) -> b l h d", h=h), (q, k, v))

        # Create attn_bias from padding mask
        if mask is not None:
            attn_bias = torch.zeros_like(mask, dtype=dtype)
            attn_bias = attn_bias.masked_fill(~mask, float('-inf'))
            attn_bias = rearrange(attn_bias, "b l -> b () () l")  # Shape: (batch_size, 1, 1, seq_len)
            attn_bias = attn_bias.repeat(1, h, l, 1)
            attn_bias = attn_bias.to(device)
        else:
            attn_bias = None

        out = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        out = rearrange(out, "b l h d -> b l (h d)")
        return self.to_out(out)

    def attention(self, x, mask=None):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b l (h d) -> b h l d", h=h), (q, k, v))

        q = q * self.scale

        sim = einsum("b h i d, b h j d -> b h i j", q, k)
        mask_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = rearrange(mask, "b j -> b () () j")
            sim = sim.masked_fill(~mask, mask_value)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h l d -> b l (h d)", h=h)
        return self.to_out(out)

    def forward(self, x, mask=None):
        return self.xformers_attention(x, mask) if self.use_xformers else self.attention(x, mask)
