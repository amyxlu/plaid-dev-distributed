import torch
from torch import nn, einsum
from einops import rearrange
import torch.nn.functional as F


def exists(val):
    return val is not None


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        qkv_bias=False,
        dropout=0.0,
    ):
        super().__init__()
        self.heads = heads
        dim_head = dim // heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        h, device = self.heads, x.device
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

