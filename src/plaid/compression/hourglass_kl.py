import lightning as L
import numpy as np
import torch.nn as nn
import torch
from einops import rearrange, repeat
from omegaconf import ListConfig

from .hourglass_lightning import (
    NaiveDownsample, 
    NaiveUpsample,
    LinearDownsample,
    LinearUpsample,
    PreNormResidual,
    PreNormLinearDownProjection,
    PreNormLinearUpProjection,
    Transformer,
    exists,
    pad_to_multiple
)


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


# def make_attn(in_channels, attn_type="vanilla"):
#     assert attn_type in ["vanilla", "linear", "none"], f"attn_type {attn_type} unknown"
#     print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
#     if attn_type == "vanilla":
#         return AttnBlock(in_channels)
#     elif attn_type == "none":
#         return nn.Identity(in_channels)
#     else:
#         return LinAttnBlock(in_channels)

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean


class HourglassEncoder(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth = (4, 4),
        shorten_factor = (4, 4),
        downproj_factor = (4, 4),
        attn_resampling = True,
        updown_sample_type = 'naive',
        heads = 8,
        dim_head = 64,
        causal = False,
        norm_out = False,
    ):
        super().__init__()
        upper_depth, lower_depth = depth
        upper_shorten_factor, lower_shorten_factor = shorten_factor
        upper_downproj_factor, lower_downproj_factor = downproj_factor
        transformer_kwargs = dict(
            heads = heads,
            dim_head = dim_head
        )

        self.causal = causal

        if updown_sample_type == 'naive':
            self.upper_downsample = NaiveDownsample(upper_shorten_factor)
            self.lower_downsample = NaiveDownsample(lower_shorten_factor)
        elif updown_sample_type == 'linear':
            self.downsample = LinearDownsample(dim, upper_shorten_factor)
        else:
            raise ValueError(f'unknown updown_sample_type keyword value - must be either naive or linear for now')

        self.upper_down_projection = PreNormLinearDownProjection(dim, downproj_factor)
        self.upper_attn_resampling_context_downproj = PreNormLinearDownProjection(dim, downproj_factor) if attn_resampling else None
        self.upper_attn_resampling_pre_valley = Transformer(dim = dim // downproj_factor, depth = 1, **transformer_kwargs) if attn_resampling else None

        self.lower_down_projection = PreNormLinearDownProjection(dim // downproj_factor, dim // (downproj_factor * 2))
        self.lower_attn_resampling_context_downproj = PreNormLinearDownProjection(dim, downproj_factor) if attn_resampling else None
        self.lower_attn_resampling_pre_valley = Transformer(dim = dim // downproj_factor, depth = 1, **transformer_kwargs) if attn_resampling else None

        self.down_projection = PreNormLinearDownProjection(dim, downproj_factor)
        self.attn_resampling_context_downproj = PreNormLinearDownProjection(dim, downproj_factor) if attn_resampling else None
        self.attn_resampling_pre_valley = Transformer(dim = dim // downproj_factor, depth = 1, **transformer_kwargs) if attn_resampling else None

        self.pre_transformer = Transformer(dim = dim, causal = causal, **transformer_kwargs)
        self.norm_out = nn.LayerNorm(dim) if norm_out else nn.Identity()

    def forward(self, x, mask = None):
        # b : batch, n : sequence length, d : feature dimension, s : shortening factor
        s, b, n = self.shorten_factor, *x.shape[:2]

        # top half of hourglass, pre-transformer layers
        x = self.pre_transformer(x, mask = mask)

        # pad to multiple of shortening factor, in preparation for pooling
        x = pad_to_multiple(x, s, dim = -2)

        if exists(mask):
            padded_mask = pad_to_multiple(mask, s, dim = -1, value = False)

        # save the residual, and for "attention resampling" at downsample and upsample
        x_residual = x.clone()

        # if autoregressive, do the shift by shortening factor minus one
        if self.causal:
            shift = s - 1
            x = F.pad(x, (0, 0, shift, -shift), value = 0.)

            if exists(mask):
                padded_mask = F.pad(padded_mask, (shift, -shift), value = False)

        # naive average pool
        downsampled = self.downsample(x)
        if exists(mask):
            downsampled_mask = reduce(padded_mask, 'b (n s) -> b n', 'sum', s = s) > 0
        else:
            downsampled_mask = None

        # also possibly reduce along dim=-1
        downsampled = self.down_projection(downsampled)

        # pre-valley "attention resampling" - they have the pooled token in each bucket attend to the tokens pre-pooled
        if exists(self.attn_resampling_pre_valley):
            if exists(mask):
                attn_resampling_mask = rearrange(padded_mask, 'b (n s) -> (b n) s', s = s)
            else:
                attn_resampling_mask = None
            downsampled = self.attn_resampling_pre_valley(
                rearrange(downsampled, 'b n d -> (b n) () d'),
                rearrange(self.attn_resampling_context_downproj(x), 'b (n s) d -> (b n) s d', s = s),
                mask = attn_resampling_mask
            )

            downsampled = rearrange(downsampled, '(b n) () d -> b n d', b = b)
        
        return downsampled