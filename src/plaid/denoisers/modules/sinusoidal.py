import torch.nn as nn
from einops import rearrange
import torch
import math

    # def timestep_embedding(t, dim, max_period=10000):
    #     """
    #     Create sinusoidal timestep embeddings.
    #     :param t: a 1-D Tensor of N indices, one per batch element.
    #                       These may be fractional.
    #     :param dim: the dimension of the output.
    #     :param max_period: controls the minimum frequency of the embeddings.
    #     :return: an (N, D) Tensor of positional embeddings.
    #     """
    #     # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    #     half = dim // 2
    #     freqs = torch.exp(
    #         -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    #     ).to(device=t.device)
    #     args = t[:, None].float() * freqs[None]
    #     embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    #     if dim % 2:
    #         embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    #     return embedding


# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/guided_diffusion.py#L186

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered