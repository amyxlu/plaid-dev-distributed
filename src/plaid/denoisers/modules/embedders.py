import torch.nn as nn
from einops import rearrange
import math
import torch
import numpy as np


#################################################################################
# Fourier 
#################################################################################


class GaussianFourierProjection(nn.Module):
    """
    https://arxiv.org/abs/2006.10739
    https://github.com/tancik/fourier-feature-networks/blob/master/Demo.ipynb
    """

    def __init__(self, embed_dim: int, scale: float = 2 * torch.pi):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        w = torch.randn(embed_dim // 2) * scale
        assert w.requires_grad == False
        self.register_buffer("W", w)

    def forward(self, t: torch.Tensor):
        # t: (batch_size,)
        # w: (embed_dim // 2,)
        t = t.to(self.W.dtype)
        t_proj = 2.0 * torch.pi * t[:, None] @ self.W[None, :]
        embed = torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)
        return embed
    

class Base2FourierFeatures(nn.Module):
    # jax to torch adaptation of VDM code
    # https://github.com/google-research/vdm/blob/main/model_vdm.py#L618
    def __init__(self, start=4, stop=8, step=1):
        self.start = start
        self.stop = stop
        self.step = step

    def forward(self, inputs):
        freqs = torch.arange(self.start, self.stop, self.step).to(dtype=inputs.dtype)

        # Create Base 2 Fourier features
        w = 2. ** freqs * 2 * math.pi
        w = torch.tile(w[None, :], (1, inputs.shape[-1]))

        # Compute features
        h = torch.repeat(inputs, len(freqs), axis=-1)
        h = w * h
        h = torch.concatenate([torch.sin(h), torch.cos(h)], axis=-1)
        return h


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer('weight', torch.randn([out_features // 2, in_features]) * std)

    def forward(self, sigma):
        if sigma.ndim < 2:
            sigma = sigma.unsqueeze(1)

        f = 2 * math.pi * sigma @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


#################################################################################
# Sinusoidal
#################################################################################


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        """fixed trig projection"""
        # https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/guided_diffusion.py#L115
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


#################################################################################

# From DIT:

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def get_1d_sincos_pos_embed(embed_dim, pos):
    """
    Get fixed embedding for 1d sincons
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
