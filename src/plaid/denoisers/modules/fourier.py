import torch.nn as nn
import math
import torch


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


