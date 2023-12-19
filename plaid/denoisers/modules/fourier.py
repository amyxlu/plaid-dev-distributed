import torch.nn as nn
import math
import torch


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer('weight', torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


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