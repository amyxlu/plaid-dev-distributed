# from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import einops



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv1d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool1d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv1d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool1d(stride)),
                ("0", nn.Conv1d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm1d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        print(out.shape)
        out = self.relu2(self.bn2(self.conv2(out)))
        print(out.shape)
        out = self.avgpool(out)
        print(out.shape)
        out = self.bn3(self.conv3(out))
        print(out.shape)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        print(out.shape)
        out = self.relu3(out)
        return out


class AttentionPool1d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        # add additional token to attention pool over
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        
    def forward(self, x):
        x = einops.rearrange(x, "n c l -> l n c")
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (L+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (L+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)

from typing import List

class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, input_dim: int, layers: List[int], widths: List[int], output_dim: int, heads: int, input_seq_len: int): #, attn_pool_final_layer: bool = False):
        # width = internal dimension
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_seq_len = input_seq_len
        assert len(widths) == len(layers)

        # the 3-layer stem
        self.conv1 = nn.Conv1d(input_dim, input_dim // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(input_dim // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(input_dim // 2, input_dim // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(input_dim // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv1d(input_dim // 2, widths[0], kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(widths[0])
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool1d(2)

        # residual layers
        self._inplanes = widths[0]  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(widths[0], layers[0])
        self.layer2 = self._make_layer(widths[1], layers[1], stride=2)
        self.layer3 = self._make_layer(widths[2], layers[2], stride=2)
        self.layer4 = self._make_layer(widths[3], layers[3], stride=2)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = einops.rearrange(x, "n l c -> n c l")
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = einops.rearrange(x, "n c l -> n l c")
        # removes attention pool layers since need more than just the embedding
        return x

# Union[Tuple[int, int, int, int], int],
layers = (3, 4, 6, 3)
widths = (512, 256, 128, 64)
model = ModifiedResNet(input_dim=1024, layers=layers, widths=widths, output_dim=128, heads=8, input_seq_len=seq_len)
model = model.to(device)

inp = x * mask.unsqueeze(-1).expand_as(x)
out = model(inp)
print(out.shape)
