import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualLayer(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """

    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(
                in_dim, res_h_dim, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.SiLU(),
            nn.Conv1d(res_h_dim, h_dim, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x


class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.silu = nn.SiLU()
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim)] * n_res_layers
        )

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = self.silu(x)
        return x


if __name__ == "__main__":
    # random data
    N, L, C = 3, 128, 1024
    x = np.random.random_sample((N, C, L))
    x = torch.tensor(x).float()
    # test Residual Layer
    res = ResidualLayer(C, C, C // 2)
    res_out = res(x)
    print("Res Layer out shape:", res_out.shape)

    # test res stack
    res_stack = ResidualStack(C, C, C // 2, 3)
    res_stack_out = res_stack(x)
    print("Res Stack out shape:", res_stack_out.shape)