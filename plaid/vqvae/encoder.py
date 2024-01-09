import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from plaid.vqvae.residual import ResidualStack


class Encoder(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, kernel=4, stride=2):
        super(Encoder, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv1d(in_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
            nn.SiLU(),
            nn.Conv1d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1),
            nn.SiLU(),
            nn.Conv1d(
                h_dim, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1
            ),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
        )

    def forward(self, x):
        # outshape: (N, C, L)
        return self.conv_stack(x)


if __name__ == "__main__":
    # random data
    N, L, D_in = 8, 8, 1024
    D_hid = D_in // 2
    x = np.random.random_sample((N, D_in, L))
    x = torch.tensor(x).float()

    # test encoder
    encoder = Encoder(D_in, D_hid, 3, 64)
    encoder_out = encoder(x)
    print("Encoder out shape:", encoder_out.shape)
