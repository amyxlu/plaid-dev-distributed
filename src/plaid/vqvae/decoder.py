import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from plaid.vqvae.residual import ResidualStack


class Decoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, e_dim, h_dim, out_dim, n_res_layers, res_h_dim, kernel=4, stride=2):
        super(Decoder, self).__init__()

        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose1d(e_dim, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose1d(h_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
            nn.SiLU(),
            nn.ConvTranspose1d(h_dim // 2, out_dim, kernel_size=kernel, stride=stride, padding=1),
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)


if __name__ == "__main__":
    # random data
    N, L, D_in = 8, 8, 1024
    D_hid = 128 // 2
    x = np.random.random_sample((N, D_in, L))
    x = torch.tensor(x).float()

    # test decoder
    decoder = Decoder(D_in, D_hid, 3, 64)
    decoder_out = decoder(x)
    print("Decoder out shape:", decoder_out.shape)
