import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def expand_to_shape(x, target_shape):
    # keep adding dimensions to the end until we match target dimensions
    while len(x.shape) < len(target_shape):
        x = x[..., None]
    return x.expand(target_shape)


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask=None, reduce="mean"):
    """Computes the mean squared error loss.
    assumes that the axis order is (B, L, ...)
    """
    if mask is None:
        return torch.mean((pred - target) ** 2)
    else:
        mask = expand_to_shape(mask, pred.shape)
        if reduce == "mean":
            return ((((pred - target) ** 2) * mask).sum()) / mask.sum()
        elif reduce == "batch":
            dims = tuple(range(1, len(pred.shape)))
            return ((((pred - target) ** 2) * mask).sum(dim=dims)) / mask.sum(dim=dims)
        else:
            raise ValueError(
                f"Unknown reduce type: {reduce}. Expected: 'mean' or 'batch'."
            )


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z, mask):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, length)
        mask.shape = (batch)

        quantization pipeline:

            1. get encoder input (B,C,L)
            2. flatten input to (B*L,C)

        The mask is False if all original x values for a patch were padded,
        in which case the sample would be ignored during loss and transformer computations.
        """
        # reshape z -> (batch, length, channel) and flatten
        device = z.device
        z = z.permute(0, 2, 1).contiguous()
        mask = expand_to_shape(mask, z.shape)
        z_flattened = z.view(-1, self.e_dim)
        mask = mask.view(-1)
        assert z_flattened.shape[0] == mask.shape[0]

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (
            torch.sum((z_flattened**2) * mask, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened * mask, self.embedding.weight.t())
        )

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)
        import pdb

        pdb.set_trace()

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        embedding_loss = masked_mse_loss(z_q.detach(), z, mask)
        commitment_loss = masked_mse_loss(z_q, z.detach(), mask)
        loss = embedding_loss + self.beta * commitment_loss
        # loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
        #     torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 2, 1).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices
