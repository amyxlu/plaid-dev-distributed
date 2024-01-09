import torch
import torch.nn as nn
import numpy as np
from plaid.vqvae.encoder import Encoder
from plaid.vqvae.quantizer import VectorQuantizer
from plaid.vqvae.decoder import Decoder
from transformers.optimization import get_linear_schedule_with_warmup
import lightning as L


class VQVAE(L.LightningModule):
    def __init__(
        self, in_dim, h_dim, res_h_dim, n_res_layers, n_embeddings, embedding_dim, beta
    ):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(
            in_dim=in_dim, h_dim=h_dim, n_res_layers=n_res_layers, res_h_dim=res_h_dim
        )
        self.pre_quantization_conv = nn.Conv1d(
            h_dim, embedding_dim, kernel_size=1, stride=1
        )
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = Decoder(
            e_dim=embedding_dim,
            h_dim=h_dim,
            out_dim=in_dim,
            n_res_layers=n_res_layers,
            res_h_dim=res_h_dim,
        )

    def forward(self, x, verbose=False):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print("original data shape:", x.shape)
            print("encoded data shape:", z_e.shape)
            print("recon data shape:", x_hat.shape)
            assert False
        return embedding_loss, x_hat, perplexity

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self._lr,
            betas=(self._beta1, self._beta2),
            eps=self._eps,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self._num_warmup_steps,
            num_training_steps=self._num_training_steps,
        )

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def loss(self, x, verbose=False):
        embedding_loss, x_hat, perplexity = self.forward(x, verbose)
        recon_loss = torch.mean((x_hat - x) ** 2) / x.shape[1]
        loss = recon_loss + embedding_loss
        return loss, recon_loss, embedding_loss, perplexity

    def training_step(self, batch, batch_idx):
        x = batch
        loss, recon_loss, embedding_loss, perplexity = self.loss(x)
        self.log("train_loss", loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_embedding_loss", embedding_loss)
        self.log("train_perplexity", perplexity)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        loss, recon_loss, embedding_loss, perplexity = self.loss(x)
        self.log("val_loss", loss)
        self.log("val_recon_loss", recon_loss)
        self.log("val_embedding_loss", embedding_loss)
        self.log("val_perplexity", perplexity)
        return loss

    def test_step(self, batch, batch_idx):
        raise NotImplementedError


if __name__ == "__main__":
    # - n_e : number of embeddings
    # - e_dim : dimension of embedding
    # - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2

    vqvae = VQVAE(
        in_dim=1024,
        h_dim=1024 // 2,
        res_h_dim=64,
        n_res_layers=3,
        n_embeddings=512,
        embedding_dim=64,
        beta=0.25,
    )

    # random data
    N, L, D_in = 8, 8, 1024
    D_hid = 128 // 2
    x = np.random.random_sample((N, D_in, L))
    x = torch.tensor(x).float()

    # test vqvae
    # output = vqvae(x, verbose=True)
    output = vqvae(x)
    print(output)
    loss, recon_loss, embedding_loss, perplexity = vqvae.loss(x)
    print(loss)
    print(recon_loss)
    print(embedding_loss)
    print(perplexity)

