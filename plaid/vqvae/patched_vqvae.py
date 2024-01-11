import einops
import math
import typing as T
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder
import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

# from .vqvae import VQVAE
from plaid.vqvae.encoder import Encoder
from plaid.vqvae.decoder import Decoder
from plaid.vqvae.quantizer import VectorQuantizer


def maybe_unsqueeze(x):
    if len(x.shape) == 1:
        return x.unsqueeze(-1)
    return x


def xavier_init(module):
    for p in module.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)
    return module


def shape_report(x, name, verbose=False):
    if verbose:
        print(f"{name} shape:", x.shape)
    return x


class TransformerVQVAE(L.LightningModule):
    def __init__(
        self,
        in_dim=1024,
        vqvae_h_dim=512,
        vqvae_res_h_dim=512,
        vqvae_n_res_layers=3,
        vqvae_n_embeddings=1024,
        vqvae_kernel=4,
        vqvae_stride=2,
        vqvae_embedding_dim=64,
        vqvae_beta=0.25,
        patch_len: int = 16,
        transformer_hidden_act="gelu",
        transformer_intermediate_size=3072,
        transformer_num_attention_heads=16,
        transformer_num_hidden_layers=12,
        transformer_position_embedding_type="absolute",
        lr=1e-4,
        lr_beta1=0.9,
        lr_beta2=0.999,
        lr_num_warmup_steps=0,
        lr_num_training_steps=10_000_000,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.bert_config = BertConfig(
            hidden_act=transformer_hidden_act,
            hidden_size=vqvae_embedding_dim,
            intermediate_size=transformer_intermediate_size,
            max_position_embeddings=1024,
            num_attention_heads=transformer_num_attention_heads,
            num_hidden_layers=transformer_num_hidden_layers,
            pad_token_id=0,
            position_embedding_type=transformer_position_embedding_type,
        )
        self.transformer = BertEncoder(self.bert_config)

        # encode image into continuous latent space
        self.vqvae_encoder = Encoder(
            in_dim=in_dim,
            h_dim=vqvae_h_dim,
            n_res_layers=vqvae_n_res_layers,
            res_h_dim=vqvae_res_h_dim,
            kernel=vqvae_kernel,
            stride=vqvae_stride,
        )
        self.pre_quantization_conv = nn.Conv1d(
            vqvae_h_dim, vqvae_embedding_dim, kernel_size=1, stride=1
        )
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            vqvae_n_embeddings, vqvae_embedding_dim, vqvae_beta
        )
        # decode the discrete latent representation
        self.vqvae_decoder = Decoder(
            e_dim=vqvae_embedding_dim,
            h_dim=vqvae_h_dim,
            out_dim=in_dim,
            n_res_layers=vqvae_n_res_layers,
            res_h_dim=vqvae_res_h_dim,
            kernel=vqvae_kernel,
            stride=vqvae_stride,
        )

        # optimizer
        self.lr = lr
        self.lr_beta1 = lr_beta1
        self.lr_beta2 = lr_beta2
        self.lr_num_warmup_steps = lr_num_warmup_steps
        self.lr_num_training_steps = lr_num_training_steps

        xavier_init(self.transformer)
        xavier_init(self.vqvae_encoder)
        xavier_init(self.pre_quantization_conv)
        xavier_init(self.vqvae_decoder)

    def transformer_forward(self, z_q):
        # z_q shape: (N, L', C')
        return self.transformer(z_q)["last_hidden_state"]  # (N, L', C')

    def stack_patches(self, x):
        N, L, C = x.shape
        n_chunks = math.ceil(L / self.patch_len)
        self.n_chunks = n_chunks
        chunks = x.chunk(self.n_chunks, dim=1)
        if L % self.patch_len != 0:
            chunks = chunks[:-1]
        return torch.cat(chunks, dim=0)  # (N * n_chunks, patch_len, C)

    def unstack_z_q(self, stacked_z_q):
        # stacked_z_q = (N * n_chunks, C', conv_patch)
        N = stacked_z_q.shape[0] // self.n_chunks
        chunked_z_q = stacked_z_q.chunk(
            N, dim=0
        )  # N element list of (n_chunks, C', conv_patch)

        def pivot_chunk(chunk):
            chunk = chunk.transpose(1, 2)  # (n_chunks, conv_patch, C')
            return chunk.reshape(-1, chunk.shape[-1])  # (n_chunks * conv_patch, C')

        chunked_z_q = [
            pivot_chunk(chunk) for chunk in chunked_z_q
        ]  # N element list of (n_chunks * conv_patch, C')
        z_q = torch.stack(chunked_z_q, dim=0)  # (N, n_chunks * conv_patch, C'
        return z_q

    def forward(self, x, verbose=False):
        # C': vqvae_embedding_dim
        stacked_chunks = self.stack_patches(x)
        stacked_chunks = einops.rearrange(stacked_chunks, "N L C -> N C L")
        stacked_z_e = self.vqvae_encoder(
            stacked_chunks
        )  # (N * n_chunks, vqvae_h_dim, conv_patch)
        stacked_z_e = self.pre_quantization_conv(stacked_z_e)
        embedding_loss, stacked_z_q, perplexity, _, _ = self.vector_quantization(
            stacked_z_e
        )

        z_q = self.unstack_z_q(stacked_z_q)
        z_q = self.transformer_forward(z_q)

        z_q = einops.rearrange(z_q, "N (n l) c -> N n l c", n=self.n_chunks)
        z_q = einops.rearrange(z_q, "N n l c -> (N n) c l")

        chunked_x_hat = self.vqvae_decoder(z_q)
        x_hat = einops.rearrange(
            chunked_x_hat, "(N n) C L -> N (n L) C", n=self.n_chunks
        )
        return embedding_loss, x_hat, perplexity

    def loss(self, x, verbose=False):
        embedding_loss, x_hat, perplexity = self.forward(x, verbose)
        recon_loss = torch.mean((x_hat - x) ** 2) / x.shape[1]
        loss = recon_loss + embedding_loss
        return loss, recon_loss, embedding_loss, perplexity

    def training_step(self, batch, batch_idx):
        x, _, _ = batch
        x = x[:, :12, :]  # tmp before patching
        x = x.permute(0, 2, 1)
        loss, recon_loss, embedding_loss, perplexity = self.loss(x)
        self.log("train_loss", loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_embedding_loss", embedding_loss)
        self.log("train_perplexity", perplexity)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, _ = batch
        x = x[:, :12, :]  # tmp before patching
        x = x.permute(0, 2, 1)
        loss, recon_loss, embedding_loss, perplexity = self.loss(x)
        self.log("val_loss", loss)
        self.log("val_recon_loss", recon_loss)
        self.log("val_embedding_loss", embedding_loss)
        self.log("val_perplexity", perplexity)
        return loss

    def test_step(self, batch, batch_idx):
        raise NotImplementedError


if __name__ == "__main__":
    model = TransformerVQVAE()

    # random data
    N, L, D_in = 128, 512, 1024
    x = np.random.random_sample((N, L, D_in))
    x = torch.tensor(x).float()

    # test vqvae
    # output = model(x, verbose=True)
    print(model.loss(x))
