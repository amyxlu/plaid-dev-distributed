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
from plaid.utils import get_lr_scheduler


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
        vqvae_h_dim=1024,
        vqvae_res_h_dim=1024,
        vqvae_n_res_layers=12,
        vqvae_n_embeddings=1024,
        vqvae_kernel=4,
        vqvae_stride=2,
        vqvae_embedding_dim=64,
        vqvae_beta=0.25,
        patch_len: int = 16,
        transformer_hidden_act="gelu",
        transformer_intermediate_size=2048,
        transformer_num_attention_heads=8,
        transformer_num_hidden_layers=6,
        transformer_position_embedding_type="absolute",
        lr=1e-4,
        lr_beta1=0.9,
        lr_beta2=0.999,
        lr_sched_type="constant",
        lr_num_warmup_steps=0,
        lr_num_training_steps=10_000_000,
        lr_num_cycles=1,
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
        self.lr_sched_type = lr_sched_type
        self.lr_num_warmup_steps = lr_num_warmup_steps
        self.lr_num_training_steps = lr_num_training_steps
        self.lr_num_cycles = lr_num_cycles

        xavier_init(self.transformer)
        xavier_init(self.vqvae_encoder)
        xavier_init(self.pre_quantization_conv)
        xavier_init(self.vqvae_decoder)
        self.save_hyperparameters()

    def transformer_forward(self, z_q, mask):
        # z_q shape: (N, L', C')
        # mask shape: (N, L')
        output = self.transformer(hidden_state=z_q, encoder_attention_mask=mask)
        return output["last_hidden_state"]  # (N, L', C')

    def stack_patches(self, x, mask):
        N, L, C = x.shape
        assert mask.shape == (N, L)
        self.n_chunks = math.ceil(L / self.patch_len)
        x, mask = x[:, :self.n_chunks * self.patch_len, :], mask[:, : self.n_chunks * self.patch_len]

        x_chunks = einops.rearrange(x, "N (L l) C -> (N L) l C", l=self.patch_len)
        mask_chunks = einops.rearrange(mask, "N (L l) -> (N L) l", l=self.patch_len)
        mask_chunks = mask_chunks.bool().any(dim=-1)

        # x_chunks = x.chunk(self.n_chunks, dim=1)
        # mask_chunks = mask.chunk(self.n_chunks, dim=1)
        # mask_chunks = [m.bool().any(dim=1) for m in mask_chunks]

        # if L % self.patch_len != 0:
        #     x_chunks = x_chunks[:-1]
        #     mask_chunks = mask_chunks[:-1]

        # x_chunks = torch.cat(x_chunks, dim=0)  # (n_chunks * N, patch_len, C)
        # mask_chunks = torch.cat(mask_chunks, dim=0).to(
        #     dtype=x_chunks.dtype
        # )  # (n_chunks * N)
        # import pdb;pdb.set_trace()
        return x_chunks, mask_chunks

    # def unstack_z_q(self, stacked_z_q, stacked_mask):
    #     # stacked_z_q = (N * n_chunks, C', conv_patch)
    #     # mask = (N * n_chunks)
    #     N = stacked_z_q.shape[0] // self.n_chunks
    #     z_q = einops.rearrange(stacked_z_q, "(N n) c l -> N (n l) c", n=self.n_chunks)
    #     mask = einops.rearrange(stacked_mask, "(N n) l -> N (n l)", n=self.n_chunks)
    #     import pdb;pdb.set_trace()
    #     # chunked_z_q = stacked_z_q.chunk(
    #     #     N, dim=0
    #     # )  # N element list of (n_chunks, C', conv_patch)
    #     # chunk_mask = mask.chunk(N, dim=0)  # N element list of (n_chunks)

    #     # def pivot_chunk(chunk):
    #     #     chunk = chunk.transpose(1, 2)  # (n_chunks, conv_patch, C')
    #     #     return chunk.reshape(-1, chunk.shape[-1])  # (n_chunks * conv_patch, C')

    #     # chunked_z_q = [
    #     #     pivot_chunk(chunk) for chunk in chunked_z_q
    #     # ]  # N element list of (n_chunks * conv_patch, C')
    #     # z_q = torch.stack(chunked_z_q, dim=0)  # (N, n_chunks * conv_patch, C')
    #     # chunk_mask = torch.stack(chunk_mask, dim=0)  # (N, n_chunks)
    #     return z_q, mask

    def forward(self, x, mask, verbose=False):
        N, L, C = x.shape
        if mask is None:
            mask = x.new_ones((N, L))

        # C': vqvae_embedding_dim
        stacked_chunks, stacked_mask = self.stack_patches(x, mask)
        stacked_chunks = einops.rearrange(stacked_chunks, "N L C -> N C L")
        stacked_z_e = self.vqvae_encoder(
            stacked_chunks
        )  # (N * n_chunks, vqvae_h_dim, conv_patch)
        stacked_z_e = self.pre_quantization_conv(stacked_z_e)
        (
            embedding_loss,
            stacked_z_q,
            perplexity,
            min_encodings,
            min_encoding_indices,
        ) = self.vector_quantization(stacked_z_e, stacked_mask)

        # unstack z_q and the masks
        z_q = einops.rearrange(stacked_z_q, "(N n) c l -> N (n l) c", n=self.n_chunks)
        mask = einops.rearrange(stacked_mask, "(N n) l -> N (n l)", n=self.n_chunks)
        z_q = self.transformer_forward(z_q, mask)

        z_q = einops.rearrange(z_q, "N (n l) c -> N n l c", n=self.n_chunks)
        z_q = einops.rearrange(z_q, "N n l c -> (N n) c l")

        chunked_x_hat = self.vqvae_decoder(z_q)
        x_hat = einops.rearrange(
            chunked_x_hat, "(N n) C L -> N (n L) C", n=self.n_chunks
        )
        return embedding_loss, x_hat, perplexity, min_encodings, min_encoding_indices

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(self.lr_beta1, self.lr_beta2),
        )
        scheduler = get_lr_scheduler(
            optimizer,
            self.lr_sched_type,
            num_warmup_steps=self.lr_num_warmup_steps,
            num_training_steps=self.lr_num_training_steps,
            num_cycles=self.lr_num_cycles,
        )

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def loss(self, x, mask, verbose=False):
        embedding_loss, x_hat, perplexity, _, _ = self.forward(x, mask, verbose)
        recon_loss = torch.mean((x_hat - x) ** 2) / x.shape[1]
        loss = recon_loss + embedding_loss
        return loss, recon_loss, embedding_loss, perplexity

    def training_step(self, batch, batch_idx):
        x, mask, sequence = batch
        loss, recon_loss, embedding_loss, perplexity = self.loss(x, mask)
        self.log("train_loss", loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_embedding_loss", embedding_loss)
        self.log("train_perplexity", perplexity)
        return loss

    def validation_step(self, batch, batch_idx):
        x, mask, _ = batch
        loss, recon_loss, embedding_loss, perplexity = self.loss(x, mask)
        self.log("val_loss", loss)
        self.log("val_recon_loss", recon_loss)
        self.log("val_embedding_loss", embedding_loss)
        self.log("val_perplexity", perplexity)
        return loss

    def test_step(self, batch, batch_idx):
        raise NotImplementedError


if __name__ == "__main__":
    from plaid.datasets import CATHShardedDataModule

    model = TransformerVQVAE()
    device = torch.device("cuda:4")

    datadir = "/shared/amyxlu/data/cath/shards/"
    pklfile = "/shared/amyxlu/data/cath/sequences.pkl"
    dm = CATHShardedDataModule(
        shard_dir=datadir,
        header_to_sequence_file=pklfile,
    )
    dm.setup("fit")
    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))
    x, mask, sequence = batch

    model.to(device)
    x, mask = x.to(device=device, dtype=torch.float32), mask.to(
        device, dtype=torch.float32
    )

    # test vqvae
    # output = model(x, verbose=True)
    # print(model.loss(x, mask))
    model.training_step((x, mask, None), 0)
