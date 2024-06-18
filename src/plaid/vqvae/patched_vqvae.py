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
from plaid.utils import get_lr_scheduler, LatentScaler
from plaid.losses.modules import SequenceAuxiliaryLoss, BackboneAuxiliaryLoss
from plaid.proteins import LatentToSequence, LatentToStructure
from plaid.esmfold.misc import batch_encode_sequences


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
        latent_scaler=None,
        # vqvae specifications
        vqvae_h_dim=1024,
        vqvae_res_h_dim=1024,
        vqvae_n_res_layers=12,
        vqvae_n_embeddings=1024,
        vqvae_kernel=4,
        vqvae_stride=2,
        vqvae_embedding_dim=64,
        vqvae_beta=0.25,
        # autoregressive-over-patches specifications
        patch_len: int = 16,
        transformer_hidden_act="gelu",
        transformer_intermediate_size=2048,
        transformer_num_attention_heads=8,
        transformer_num_hidden_layers=6,
        transformer_position_embedding_type="absolute",
        # optimization
        lr=1e-4,
        lr_beta1=0.9,
        lr_beta2=0.999,
        lr_sched_type="constant",
        lr_num_warmup_steps=0,
        lr_num_training_steps=10_000_000,
        lr_num_cycles=1,
        # auxiliary losses
        sequence_constructor=None,
        structure_constructor=None,
        sequence_decoder_weight=0.0,
        structure_decoder_weight=0.0,
        latent_reconstruction_method="unnormalized_x_recons",
        log_reconstructed_sequences=False,
    ):
        super().__init__()

        self.normalize_latent_input = latent_scaler is not None
        self.latent_scaler = latent_scaler
        assert latent_reconstruction_method in {"x_recons", "unnormalized_x_recons"}
        if latent_reconstruction_method == "unnormalized_x_recons":
            assert not latent_scaler is None
        self.latent_recons_method = latent_reconstruction_method

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
        self.pre_quantization_conv = nn.Conv1d(vqvae_h_dim, vqvae_embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(vqvae_n_embeddings, vqvae_embedding_dim, vqvae_beta)
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

        # auxiliary losses
        self.need_to_setup_sequence_decoder = sequence_decoder_weight > 0.0
        self.need_to_setup_structure_decoder = structure_decoder_weight > 0.0

        self.sequence_constructor = sequence_constructor
        self.structure_constructor = structure_constructor
        self.sequence_decoder_weight = sequence_decoder_weight
        self.structure_decoder_weight = structure_decoder_weight
        self.sequence_loss_fn = None
        self.structure_loss_fn = None

        self.log_reconstructed_sequences = log_reconstructed_sequences

        self.save_hyperparameters()

    def transformer_forward(self, z_q):
        # z_q shape: (N, L', C')
        output = self.transformer(hidden_states=z_q)
        return output["last_hidden_state"]  # (N, L', C')

    def stack_patches(self, x):
        N, L, C = x.shape
        self.n_chunks = math.ceil(L / self.patch_len)
        x = x[:, : self.n_chunks * self.patch_len, :]
        x_chunks = einops.rearrange(x, "N (L l) C -> (N L) l C", l=self.patch_len)
        return x_chunks

    def unpack_batch(self, batch):
        # 2024/02/08: For CATHShardedDataModule HDF5 loaders
        if isinstance(batch[-1], dict):
            # dictionary of structure features
            embs, sequences, gt_structures = batch
            assert "backbone_rigid_tensor" in batch[-1].keys()
            assert max([len(s) for s in sequences]) <= embs.shape[1]
            return embs, sequences, gt_structures
        elif isinstance(batch[-1][0], str):
            embs, sequences, _ = batch
            return embs, sequences, None
        else:
            raise Exception(
                f"Batch tuple not understood. Data type of last element of batch tuple is {type(batch[-1])}."
            )

    def forward(self, batch):
        # sequences *must* be already trimmed
        x, sequences, gt_structures = self.unpack_batch(batch)
        true_aatype, _, _, _, _ = batch_encode_sequences(sequences)
        device = x.device
        true_aatype = true_aatype.to(device)

        if self.normalize_latent_input:
            x = self.latent_scaler.scale(x)
        N, L, C = x.shape

        # C': vqvae_embedding_dim
        stacked_chunks = self.stack_patches(x)
        stacked_chunks = einops.rearrange(stacked_chunks, "N L C -> N C L")
        stacked_z_e = self.vqvae_encoder(stacked_chunks)  # (N * n_chunks, vqvae_h_dim, conv_patch)
        stacked_z_e = self.pre_quantization_conv(stacked_z_e)
        (
            embedding_loss,
            stacked_z_q,
            perplexity,
            min_encodings,
            min_encoding_indices,
        ) = self.vector_quantization(stacked_z_e)

        # unstack z_q
        z_q = einops.rearrange(stacked_z_q, "(N n) c l -> N (n l) c", n=self.n_chunks)
        z_q = self.transformer_forward(z_q)

        z_q = einops.rearrange(z_q, "N (n l) c -> N n l c", n=self.n_chunks)
        z_q = einops.rearrange(z_q, "N n l c -> (N n) c l")

        chunked_x_hat = self.vqvae_decoder(z_q)
        x_hat = einops.rearrange(chunked_x_hat, "(N n) C L -> N (n L) C", n=self.n_chunks)
        recons_loss = torch.mean((x_hat - x) ** 2) / x.shape[1]
        vqvae_loss = recons_loss + embedding_loss

        """
        Auxiliary losses: constrain sequence and structure
        """
        if self.latent_recons_method == "unnormalized_x_recons":
            x_hat = self.latent_scaler.unscale(x_hat)

        # TODO: anneal losses
        if self.sequence_decoder_weight > 0.0:
            seq_loss, seq_loss_dict = self.sequence_loss(x_hat, sequences, cur_weight=None)
        else:
            seq_loss = 0.0

        if self.structure_decoder_weight > 0.0:
            assert (
                not gt_structures is None
            ), "If using structure as an auxiliary loss, ground truth structures must be provided"
            struct_loss, struct_loss_dict = self.structure_loss(
                x_hat, gt_structures, sequences, cur_weight=None
            )
        else:
            struct_loss = 0.0
            struct_loss_dict = None

        loss = vqvae_loss + seq_loss + struct_loss

        log_dict = {}
        log_dict = seq_loss_dict | log_dict if not seq_loss_dict is None else log_dict
        log_dict = struct_loss_dict | log_dict if not struct_loss_dict is None else log_dict

        log_dict["loss"] = loss.item()
        log_dict["vqvae_loss"] = vqvae_loss.item()
        log_dict["embedding_loss"] = embedding_loss.item()
        log_dict["vq_perplexity"] = perplexity.item()
        # todo: also log min_encoding_indices

        return loss, log_dict, min_encodings, min_encoding_indices

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

    def setup_sequence_decoder(self):
        """If a reference pointer to the auxiliary sequence decoder wasn't already passed in
        at the construction of the class, load the sequence decoder onto the GPU now.
        """
        assert self.need_to_setup_sequence_decoder
        if self.sequence_constructor is None:
            self.sequence_constructor = LatentToSequence()

        self.sequence_loss_fn = SequenceAuxiliaryLoss(
            self.sequence_constructor, weight=self.sequence_decoder_weight
        )
        self.need_to_setup_sequence_decoder = False

    def setup_structure_decoder(self):
        assert self.need_to_setup_structure_decoder
        if self.structure_constructor is None:
            # Note: this will make ESMFold in the function and might be expensive
            self.structure_constructor = LatentToStructure()

        self.structure_loss_fn = BackboneAuxiliaryLoss(
            self.structure_constructor, weight=self.structure_decoder_weight
        )
        self.need_to_setup_structure_decoder = False

    def sequence_loss(self, latent, sequence, cur_weight=None):
        if self.need_to_setup_sequence_decoder:
            self.setup_sequence_decoder()
        # sequence should be the one generated when saving the latents,
        # i.e. lengths are already trimmed to self.max_seq_len
        # if cur_weight is None, no annealing is done except for the weighting
        # specified when specifying the class. The mask is implicit.
        return self.sequence_loss_fn(
            latent, sequence, cur_weight, log_recons_strs=self.log_reconstructed_sequences
        )

    def structure_loss(self, latent, gt_structures, sequences, cur_weight=None):
        if self.need_to_setup_structure_decoder:
            self.setup_structure_decoder()
        return self.structure_loss_fn(latent, gt_structures, sequences, cur_weight)

    def training_step(self, batch, batch_idx):
        loss, log_dict, min_encodings, min_encoding_indices = self(batch)
        self.log_dict(log_dict)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, log_dict, min_encodings, min_encoding_indices = self(batch)
        self.log_dict(log_dict)
        return loss

    def test_step(self, batch, batch_idx):
        raise NotImplementedError


if __name__ == "__main__":
    from plaid.datasets import CATHShardedDataModule

    model = TransformerVQVAE()
    device = torch.device("cuda")

    datadir = "/homefs/home/lux70/storage/data/cath/shards"
    dm = CATHShardedDataModule(
        storage_type="hdf5", shard_dir=datadir, batch_size=32, seq_len=64, dtype="fp32"
    )
    dm.setup("fit")
    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))

    model.to(device)
    x, sequence, _ = batch

    # test vqvae
    # output = model(x, verbose=True)
    model.training_step((x.to(device), sequence, "sdf"), 0)
