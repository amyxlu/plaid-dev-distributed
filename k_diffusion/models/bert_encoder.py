from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import einops
import typing as T
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder
import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from .. import utils, layers
import torch.nn as nn


def maybe_unsqueeze(x):
    if len(x.shape) == 1:
        return x.unsqueeze(-1)
    return x


def xavier_init(module):
    for p in module.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)
    return module


class ProteinBertDenoiser(nn.Module):
    def __init__(
        self,
        max_seq_len=256,
        min_len=8,
        attention_probs_dropout_prob=0.1,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        hidden_size=1024,
        initializer_range=0.02,
        intermediate_size=3072,
        layer_norm_eps=1e-12,
        num_attention_heads=16,
        num_hidden_layers=12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.min_len = min_len
        self.bert_config = BertConfig(
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_size=hidden_size,
            initializer_range=initializer_range,
            intermediate_size=intermediate_size,
            layer_norm_eps=layer_norm_eps,
            max_position_embeddings=max_seq_len,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            pad_token_id=pad_token_id,
            position_embedding_type=position_embedding_type,
            use_cache=use_cache,
        )
        self.encoder = BertEncoder(self.bert_config)
        self.time_emb = layers.FourierFeatures(1, hidden_size)
        xavier_init(self.encoder)
    
    def concat_time_embedding(self, x_noised, sigma, mask):
        time_emb = einops.rearrange(self.time_emb(maybe_unsqueeze(sigma)), "b c -> b 1 c")
        x = torch.cat([x_noised, time_emb], dim=1)
        mask = torch.cat([mask, torch.zeros((mask.shape[0], 1), dtype=mask.dtype, device=mask.device)], dim=1)
        return x, mask

    def embed_from_sequences(self, sequences: T.List[str]):
        sequences = utils.get_random_sequence_crop_batch(
            sequences, self.max_seq_len, self.min_len
        )
        with torch.no_grad():
            embeddings_dict = self.esmfold_embedder.infer_embedding(sequences)
            return embeddings_dict["s"], embeddings_dict["mask"]
    
    def forward(
        self,
        x_noised: torch.Tensor,
        sigma: torch.Tensor,
        mask: torch.Tensor,
    ):
        assert (
            x_noised.shape[-1] == self.bert_config.hidden_size
        ), "x must have the same dim as d_model."
        x, mask = self.concat_time_embedding(x_noised, sigma, mask)  # (B, L+1, C), (B, L+1)
        mask = einops.rearrange(mask, "b l -> b 1 1 l")  # add dimension to allow for broadcasting by heads
        output = self.encoder(
            hidden_states=x,
            attention_mask=mask,
        )
        denoised = output['last_hidden_state'][:, :-1, :]
        assert denoised.shape[1] == self.max_seq_len
        return denoised


        # TODO: make U-connections in Bert encoder
        # if self.in_blocks is None:
        #     for block in self.blocks:
        #         x = block(x, mask, cond)
        #     return x
        # else:
        #     skips = []
        #     for i, block in enumerate(self.in_blocks):
        #         x = block(x, mask, cond)
        #         skips.append(x)
        #     x = self.mid_block(x, mask, cond)
        #     for i, block in enumerate(self.out_blocks):
        #         x = block(x, mask, cond, skip=skips.pop())
        #     return x
