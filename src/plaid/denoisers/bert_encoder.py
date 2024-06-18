import einops
import torch
import typing as T
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder
import torch.nn as nn

from plaid.denoisers import BaseDenoiser


def maybe_unsqueeze(x):
    if len(x.shape) == 1:
        return x.unsqueeze(-1)
    return x


def xavier_init(module):
    for p in module.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)
    return module


def make_bert_encoder(
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
    bert_config = BertConfig(
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        hidden_act=hidden_act,
        hidden_dropout_prob=hidden_dropout_prob,
        hidden_size=hidden_size,
        initializer_range=initializer_range,
        intermediate_size=intermediate_size,
        layer_norm_eps=layer_norm_eps,
        max_position_embeddings=1024,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        pad_token_id=pad_token_id,
        position_embedding_type=position_embedding_type,
        use_cache=use_cache,
    )
    return BertEncoder(bert_config)


class ProteinBertDenoiser(BaseDenoiser):
    def __init__(
        self,
        # denoiser settings
        hid_dim,
        timestep_embedding_strategy: str = "fourier",
        pos_embedding_strategy: str = "rotary",
        use_self_conditioning: bool = False,
        label_num_classes: T.Optional[int] = None,
        cfg_dropout: float = 0.0,
        input_dim_if_different: T.Optional[int] = None,
        # BERT huggingface architecture
        intermediate_size=3072,
        num_attention_heads=16,
        num_hidden_layers=12,
    ):
        super().__init__(
            hid_dim=hid_dim,
            timestep_embedding_strategy=timestep_embedding_strategy,
            pos_embedding_strategy=pos_embedding_strategy,
            use_self_conditioning=use_self_conditioning,
            label_num_classes=label_num_classes,
            cfg_dropout=cfg_dropout,
            input_dim_if_different=input_dim_if_different,
        )
        self.bert_encoder = make_bert_encoder(
            hidden_size=hid_dim,
            intermediate_size=intermediate_size,
            max_position_embeddings=1024,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
        )
        self.conditioning_mlp = self.make_projection_mlp(hid_dim * 2, hid_dim)

    def make_projection_mlp(self, in_dim, hid_dim):
        return nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hid_dim, hid_dim, bias=True),
        )

    def hidden_concat(self, x, c):
        # x: (B, L, C)
        # c: (B, C)
        c = einops.repeat(c, "b c -> b l c", l=x.shape[1])
        x = torch.concat((x, c), dim=-1)
        return x

    def forward(self, x, t, mask=None, y=None, x_self_cond=None):
        # implicitly expands x dimension if necessary
        if self.input_projection is not None:
            x = self.input_projection(x)

        B, L, _ = x.shape

        if mask is None:
            mask = x.new_ones(B, L).long()

        if not x_self_cond is None:
            x = self.self_conditioning_mlp(torch.cat((x, x_self_cond), dim=-1))

        t = self.timestep_embedder(t)

        # TODO: this might have multiple labels?
        if not y is None:
            y = self.label_embedder(y)
            c = t + y
        else:
            c = t

        x = self.hidden_concat(x, c)
        x = self.conditioning_mlp(x)

        mask = einops.rearrange(mask, "b l -> b 1 1 l")  # add dimension to allow for broadcasting by heads
        output = self.encoder(
            hidden_states=x,
            attention_mask=mask,
        )
        denoised = output["last_hidden_state"][:, :-1, :]
        # assert denoised.shape[1] == self.max_seq_len, f"{denoised.shape}"
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
