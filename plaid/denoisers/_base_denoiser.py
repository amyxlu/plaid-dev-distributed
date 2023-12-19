import typing as T
import torch
import torch.nn as nn
import abc
import einops

from .modules import FourierFeatures, RotaryEmbedding, LabelEmbedder
from .. import utils


@abc.ABC
class BaseDenoiser(nn.Module):
    def __init__(
        self,
        hid_dim,
        timestep_embedding_strategy: str = "fourier",
        pos_embedding_strategy: str = "rotary",
        use_self_conditioning: bool = False,
        label_num_classes: T.Optional[int] = None,
        cfg_dropout: float = 0.0,
        *args,
        **kwargs
    ):
        super().__init__()
        self.timestep_embedder = self.make_timestep_embedding(
            timestep_embedding_strategy, hid_dim
        )
        self.pos_embedder = self.make_positional_embedding(pos_embedding_strategy, hid_dim)

        self.self_conditioning_mlp = None
        self.label_embedder = None
        if use_self_conditioning:
            self.self_conditioning_mlp = self.make_projection_mlp(hid_dim * 2, hid_dim)
        if label_num_classes:
            self.label_embedder = self.make_label_embedding(
                num_classes=label_num_classes,
                hidden_size=hid_dim,
                dropout_prob=cfg_dropout,
            )
    
    @abc.abstractmethod
    def make_denoising_blocks(self, *args, **kwargs):
        raise NotImplementedError

    def make_projection_mlp(self, in_dim, hid_dim):
        return nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias=True),
            nn.SiLU(),
            nn.Linear(in_dim, hid_dim, bias=True),
        )

    def make_positional_embedding(self, strategy: str, hid_dim: int):
        assert strategy in ["rotary", "sinusoidal"]
        if strategy == "rotary":
            return RotaryEmbedding(dim=hid_dim)
        else:
            raise NotImplementedError

    def make_timestep_embedding(self, strategy: str, hid_dim: int):
        assert strategy in ["fourier", "learned_sinusoidal"]
        if strategy == "fourier":
            return FourierFeatures(in_features=1, out_features=hid_dim)
        else:
            raise NotImplementedError

    def make_label_embedding(
        self, num_classes: int, hid_dim: int, dropout_prob: float = 0.0
    ):
        return LabelEmbedder(
            num_classes=num_classes, hidden_size=hid_dim, dropout_prob=dropout_prob
        )

    def xavier_init_module(self, m):
        for p in m.parameters():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            else:
                if p.dim() > 1:
                    torch.nn.init.xavier_normal_(p)

    @abc.abstractmethod
    def forward(self, x):
        raise NotImplementedError
