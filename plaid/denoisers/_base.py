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
        timestep_embedding_strategy,
        conditioning_strategy,
        use_self_conditioning,
        hid_dim,
        *args,
        **kwargs
    ):
        super().__init__()
        if conditioning_strategy == "hidden_concat":
            self.conditioning_mlp = self.make_projection_mlp(hid_dim * 2, hid_dim)
        else:
            self.conditioning_mlp = None

        if conditioning_strategy == "length_concat":
            self.extras = 1

        if use_self_conditioning:
            self.self_conditioning_mlp = self.make_projection_mlp(hid_dim * 2, hid_dim)
        else:
            self.self_conditioning_mlp = None

        self.timestep_embedder = self.make_timestep_embedding(
            timestep_embedding_strategy, hid_dim
        )

    def pointwise_add_emb(self, ss0, emb):
        # ss0: (B, L, C)
        # emb: (B, C)
        emb = einops.repeat(emb, "b c -> b l c", l=ss0.shape[1])
        assert ss0.shape == emb.shape
        ss0 = ss0 + emb
        return ss0

    def length_concat_emb(self, ss0, emb):
        # ss0: (B, L, C)
        # emb: (B, C)
        emb = einops.rearrange(emb, "b c -> b 1 c")
        ss0 = torch.concat((ss0, emb), dim=1)
        return ss0

    def hidden_concat_emb(self, ss0, emb):
        # ss0: (B, L, C)
        # emb: (B, C)
        emb = einops.rearrange(emb, "b c -> b c 1")
        ss0 = torch.concat((ss0, emb), dim=-1)
        return ss0

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

    def conditioning(
        self,
        x,
        c,
        strategy: T.Optional[str] = None,
        proj_fn: T.Optional[T.Callable] = None,
    ):
        if strategy is None:
            return x, c

        else:
            assert strategy in ["length_concat", "add", "hidden_concat", "adaln_zero"]
            if strategy == "length_concat":
                x = self.length_concat_emb(x, c)  # (N, L + 1, C)
            elif strategy == "add":
                x = self.pointwise_add_emb(x, c)  # (N, L, C)
            elif strategy == "hidden_concat":
                assert not proj_fn is None
                x = self.hidden_concat_emb(x, c)  # (N, L, C * 2)
            else:
                raise NotImplementedError

            if not proj_fn is None:
                return proj_fn(x)
            else:
                return x

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
