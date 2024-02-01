"""
U-TriangularSelfAttention (U-TSA) denoiser
"""

import typing as T
import einops
from pathlib import Path
import os
import abc

import numpy as np
import torch
from torch import nn
from torch import Tensor

from plaid.esmfold import (
    RelativePosition,
    FoldingTrunkConfig,
)
from plaid.denoisers import BaseDenoiser
from plaid.constants import c_s, c_z, structure_module_c_s, structure_module_c_z
from plaid.denoisers.modules import TriangularSelfAttentionBlock
from plaid.esmfold.misc import get_esmfold_model_state


PathLike = T.Union[str, Path]
ArrayLike = T.Union[np.ndarray, torch.Tensor]


class BaseTriSelfAttnDenoiser(BaseDenoiser):
    def __init__(
        self,
        hid_dim: int,
        conditioning_strategy: T.Optional[str] = "hidden_concat",
        timestep_embedding_strategy: str = "fourier",
        pos_embedding_strategy: str = "rotary",
        use_self_conditioning: bool = False,
        label_num_classes: T.Optional[int] = None,
        cfg_dropout: float = 0.0,
    ):
        super().__init__(
            hid_dim=hid_dim,
            timestep_embedding_strategy=timestep_embedding_strategy,
            pos_embedding_strategy=pos_embedding_strategy,
            use_self_conditioning=use_self_conditioning,
            label_num_classes=label_num_classes,
            cfg_dropout=cfg_dropout,
        )

        # fixed dimensions
        trunk_cfg: FoldingTrunkConfig = FoldingTrunkConfig()
        self.trunk_cfg = trunk_cfg
        self.chunk_size = trunk_cfg.chunk_size
        self.pairwise_positional_embedding = RelativePosition(
            trunk_cfg.position_bins, c_z
        )

        self.conditioning_strategy = conditioning_strategy
        self.make_blocks()

    @abc.abstractmethod
    def make_blocks(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def _iteration(self, *args, **kwargs):
        raise NotImplementedError

    def set_chunk_size(self, chunk_size):
        # This parameter means the axial attention will be computed
        # in a chunked manner. This should make the memory used more or less O(L) instead of O(L^2).
        # It's equivalent to running a for loop over chunks of the dimension we're iterative over,
        # where the chunk_size is the size of the chunks, so 128 would mean to parse 128-lengthed chunks.
        self.chunk_size = chunk_size

    def forward(self, x, t, mask=None, z=None, y=None, x_self_cond=None):
        """
        x: (B, L, C) sequence representation
        z: (B, L, L, C) pairwise representation
        t: (B, L) time embedding
        y: (B,) label
        mask: (B, L) mask
        """
        # self conditioning should be either zeros or x_prev, determined in outer loop
        if mask is None:
            mask = x.new_ones(x.shape[:2]).long()

        if not x_self_cond is None:
            x = self.self_conditioning_mlp(torch.cat((x, x_self_cond), dim=-1))

        B, L, _ = x.shape
        if z is None:
            z = x.new_zeros(B, L, L, c_z)

        t = self.timestep_embedder(t)

        # TODO: this might have multiple labels?
        if not y is None:
            y = self.label_embedder(y)
            c = t + y
        else:
            c = t

        residx = einops.repeat(
            torch.arange(L, device=x.device, dtype=int), "L -> B L", B=B
        )
        z = z + self.pairwise_positional_embedding(residx, mask=mask)

        # TODO: multiple iterations?
        x, z = self._iteration(x, z, c, mask)

        if self.conditioning_strategy == "length_concat":
            x = x[:, :-1, :]
            z = z[:, :-1, :-1, :]

        return x


class PreinitializedTriSelfAttnDenoiser(BaseTriSelfAttnDenoiser):
    def __init__(
        self,
        hid_dim: int,
        conditioning_strategy: T.Optional[str] = "hidden_concat",
        timestep_embedding_strategy: str = "fourier",
        pos_embedding_strategy: str = "rotary",
        use_self_conditioning: bool = False,
        label_num_classes: T.Optional[int] = None,
        cfg_dropout: float = 0.0,
    ):
        super().__init__(
            hid_dim=hid_dim,
            conditioning_strategy=conditioning_strategy,
            timestep_embedding_strategy=timestep_embedding_strategy,
            pos_embedding_strategy=pos_embedding_strategy,
            use_self_conditioning=use_self_conditioning,
            label_num_classes=label_num_classes,
            cfg_dropout=cfg_dropout,
        )
        assert hid_dim == c_s

    def _filter_state_dict(self, state_dict):
        orig_keys = list(state_dict.keys())
        for k in orig_keys:
            if k.startswith("trunk."):
                state_dict[k.replace("trunk.", "")] = state_dict[k]
            else:
                pass
            state_dict.pop(k)

    def load_pretrained_weights(self):
        _, model_state = get_esmfold_model_state()
        self._filter_state_dict(model_state)

        unmatched_keys = self.load_state_dict(model_state, strict=False)
        missing_keys = unmatched_keys.missing_keys
        # unexpected_keys = unmatched_keys.unexpected_keys
        missing_keys = list(filter(lambda x: not "conditioning_mlp" in x, missing_keys))
        print(
            f"Loaded pretrained weights for {len(model_state) - len(unmatched_keys)} keys."
        )
        print(f"Missing keys: block conditioning_mlp weights, {','.join(missing_keys)}")

    def make_blocks(self):
        # NOTE: this imports our modified denoiser block, but is named the same as the original
        # in order to load in weights more easily and also have the same size.
        block = TriangularSelfAttentionBlock
        self.blocks = nn.ModuleList(
            [
                block(
                    sequence_state_dim=c_s,
                    pairwise_state_dim=c_z,
                    sequence_head_width=self.trunk_cfg.sequence_head_width,
                    pairwise_head_width=self.trunk_cfg.pairwise_head_width,
                    conditioning_strategy=self.conditioning_strategy,
                    dropout=self.trunk_cfg.dropout,
                    skip=False,
                )
                for i in range(self.trunk_cfg.num_blocks)
            ]
        )
        self.load_pretrained_weights()

    def _iteration(self, x, z, c, mask, *args, **kwargs):
        for block in self.blocks:
            # this looks similar to https://github.com/facebookresearch/esm/blob/main/esm/esmfold/v1/trunk.py#L183
            # except that our modified block also takes in the conditioning, c.
            x, z = block(
                x,
                z,
                c,
                mask=mask,
                chunk_size=self.chunk_size,
                conditioning_strategy=self.conditioning_strategy,
            )
        return x, z


class UTriSelfAttnDenoiser(BaseTriSelfAttnDenoiser):
    def __init__(
        self,
        hid_dim: int,
        num_blocks: int,
        conditioning_strategy: T.Optional[str] = "hidden_concat",
        timestep_embedding_strategy: str = "fourier",
        pos_embedding_strategy: str = "rotary",
        use_self_conditioning: bool = False,
        use_skip_connections: bool = True,
        label_num_classes: T.Optional[int] = None,
        cfg_dropout: float = 0.0,
    ):
        self.num_blocks = num_blocks
        self.use_skip_connections = use_skip_connections

        super().__init__(
            hid_dim=hid_dim,
            conditioning_strategy=conditioning_strategy,
            timestep_embedding_strategy=timestep_embedding_strategy,
            pos_embedding_strategy=pos_embedding_strategy,
            use_self_conditioning=use_self_conditioning,
            label_num_classes=label_num_classes,
            cfg_dropout=cfg_dropout,
        )

    def make_blocks(self):
        block = TriangularSelfAttentionBlock

        # TODO: make the sequence and pairwise state dims configurable
        # in case of using VQVAE
        self.in_blocks = nn.ModuleList(
            [
                block(
                    sequence_state_dim=c_s,
                    pairwise_state_dim=c_z,
                    sequence_head_width=self.trunk_cfg.sequence_head_width,
                    pairwise_head_width=self.trunk_cfg.pairwise_head_width,
                    conditioning_strategy=self.conditioning_strategy,
                    dropout=self.trunk_cfg.dropout,
                    skip=False,
                )
                for _ in range((self.num_blocks - 1) // 2)
            ]
        )

        self.mid_block = block(
            sequence_state_dim=c_s,
            pairwise_state_dim=c_z,
            sequence_head_width=self.trunk_cfg.sequence_head_width,
            pairwise_head_width=self.trunk_cfg.pairwise_head_width,
            conditioning_strategy=self.conditioning_strategy,
            dropout=self.trunk_cfg.dropout,
            skip=False,
        )

        self.out_blocks = nn.ModuleList(
            [
                block(
                    sequence_state_dim=c_s,
                    pairwise_state_dim=c_z,
                    sequence_head_width=self.trunk_cfg.sequence_head_width,
                    pairwise_head_width=self.trunk_cfg.pairwise_head_width,
                    conditioning_strategy=self.conditioning_strategy,
                    dropout=self.trunk_cfg.dropout,
                    skip=self.use_skip_connections,
                )
                for _ in range((self.num_blocks - 1) // 2)
            ]
        )

        self.xavier_init_module(self.in_blocks)
        self.xavier_init_module(self.out_blocks)
        self.xavier_init_module(self.mid_block)

    def _iteration(self, x, z, c, mask):
        x_skips = []
        z_skips = []
        for block in self.in_blocks:
            x, z = block(x, z, c, mask=mask, chunk_size=self.chunk_size)
            if self.use_skip_connections:
                x_skips.append(x)
                z_skips.append(z)

        x, z = self.mid_block(x, z, c, mask=mask, chunk_size=self.chunk_size)

        for block in self.out_blocks:
            x_skip = x_skips.pop() if self.use_skip_connections else None
            z_skip = z_skips.pop() if self.use_skip_connections else None
            x, z = block(
                x,
                z,
                c,
                mask=mask,
                chunk_size=self.chunk_size,
                skip_seq_state=x_skip,
                skip_pairwise_state=z_skip,
            )
        return x, z


if __name__ == "__main__":
    from plaid.datasets import CATHShardedDataModule

    device = torch.device("cuda:1")

    model = PreinitializedTriSelfAttnDenoiser(hid_dim=1024)
    # model = UTriSelfAttnDenoiser(
    #     num_blocks=7,
    #     hid_dim=1024,
    #     conditioning_strategy="length_concat",
    #     use_self_conditioning=True)
    model.to(device)
    # datadir = "/homefs/home/lux70/storage/data/cath/shards/"
    # pklfile = "/homefs/home/lux70/storage/data/cath/sequences.pkl"
    datadir = "/shared/amyxlu/data/cath/shards/"
    pklfile = "/shared/amyxlu/data/cath/sequences.pkl"
    dm = CATHShardedDataModule(
        shard_dir=datadir,
        header_to_sequence_file=pklfile,
    )
    dm.setup("fit")
    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))
    # x, seqlens = batch
    # mask = mask_from_seq_lens(x, seqlens)
    x, mask, sequence = batch
    x, mask = x.to(device), mask.to(device)
    x = x[:4, :64, :]
    mask = mask[:4, :64]
    N, L, _ = x.shape
    t = torch.randint(0, 100, (N,)).to(device)
    import IPython

    IPython.embed()
    epsilon_pred = model(x, t, mask)
    print(epsilon_pred)
