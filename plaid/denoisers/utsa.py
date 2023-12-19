"""
U-TriangularSelfAttention (U-TSA) denoiser
"""

import typing as T
import einops
from pathlib import Path
import os

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
from plaid.denoisers.modules import TriSelfAttnDenoiserBlock


PathLike = T.Union[str, Path]
ArrayLike = T.Union[np.ndarray, torch.Tensor]


class UTriSelfAttnDenoiser(BaseDenoiser):
    def __init__(
        self,
        hid_dim: int,
        num_blocks: int, 
        conditioning_strategy: str = "hidden_concat",
        timestep_embedding_strategy: str = "fourier",
        pos_embedding_strategy: str = "rotary",
        use_self_conditioning: bool = False,
        use_skip_connections: bool = True,
        label_num_classes: T.Optional[int] = None,
        cfg_dropout: float = 0.0,
    ):
        super().__init__(
            hid_dim=hid_dim,
            timestep_embedding_strategy=timestep_embedding_strategy,
            pos_embedding_strategy=pos_embedding_strategy,
            use_self_conditioning=use_self_conditioning,
            label_num_classes=label_num_classes,
            cfg_dropout=cfg_dropout
        )

        # fixed dimensions
        trunk_cfg: FoldingTrunkConfig = FoldingTrunkConfig()
        self.trunk_cfg = trunk_cfg
        self.chunk_size = trunk_cfg.chunk_size
        self.pairwise_positional_embedding = RelativePosition(
            trunk_cfg.position_bins, c_z 
        )

        block = TriSelfAttnDenoiserBlock
        self.use_skip_connections = use_skip_connections
        self.conditioning_strategy = conditioning_strategy

        # TODO: make the sequence and pairwise state dims configurable
        # in case of using VQVAE
        assert hid_dim == c_s
        self.in_blocks = nn.ModuleList(
            [
                block(
                    sequence_state_dim=c_s,
                    pairwise_state_dim=c_z,
                    sequence_head_width=trunk_cfg.sequence_head_width,
                    pairwise_head_width=trunk_cfg.pairwise_head_width,
                    dropout=trunk_cfg.dropout,
                    skip=False,
                )
                for _ in range((num_blocks - 1) // 2)
            ]
        )

        self.mid_block = block(
            sequence_state_dim=c_s,
            pairwise_state_dim=c_z,
            sequence_head_width=trunk_cfg.sequence_head_width,
            pairwise_head_width=trunk_cfg.pairwise_head_width,
            dropout=trunk_cfg.dropout,
            skip=False,
        )

        self.out_blocks = nn.ModuleList(
            [
                block(
                    sequence_state_dim=c_s,
                    pairwise_state_dim=c_z,
                    sequence_head_width=trunk_cfg.sequence_head_width,
                    pairwise_head_width=trunk_cfg.pairwise_head_width,
                    dropout=trunk_cfg.dropout,
                    skip=use_skip_connections,
                )
                for _ in range((num_blocks - 1) // 2)
            ]
        )

        # self.s_mlp_proj = nn.Linear(c_s, c_s)
        # self.z_mlp_proj = nn.Linear(c_z, c_z)
        # self.trunk2sm_s = nn.Linear(c_s, structure_module_c_s, bias=False)
        # self.trunk2sm_z = nn.Linear(c_z, structure_module_c_z, bias=False)

        self.xavier_init_module(self.in_blocks)
        self.xavier_init_module(self.out_blocks)
        self.xavier_init_module(self.mid_block)
        self.xavier_init_module(self.s_mlp_proj)
        self.xavier_init_module(self.z_mlp_proj)

    def set_chunk_size(self, chunk_size):
        # This parameter means the axial attention will be computed
        # in a chunked manner. This should make the memory used more or less O(L) instead of O(L^2).
        # It's equivalent to running a for loop over chunks of the dimension we're iterative over,
        # where the chunk_size is the size of the chunks, so 128 would mean to parse 128-lengthed chunks.
        self.chunk_size = chunk_size

    def forward(self, x, t, mask, z=None, y=None, x_self_condition=None): 
        """
        x: (B, L, C) sequence representation
        z: (B, L, L, C) pairwise representation
        t: (B, L) time embedding
        y: (B,) label
        mask: (B, L) mask
        """
        # self conditioning should be either zeros or x_prev, determined in outer loop
        if not x_self_condition is None:
            x = self.self_conditioning_mlp(torch.cat((x, x_self_condition), dim=-1))
        
        B, L, _ = x.shape
        if z is None:
            z = x.new_zeros(B, L, L, c_z)
        
        t = self.time_embedding(t)
        if not y is None:
            y = self.label_embedding(y)
            c = t + y
        else:
            c = t
        x = self.conditioning(x, c, self.conditioning_strategy, proj_fn=self.conditioning_mlp)

        residx = torch.arange(L, device=x.device, dtype=int)
        z = z + self.pairwise_positional_embedding(residx, mask=mask)

        # TODO: multiple iterations?
        x, z = self._iteration(x, z, mask)

        # remove dimensions added due to concating time/cond embeddings
        if self.extras != 0:
            x = x[:, : -self.extras, :]
            z = z[:, : -self.extras, : -self.extras, :]
        return x, z

    def _iteration(self, x, z, mask):
        x_skips = []
        z_skips = []
        for block in self.in_blocks:
            x, z = block(x, z, mask=mask, chunk_size=self.chunk_size)
            if self.use_skip_connections:
                x_skips.append(x)
                z_skips.append(z)

        x, z = self.mid_block(x, z, mask=mask, chunk_size=self.chunk_size)

        for block in self.out_blocks:
            x_skip = x_skips.pop() if self.use_skip_connections else None
            z_skip = z_skips.pop() if self.use_skip_connections else None
            x, z = block(
                x,
                z,
                mask=mask,
                chunk_size=self.chunk_size,
                skip_seq_state=x_skip,
                skip_pairwise_state=z_skip,
            )
        return x, z 


if __name__ == "__main__":
    model = UTriSelfAttnDenoiser(num_blocks=7, hid_dim=1024)
    import IPython;IPython.embed()