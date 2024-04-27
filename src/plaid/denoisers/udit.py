from .dit import BaseDiT, DiTBlock, modulate, InputProj

import torch
from torch import nn



class UDiTBlock(DiTBlock):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, skip_connection=False): 
        super().__init__(hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio)
        if skip_connection:
            self.skip_proj = InputProj(input_dim=hidden_size * 2, hidden_size=hidden_size, bias=False)
        else:
            self.skip_proj = None
        
    def forward(self, x, c, mask, x_skip=None):
        """Apply multi-head attention (with mask) and adaLN conditioning (mask agnostic)."""
        if x_skip is not None:
            assert not self.skip_proj is None, "Block is missing skip projection layer."
            x = torch.cat([x, x_skip], dim=-1)  # (N, L, C*2)
            x = self.skip_proj(x)  # (N, L, C)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class UDiT(BaseDiT):
    def __init__(
        self,
        input_dim=8,
        hidden_size=1024,
        max_seq_len=512, 
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        use_self_conditioning=False,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_size=hidden_size,
            max_seq_len=max_seq_len,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            use_self_conditioning=use_self_conditioning,
        )
    
    def make_blocks(self):
        assert self.depth % 2 != 0, "Depth must be an odd number."

        # same as before but with mid blocks
        self.in_blocks = nn.ModuleList([
            UDiTBlock(self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio, skip_connection=False) for _ in range(self.depth // 2)
        ])
        self.mid_block = UDiTBlock(self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio, skip_connection=False)
        self.out_blocks = nn.ModuleList([
            UDiTBlock(self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio, skip_connection=True) for _ in range(self.depth // 2)
        ])

    def forward(self, x, t, mask=None, x_self_cond=None):
        if x_self_cond is not None:
            x = self.self_conditioning_mlp(torch.cat([x, x_self_cond], dim=-1))
        x = self.x_proj(x)
        x += self.pos_embed[:, :x.shape[1], :]
        t = self.t_embedder(t)                   # (N, D)
        c = t  # TODO: add y embedding and clf guidance
        
        if mask is None:
            mask = torch.ones(x.shape[:2], device=x.device).bool()

        # Modified to add skips:
        x_skips = []

        for block in self.in_blocks:
            x = block(x, c, mask)                # (N, L, D)
            x_skips.append(x)
        
        x = self.mid_block(x, c, mask)

        for block in self.out_blocks:
            x_skip = x_skips.pop()
            x = block(x, c, mask, x_skip=x_skip)

        x = self.final_layer(x, c)               # (N, L, out_channels)

        return x
