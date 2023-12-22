import dataclasses
import os
import typing as T
from pathlib import Path

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math
import einops
from plaid.esmfold import FoldingTrunkConfig, FoldingTrunk, RelativePosition
from plaid.denoisers.modules import TriangularSelfAttentionBlock

import json

ESMFOLD_C_S = 1024
ESMFOLD_C_Z = 128
NUM_SECONDARY_STRUCTURE_BINS = 6
NUM_SECONDARY_STRUCTURE_BINS_WITH_DUMMY = 7  
UNCOND_IDX = 6  
UNIREF90_PATH = "/shared/amyxlu/data/uniref90/uniref90.fasta"


class GaussianFourierProjection(nn.Module):
    """
    https://arxiv.org/abs/2006.10739
    https://github.com/tancik/fourier-feature-networks/blob/master/Demo.ipynb
    """

    def __init__(self, embed_dim: int, scale: float = 2 * torch.pi):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        w = torch.randn(embed_dim // 2) * scale
        assert w.requires_grad == False
        self.register_buffer("W", w)

    def forward(self, t: torch.Tensor):
        # t: (batch_size,)
        # w: (embed_dim // 2,)
        t = t.to(self.W.dtype)
        t_proj = 2.0 * torch.pi * t[:, None] @ self.W[None, :]
        embed = torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)
        return embed


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.dim = embed_dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        # TODO: Not yet tested.
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        # half_dim shape
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # outer product (batch, 1) x (1, half_dim) -> (batch x half_dim)
        embeddings = time[:, None] * embeddings[None, :]
        # sin and cosine embeddings
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TransformerPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class SecondaryStructureEmbedding(torch.nn.Module):
    """Process quantized secondary structure information
    into a fixed-size embedding."""

    def __init__(
        self,
        embed_dim=ESMFOLD_C_S,
        use_cf_guidance: bool = False,
    ) -> None:
        super().__init__()
        if use_cf_guidance:
            self.nbins = NUM_SECONDARY_STRUCTURE_BINS_WITH_DUMMY
        else:
            self.nbins = NUM_SECONDARY_STRUCTURE_BINS

        self.helix_embedding = nn.Embedding(self.nbins, embed_dim)
        self.sheet_embedding = nn.Embedding(self.nbins, embed_dim)
        self.turns_embedding = nn.Embedding(self.nbins, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        helix, sheet, turns = x[:, 0], x[:, 1], x[:, 2]

        helix_emb = self.helix_embedding(helix)
        sheet_emb = self.sheet_embedding(sheet)
        turns_emb = self.turns_embedding(turns)
        return helix_emb, sheet_emb, turns_emb




@dataclasses.dataclass
class UIPAExperimentConfig:
    # ==== Data specifications ====
    data_version: str = "230412"
    seq_len: int = 128
    use_esm_attn_map: bool = False
    remove_bos_eos_from_s: bool = True
    fasta_file = UNIREF90_PATH
    frac_for_validation = 0.00001
    use_toy_data = False

    # ==== General tinkering ====
    # Loss weights
    loss_weight_reconstruction: float = 1.0
    loss_weight_sequence: float = 0.0
    loss_weight_structure: float = 0.0
    assert (
        loss_weight_sequence + loss_weight_structure + loss_weight_reconstruction == 1.0
    )

    # Diffusion
    T: int = 2000  # Total number of diffusion timesteps
    beta_schedule: str = "chen_sigmoid_start0_end3_tau0.5" # "cosine" / "linear" / "chen_sigmoid_start0_end3_tau0.5" / "chen_cosine_start0.2_end1_tau3" / etc.
    noise_var: float = 1.0  # Noise variance for diffusion
    timestep_proj: str = "gaussian_fourier"  # "gaussian_fourier" / "sinusoidal"
    x_scale_factor: float = 0.5  # Scale factor for x, following Chen et al.

    # Model architecture
    model_type: str = "uipa"  # "uipa"
    num_blocks: int = 27  # Number of IPA blocks, must be odd number
    assert num_blocks % 2 == 1
    assert num_blocks >= 3
    num_recycles: int = 4  # Use 0 if not recycling at all. 

    temb_cat_strategy: str = "concat"  # "add" / "concat"
    cemb_cat_strategy: str = "concat"  # "add" / "concat"

    # forward pass
    latent_scaling_mode: str = "channel_standardize"  # "none" / "global_minmaxnorm" / "channel_minmaxnorm" / "global_standardize" / "channel_standardize"
    clip_model_output: bool = False  # If true, clips the output of the model to [-1, 1] (should mirror the latent scaling mode)
    norm_model_output: bool = False  # If true, applies layer norm to model output (should mirror the latent scaling mode)
    mlp_proj_model_output: bool = (
        False  # If true, applies a trainable MLP projection to the model output.
    )

    # ==== Training specifications ===
    device_id: int = 0
    batch_size: int = 4
    n_epochs: int = 4000
    loss_fn: str = "huber"  # "mse" / "huber" / "f1"
    optimizer_type: str = "adamw"  # "adam" / "adamw" / "adafactor"
    grad_accumulation_steps: int = 1
    overfit: bool = False
    seed: int = 0
    apply_weight_norm: bool = False
    resume_training_from: str = (
        None  # String in the format of "{run_id}:{itr}" (e.g. "5jrpyn25:2000")
    )
    num_workers: int = 4

    # Conditioning
    cond_key: str = "secondary_structure"  # "none" for unconditional , otherwise "secondary_structure"
    cond_key_dropout_prob: float = 0.2

    # LR
    lr_base: float = 1e-4
    lr_scheduler: str = "constant"  # "cosine" / "constant" / "linear" / "constant_with_warmup" / "cosine_with_restarts" / "inverse_sqrt"
    lr_warmups: int = 0
    lr_num_cycles: int = 1

    # ==== Logging ====
    log_grad_norms: bool = False
    log_itr_metrics_every: int = 64
    save_ckpt_every: int = 10000
    val_every: int = 20000
    ckpt_dir: str = "/shared/amyxlu/dprot/ckpts/"
    log_reconstructions: bool = False
    log_samples: bool = False
    debug_mode: bool = True

    def __post_init__(self):
        # Use toy data in dev mode
        if self.debug_mode or self.use_toy_data:
            self.fasta_file = f"{os.environ['DPROT_PROJECT_DIR']}/data/toy.fasta"
            self.frac_for_validation = 0.2


###################################################################################################
###################################################################################################

def _maybe_dropout(
    original_array: torch.Tensor,
    replace_with_value: int,
    p_dropout: float = 0.2,
) -> int:
    """
    p_dropout: whether or not to replace the conditioning variable with a dummy token
    replace_with: the class index to replace. Must be an integer.
    """
    device = original_array.device
    dropout_mask = torch.rand(original_array.shape) < p_dropout
    dropout_mask = dropout_mask.to(device)
    original_array[dropout_mask] = replace_with_value
    return original_array


def _modify_cond_dict_maybe_dropout(cond_dict: T.Dict[str, T.Any], p_dropout=0.2):
    for key, orig_arr in cond_dict.items():
        # only handles secondary structure bins for now
        # use the last index, aka number of bins - 1, as the dummy token
        replace_with_value = NUM_SECONDARY_STRUCTURE_BINS_WITH_DUMMY - 1
        cond_dict[key] = _maybe_dropout(
            orig_arr, replace_with_value, p_dropout=p_dropout
        )
    return cond_dict


def _pointwise_add_emb(ss0, emb):
    # ss0: (B, L, C)
    # emb: (B, C)
    emb = einops.repeat(emb, "b c -> b l c", l=ss0.shape[1])
    assert ss0.shape == emb.shape
    ss0 = ss0 + emb
    return ss0


def _concat_emb(ss0, emb):
    # ss0: (B, L, C)
    # emb: (B, C)
    emb = einops.rearrange(emb, "b c -> b 1 c")
    ss0 = torch.concat((ss0, emb), dim=1)
    return ss0


###################################################################################################
###################################################################################################

class UIPA(nn.Module):
    """Deprecated, only retained to load old checkpoints."""
    def __init__(
        self,
        cfg: UIPAExperimentConfig,
    ):
        super().__init__()

        ####### Set up some dimensions #######
        self.cfg = cfg
        self.is_cond = cfg.cond_key != "none"
        c_s, c_z = ESMFOLD_C_S, ESMFOLD_C_Z
        self.hid_dim = ESMFOLD_C_S

        trunk_cfg: FoldingTrunkConfig = FoldingTrunkConfig()
        self.trunk_cfg = trunk_cfg
        self.chunk_size = trunk_cfg.chunk_size

        structure_module_c_s = 384
        structure_module_c_z = 128

        ####### Flexibly create additional embeddings ########
        self.pairwise_positional_embedding = RelativePosition(
            trunk_cfg.position_bins, c_z
        )

        # Create timestep embedding
        if cfg.timestep_proj == "gaussian_fourier":
            self.timestep_encoder = GaussianFourierProjection(ESMFOLD_C_S)
        elif cfg.timestep_proj == "sinusoidal":
            self.timestep_encoder = SinusoidalPositionEmbeddings(ESMFOLD_C_S)

        # Create conditional embedding and select input conditional information format
        if not hasattr(cfg, "cond_key_dropout_prob"):
            self.use_classifier_free_guidance = False
        else:
            self.use_classifier_free_guidance = not (cfg.cond_key_dropout_prob == 0.0)
            assert 0.0 <= cfg.cond_key_dropout_prob <= 1.0

        self.cond_embed = SecondaryStructureEmbedding(
            ESMFOLD_C_S, self.use_classifier_free_guidance
        )

        # Note down how much we're extending the L dimension by
        self.extras = 0
        if cfg.temb_cat_strategy == "concat":
            self.extras += 1
        if (cfg.cemb_cat_strategy == "concat") and (self.is_cond):
            self.extras += 3  # helix emb, turn emb, sheet emb

        ####### Set Up U-IPA blocks #######
        block = TriangularSelfAttentionBlock
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
                for _ in range((cfg.num_blocks - 1) // 2)
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
                    skip=True,
                )
                for _ in range((cfg.num_blocks - 1) // 2)
            ]
        )

        # Maybe also transform final output
        self.s_instance_norm = nn.InstanceNorm1d(ESMFOLD_C_S)
        self.z_instance_norm = nn.InstanceNorm2d(ESMFOLD_C_S)

        self.s_mlp_proj = nn.Linear(ESMFOLD_C_S, ESMFOLD_C_S)
        self.z_mlp_proj = nn.Linear(ESMFOLD_C_Z, ESMFOLD_C_Z)

        self.trunk2sm_s = nn.Linear(c_s, structure_module_c_s, bias=False)
        self.trunk2sm_z = nn.Linear(c_z, structure_module_c_z, bias=False)

        self._init_weights(self.in_blocks)
        self._init_weights(self.out_blocks)
        self._init_weights(self.mid_block)
        self._init_weights(self.s_mlp_proj)
        self._init_weights(self.z_mlp_proj)

        # for compatibility w/ diffusion modules
        self.use_self_conditioning = False

    def conditioning(self, s: Tensor, cond_dict: T.Dict[str, T.Any]) -> Tensor:
        """
        Get conditional embedding from conditional dictionary.
        """
        assert (
            self.is_cond
        ), "This model is not conditional, set --cond_key <cond_key> to use conditional model."

        # Add other conditioning variables here...
        if self.cfg.cond_key != "secondary_structure":
            raise NotImplementedError

        ##############################
        # secondary structure fraction
        ##############################
        if self.use_classifier_free_guidance:
            # a dummy variable has also been added in the architecture
            assert (
                self.cond_embed.nbins
                == NUM_SECONDARY_STRUCTURE_BINS_WITH_DUMMY
            )
            cond_dict = _modify_cond_dict_maybe_dropout(cond_dict)
        else:
            assert self.cond_embed.nbins == NUM_SECONDARY_STRUCTURE_BINS

        # Add conditional embedding info
        helix_emb, sheet_emb, turns_emb = self.cond_embed(
            cond_dict["secondary_structure"]
        )
        if self.cfg.cemb_cat_strategy == "add":
            s = _pointwise_add_emb(s, helix_emb)
            s = _pointwise_add_emb(s, sheet_emb)
            s = _pointwise_add_emb(s, turns_emb)

        elif self.cfg.cemb_cat_strategy == "concat":
            s = _concat_emb(s, helix_emb)
            s = _concat_emb(s, sheet_emb)
            s = _concat_emb(s, turns_emb)

        return s

    def add_time_embedding(self, s: Tensor, t_s: Tensor):
        temb = self.timestep_encoder(t_s)  # (B,C)
        if self.cfg.temb_cat_strategy == "add":
            s = _pointwise_add_emb(s, temb)
        elif self.cfg.temb_cat_strategy == "concat":
            # ensure that the order of concating t and c is always the same!
            s = _concat_emb(s, temb)
        else:
            raise ValueError(f"Invalid strategy for adding time embedding.")

        return s

    def set_chunk_size(self, chunk_size):
        # This parameter means the axial attention will be computed
        # in a chunked manner. This should make the memory used more or less O(L) instead of O(L^2).
        # It's equivalent to running a for loop over chunks of the dimension we're iterative over,
        # where the chunk_size is the size of the chunks, so 128 would mean to parse 128-lengthed chunks.
        self.chunk_size = chunk_size

    def _init_weights(self, m):
        for p in m.parameters():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            else:
                if p.dim() > 1:
                    torch.nn.init.xavier_normal_(p)

    def forward(
        self,
        s_s_t: Tensor,
        t_s: Tensor,
        mask: Tensor = None,
        residx: Tensor = None,
        s_z_0: Tensor = None,
        cond_dict: T.Optional[T.Dict[str, T.Any]] = None,
        project_trunk_outputs: T.Optional[bool] = False,
        return_z: T.Optional[bool] = False,
        *args, **kwargs
    ):
        B, orig_L, _ = s_s_t.shape
        if residx is None:
            residx = einops.repeat(torch.arange(orig_L, device=s_s_t.device, dtype=int), "L -> B L", B=B)
        if mask is None:
            mask = torch.ones_like(residx)
        ####### Add time and conditional embeddings #######
        s_s_t = self.add_time_embedding(
            s_s_t, t_s
        )  # (B, L, C) -> (B, L + n_added_emb, C)
        if self.is_cond:
            assert not cond_dict is None
            s_s_t = self.conditioning(s_s_t, cond_dict)

        # make sure that the dimensions are as expected after concatenating time/cond embedings
        assert self.extras == s_s_t.shape[1] - orig_L

        # Initialize z as zeros as in the ESMFold codebase
        # TODO: try how well this works if we initialize as contacts?
        # would need to figure out how to address embedding concatenation.
        if s_z_0 is None:
            b, l, c = s_s_t.shape
            s_z_0 = s_s_t.new_zeros(b, l, l, c)

        ####### Handle variable lengths caused by concatenation of embeddings ########
        assert residx.shape[1] == mask.shape[1]
        if self.extras != 0:
            zeros = residx.new_zeros(s_s_t.shape[0], self.extras)
            residx = torch.concat((residx, zeros), dim=1)
            # TODO: should the mask here be 0 or 1? Maybe we *do* want to attend to some of the concatenated pos?
            mask = torch.concat((mask, zeros), dim=1)

        s_z_0 = s_z_0 + self.pairwise_positional_embedding(residx, mask=mask)

        # to match esmfold variable names
        s, z = s_s_t, s_z_0

        ####### Blocks with U-ViT style skip connections ########
        block = TriangularSelfAttentionBlock
        s_skips = []
        z_skips = []
        for block in self.in_blocks:  # TODO:add in_blocks attribute
            s, z = block(s, z, mask=mask, chunk_size=self.chunk_size)
            s_skips.append(s)
            z_skips.append(z)

        s, z = self.mid_block(s, z)

        for block in self.out_blocks:
            s, z = block(
                s,
                z,
                mask=mask,
                chunk_size=self.chunk_size,
                skip_seq=s_skips.pop(),
                skip_z=z_skips.pop(),
            )

        if self.extras != 0:
            s = s[:, : -self.extras, :]
            z = z[:, : -self.extras, : -self.extras, :]

        ###### Output projections ######
        if self.cfg.norm_model_output:
            s = self.s_instance_norm(s)
            z = self.z_instance_norm(z)

        if self.cfg.clip_model_output:
            raise NotImplementedError

        if self.cfg.mlp_proj_model_output:
            s = self.s_mlp_proj(s)
            z = self.z_mlp_proj(z)

        if project_trunk_outputs:
            s = self.trunk2sm_s(s)
            z = self.trunk2sm_z(z)

        if return_z:
            return s, z
        else:
            return s
    
    def model_predictions(self, x, t, model_kwargs={}, clip_x_start=False):
        return self.forward(x, t, **model_kwargs)

    def prepare_esmfold_auxiliary_inputs(self, shape, device, add_extra_dim=True):
        """Given the input sequence representation, create vanilla inputs with same shape"""
        assert len(shape) == 3
        B, L, _ = shape

        # mask and residx will be reshaped in the forward pass if necessary
        mask = torch.ones(B, L).long().to(device)
        residx = torch.arange(L).unsqueeze(0).repeat(B, 1).long().to(device)

        if add_extra_dim:
            L += self.extras

        s_z_0 = torch.zeros(B, L, L, ESMFOLD_C_Z).to(device)

        return {"s_z_0": s_z_0, "mask": mask, "residx": residx}


def load_old_checkpoint():
    cfg = UIPAExperimentConfig()
    ckptdir = Path("/shared/amyxlu/dprot/ckpts/2eiqqk2u")
    d = json.load(open(ckptdir / "config.json", "r"))
    cfg.__dict__.update(d)

    model = UIPA(cfg)
    ckpt = torch.load(ckptdir / "itr1440000.ckpt", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    return model



# if __name__ == "__main__":
#     from pathlib import Path
#     import json
#     cfg = UIPAExperimentConfig()
#     ckptdir = Path("/shared/amyxlu/dprot/ckpts/2eiqqk2u")
#     d = json.load(open(ckptdir / "config.json", "r"))
#     cfg.__dict__.update(d)

#     model = UIPA(cfg)
#     ckpt = torch.load(ckptdir / "itr1440000.ckpt", map_location="cpu")
#     model.load_state_dict(ckpt["model_state_dict"], strict=False)
#     # _IncompatibleKeys(missing_keys=[], unexpected_keys=['recycle_s_norm.weight', 'recycle_s_norm.bias', 'recycle_z_norm.weight', 'recycle_z_norm.bias'])

