import typing as T

import math
from tqdm import tqdm, trange
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch.optim import AdamW
import lightning as L
import wandb
import pandas as pd
from omegaconf import ListConfig

from plaid.datasets import CATHShardedDataModule
from plaid.transforms import trim_or_pad_batch_first
from plaid.esmfold.misc import batch_encode_sequences
from plaid.utils import LatentScaler, get_lr_scheduler
from plaid.proteins import LatentToSequence, LatentToStructure
from plaid.losses.modules import SequenceAuxiliaryLoss, BackboneAuxiliaryLoss
from plaid.losses.functions import masked_token_accuracy, masked_token_cross_entropy_loss


# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pad_to_multiple(tensor, multiple, dim = -1, value = 0):
    seq_len = tensor.shape[dim]
    m = seq_len / multiple
    if m.is_integer():
        return tensor
    remainder = math.ceil(m) * multiple - seq_len
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(tensor, (*pad_offset, 0, remainder), value = value)

def cast_tuple(val, depth = 1):
    return val if isinstance(val, tuple) else ((val,) * depth)

# factory
def _valid_depth_dtype(depth):
    import pdb;pdb.set_trace()
    if isinstance(depth, int):
        return True
    if isinstance(depth, tuple) or isinstance(depth, list) or isinstance(depth, ListConfig):
        if len(depth) == 3:
            return True
    return False

def get_hourglass_transformer(
    dim,
    *,
    depth,
    shorten_factor,
    downproj_factor,
    attn_resampling,
    updown_sample_type,
    **kwargs
):
    assert _valid_depth_dtype, f'depth must be either an integer or a tuple of 3, indicating (pre_transformer_depth, <nested-hour-glass-config>, post_transformer_depth), got {type(depth)}.'
    assert not (isinstance(depth, int) and shorten_factor), 'there does not need to be a shortening factor when only a single transformer block is indicated (depth of one integer value)'

    if isinstance(depth, int):
        return Transformer(dim = dim, depth = depth, **kwargs)

    return HourglassTransformer(dim = dim, depth = depth, shorten_factor = shorten_factor, downproj_factor = downproj_factor, attn_resampling = attn_resampling, updown_sample_type = updown_sample_type, **kwargs)

# up and down sample classes

class NaiveDownsample(nn.Module):
    def __init__(self, shorten_factor):
        super().__init__()
        self.shorten_factor = shorten_factor

    def forward(self, x):
        return reduce(x, 'b (n s) d -> b n d', 'mean', s = self.shorten_factor)

class NaiveUpsample(nn.Module):
    def __init__(self, shorten_factor):
        super().__init__()
        self.shorten_factor = shorten_factor

    def forward(self, x):
        return repeat(x, 'b n d -> b (n s) d', s = self.shorten_factor)

class LinearDownsample(nn.Module):
    def __init__(self, dim, shorten_factor):
        super().__init__()
        self.proj = nn.Linear(dim * shorten_factor, dim)
        self.shorten_factor = shorten_factor

    def forward(self, x):
        x = rearrange(x, 'b (n s) d -> b n (s d)', s = self.shorten_factor)
        return self.proj(x)

class LinearUpsample(nn.Module):
    def __init__(self, dim, shorten_factor):
        super().__init__()
        self.proj = nn.Linear(dim, dim * shorten_factor)
        self.shorten_factor = shorten_factor

    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, 'b n (s d) -> b (n s) d', s = self.shorten_factor)

# classes

class PreNormLinearDownProjection(nn.Module):
    def __init__(self, dim, downproj_factor):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim // downproj_factor)

    def forward(self, x):
        return self.proj(self.norm(x))

class PreNormLinearUpProjection(nn.Module):
    def __init__(self, dim, downproj_factor):
        super().__init__()
        self.norm = nn.LayerNorm(dim // downproj_factor)
        self.proj = nn.Linear(dim // downproj_factor, dim)

    def forward(self, x):
        return self.proj(self.norm(x))

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        causal = False
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context = None, mask = None):
        h, device = self.heads, x.device
        kv_input = default(context, x)

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        mask_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = rearrange(mask, 'b j -> b () () j')
            sim = sim.masked_fill(~mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            mask = torch.ones(i, j, device = device, dtype = torch.bool).triu_(j - i + 1)
            mask = rearrange(mask, 'i j -> () () i j')
            sim = sim.masked_fill(mask, mask_value)

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)

def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

# transformer classes

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        causal = False,
        heads = 8,
        dim_head = 64,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        norm_out = False
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormResidual(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout, causal = causal)),
                PreNormResidual(dim, FeedForward(dim, mult = ff_mult, dropout = ff_dropout))
            ]))

        self.norm = nn.LayerNorm(dim) if norm_out else nn.Identity()

    def forward(self, x, context = None, mask = None):
        for attn, ff in self.layers:
            x = attn(x, context = context, mask = mask)
            x = ff(x)

        return self.norm(x)

class HourglassTransformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        shorten_factor = 2,
        downproj_factor = 2,
        attn_resampling = True,
        updown_sample_type = 'naive',
        heads = 8,
        dim_head = 64,
        causal = False,
        norm_out = False,
    ):
        super().__init__()
        pre_layers_depth, valley_depth, post_layers_depth = depth

        # shorten factor
        if isinstance(shorten_factor, (tuple, list, ListConfig)):
            shorten_factor, *rest_shorten_factor = shorten_factor
        elif isinstance(valley_depth, int):
            shorten_factor, rest_shorten_factor = shorten_factor, None
        else:
            shorten_factor, rest_shorten_factor = shorten_factor, shorten_factor

        # downproj factor
        if isinstance(downproj_factor, (tuple, list, ListConfig)):
            downproj_factor, *rest_downproj_factor = downproj_factor
        elif isinstance(valley_depth, int):
            downproj_factor, rest_downproj_factor = downproj_factor, None
        else:
            downproj_factor, rest_downproj_factor = downproj_factor, downproj_factor

        transformer_kwargs = dict(
            heads = heads,
            dim_head = dim_head
        )

        self.causal = causal
        self.shorten_factor = shorten_factor
        self.downproj_factor = downproj_factor

        if updown_sample_type == 'naive':
            self.downsample = NaiveDownsample(shorten_factor)
            self.upsample   = NaiveUpsample(shorten_factor)
        elif updown_sample_type == 'linear':
            self.downsample = LinearDownsample(dim, shorten_factor)
            self.upsample   = LinearUpsample(dim, shorten_factor)
        else:
            raise ValueError(f'unknown updown_sample_type keyword value - must be either naive or linear for now')

        self.down_projection = PreNormLinearDownProjection(dim, downproj_factor)
        self.up_projection = PreNormLinearUpProjection(dim, downproj_factor)
        
        self.valley_transformer = get_hourglass_transformer(
            dim = dim // downproj_factor,
            shorten_factor = rest_shorten_factor,
            downproj_factor = rest_downproj_factor,
            depth = valley_depth,
            attn_resampling = attn_resampling,
            updown_sample_type = updown_sample_type,
            causal = causal,
            **transformer_kwargs
        )

        self.attn_resampling_context_downproj = PreNormLinearDownProjection(dim, downproj_factor) if attn_resampling else None
        self.attn_resampling_context_upproj = PreNormLinearUpProjection(dim, downproj_factor) if attn_resampling else None
        self.attn_resampling_pre_valley = Transformer(dim = dim // downproj_factor, depth = 1, **transformer_kwargs) if attn_resampling else None
        self.attn_resampling_post_valley = Transformer(dim = dim, depth = 1, **transformer_kwargs) if attn_resampling else None

        self.pre_transformer = Transformer(dim = dim, depth = pre_layers_depth, causal = causal, **transformer_kwargs)
        self.post_transformer = Transformer(dim = dim, depth = post_layers_depth, causal = causal, **transformer_kwargs)
        self.norm_out = nn.LayerNorm(dim) if norm_out else nn.Identity()

    def forward(self, x, mask = None):
        # b : batch, n : sequence length, d : feature dimension, s : shortening factor
        s, b, n = self.shorten_factor, *x.shape[:2]
        
        # top half of hourglass, pre-transformer layers
        x = self.pre_transformer(x, mask = mask)
        
        # pad to multiple of shortening factor, in preparation for pooling
        x = pad_to_multiple(x, s, dim = -2)

        if exists(mask):
            padded_mask = pad_to_multiple(mask, s, dim = -1, value = False)

        # save the residual, and for "attention resampling" at downsample and upsample
        x_residual = x.clone()

        # if autoregressive, do the shift by shortening factor minus one
        if self.causal:
            shift = s - 1
            x = F.pad(x, (0, 0, shift, -shift), value = 0.)

            if exists(mask):
                padded_mask = F.pad(padded_mask, (shift, -shift), value = False)

        # naive average pool
        downsampled = self.downsample(x)
        if exists(mask):
            downsampled_mask = reduce(padded_mask, 'b (n s) -> b n', 'sum', s = s) > 0
        else:
            downsampled_mask = None

        # also possibly reduce along dim=-1
        downsampled = self.down_projection(downsampled)

        # pre-valley "attention resampling" - they have the pooled token in each bucket attend to the tokens pre-pooled
        if exists(self.attn_resampling_pre_valley):
            if exists(mask):
                attn_resampling_mask = rearrange(padded_mask, 'b (n s) -> (b n) s', s = s)
            else:
                attn_resampling_mask = None
            downsampled = self.attn_resampling_pre_valley(
                rearrange(downsampled, 'b n d -> (b n) () d'),
                rearrange(self.attn_resampling_context_downproj(x), 'b (n s) d -> (b n) s d', s = s),
                mask = attn_resampling_mask
            )

            downsampled = rearrange(downsampled, '(b n) () d -> b n d', b = b)
            
        # the "valley" - either a regular transformer or another hourglass
        x = self.valley_transformer(downsampled, mask = downsampled_mask)
        valley_out = x.clone()

        # naive repeat upsample
        x = self.upsample(x)
        x = self.up_projection(x)
        
        # add the residual
        x = x + x_residual
        
        # post-valley "attention resampling"
        if exists(self.attn_resampling_post_valley):
            x = self.attn_resampling_post_valley(
                rearrange(x, 'b (n s) d -> (b n) s d', s = s),
                rearrange(self.attn_resampling_context_upproj(valley_out), 'b n d -> (b n) () d')
            )

            x = rearrange(x, '(b n) s d -> b (n s) d', b = b)


        # bring sequence back to original length, if it were padded for pooling
        x = x[:, :n]
        
        # post-valley transformers
        x = self.post_transformer(x, mask = mask)
        return self.norm_out(x)


class HourglassTransformerLightningModule(L.LightningModule):
    def __init__(
        self,
        dim,
        *,
        depth,
        shorten_factor = 2,
        downproj_factor = 2,
        attn_resampling = True,
        updown_sample_type = 'naive',
        heads = 8,
        dim_head = 64,
        causal = False,
        norm_out = False,
        # learning rates
        lr = 1e-4,
        lr_sched_type: str = "constant",
        lr_num_warmup_steps: int = 0,
        lr_num_training_steps: int = 10_000_000,
        lr_num_cycles: int = 1,
        # scaler
        latent_scaler = LatentScaler(),
        lm_embedder_type: str = "esmfold",
        # auxiliary losses
        seq_loss_weight: float = 0.0,
        struct_loss_weight: float = 0.0,
        log_sequence_loss = True,
        log_structure_loss = True,
    ):
        super().__init__()
        self.latent_scaler = latent_scaler
        self.lm_embedder_type = lm_embedder_type
        self.log_sequence_loss = log_sequence_loss or (seq_loss_weight > 0.)
        self.log_structure_loss = log_structure_loss or (struct_loss_weight > 0.)
        self.seq_loss_weight = seq_loss_weight
        self.struct_loss_weight = struct_loss_weight

        self.lr = lr
        self.lr_sched_type = lr_sched_type
        self.lr_num_warmup_steps = lr_num_warmup_steps
        self.lr_num_training_steps = lr_num_training_steps
        self.lr_num_cycles = lr_num_cycles

        self.model = get_hourglass_transformer(
            dim=dim,
            depth=depth,
            shorten_factor=shorten_factor,
            downproj_factor=downproj_factor,
            attn_resampling=attn_resampling,
            updown_sample_type=updown_sample_type,
            heads=heads,
            dim_head=dim_head,
            causal=causal,
            norm_out=norm_out
        )
        self.model.to(self.device)

        if self.log_sequence_loss:
            self.sequence_constructor = LatentToSequence()
            self.sequence_constructor.to(self.device)
            self.seq_loss_fn = SequenceAuxiliaryLoss(self.sequence_constructor)

        if self.log_structure_loss:
            self.structure_constructor = LatentToStructure()
            self.structure_constructor.to(self.device)
            self.structure_loss_fn = BackboneAuxiliaryLoss(self.structure_constructor)
        self.save_hyperparameters()
    
    def forward(self, x, mask):
        return self.model(x, mask)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr
        )
        scheduler = get_lr_scheduler(
            optimizer=optimizer,
            sched_type=self.lr_sched_type,
            num_warmup_steps=self.lr_num_warmup_steps,
            num_training_steps=self.lr_num_training_steps,
            num_cycles=self.lr_num_cycles,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def run_batch(self, batch, prefix="train"):
        x, sequences, gt_structures = batch
        # todo: might be easier to just save seqlen when processing for making masks
        # in this form, the sequences *must* have the correct length (trimmed, no pad)
        tokens, mask, _, _, _ = batch_encode_sequences(sequences)
        if mask.shape[1] != x.shape[1]:
            # pad with False
            mask = trim_or_pad_batch_first(mask, x.shape[1], pad_idx=0)
            tokens = trim_or_pad_batch_first(tokens, x.shape[1], pad_idx=0)

        if self.lm_embedder_type != "esmfold":
            x, mask, tokens = x[:, 1:-1, :], mask[:, 1:-1], tokens[:, 1:-1]
        else:
            x = self.latent_scaler.scale(x)
        mask = mask.bool()
        x, mask = x.to(self.device), mask.to(self.device)
        
        output = self(x, mask)
        loss = F.mse_loss(x, output)
        scaled_output = self.latent_scaler.unscale(output)
        self.log_dict({f"{prefix}/recons_loss": loss.item()}, on_step=(prefix != "val"), on_epoch=True)
        
        if self.log_sequence_loss:
            seq_loss, seq_loss_dict, recons_strs = self.seq_loss_fn(scaled_output, tokens, mask, return_reconstructed_sequences=True)
            tbl = pd.DataFrame({"reconstructed": recons_strs, "original": sequences})
            seq_loss_dict = {f"{prefix}/{k}": v for k, v in seq_loss_dict.items()}
            self.log_dict(seq_loss_dict, on_step=(prefix != "val"), on_epoch=True)
            wandb.log({f"{prefix}/recons_strs_tbl": wandb.Table(dataframe=tbl)})
            loss += seq_loss * self.seq_loss_weight
        
        if self.log_structure_loss:
            struct_loss, struct_loss_dict = self.structure_loss_fn(scaled_output, gt_structures, sequences)
            struct_loss_dict = {f"{prefix}/{k}": v for k, v in struct_loss_dict.items()}
            self.log_dict(struct_loss_dict, on_step=(prefix != "val"), on_epoch=True)
            wandb.log(struct_loss_dict)
            loss += struct_loss * self.struct_loss_weight

        return loss
    
    def training_step(self, batch, batch_idx):
        return self.run_batch(batch, prefix="train")
    
    def validation_step(self, batch, batch_idx):
        return self.run_batch(batch, prefix="val")
    
if __name__ == "__main__":
    from plaid.datasets import CATHStructureDataModule
    shard_dir = "/homefs/home/lux70/storage/data/cath/shards/"
    pdb_dir = "/homefs/home/lux70/storage/data/cath/dompdb/"
    latent_scaler = LatentScaler()
    embedder = "esmfold"
    D = 1024 if embedder == "esmfold" else 320
    dm = CATHStructureDataModule(
        shard_dir=shard_dir,
        pdb_path_dir=pdb_dir,
        embedder=embedder,
        seq_len=256,
        batch_size=16
    )
    dm.setup()
    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()

    module = HourglassTransformerLightningModule(
        dim = D,                     # feature dimension
        heads = 8,                      # attention heads
        dim_head = 64,                  # dimension per attention head
        shorten_factor = 2,             # shortening factor
        depth = (4, 2, 4),              # tuple of 3, standing for pre-transformer-layers, valley-transformer-layers (after downsample), post-transformer-layers (after upsample) - the valley transformer layers can be yet another nested tuple, in which case it will shorten again recursively
        attn_resampling = True,
        updown_sample_type = "naive",
        causal = True,
        norm_out = True,
        latent_scaler=latent_scaler,
        log_sequence_loss=True,
        log_structure_loss=True,
        seq_loss_weight=1.0,
        struct_loss_weight=1.0
    )
    batch = next(iter(train_dataloader))
    module.training_step(batch, 0)