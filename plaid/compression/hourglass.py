import math
from tqdm import tqdm, trange
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch.optim import AdamW

from plaid.datasets import CATHShardedDataModule, CATHStructureDataModule
from plaid.esmfold.misc import batch_encode_sequences
from plaid.utils import LatentScaler
from plaid.proteins import LatentToSequence, LatentToStructure
from plaid.losses.modules import SequenceAuxiliaryLoss, BackboneAuxiliaryLoss
from plaid.transforms import trim_or_pad_batch_first

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

def get_hourglass_transformer(
    dim,
    *,
    depth,
    shorten_factor,
    attn_resampling,
    updown_sample_type,
    **kwargs
):
    assert isinstance(depth, int) or (isinstance(depth, tuple)  and len(depth) == 3), 'depth must be either an integer or a tuple of 3, indicating (pre_transformer_depth, <nested-hour-glass-config>, post_transformer_depth)'
    assert not (isinstance(depth, int) and shorten_factor), 'there does not need to be a shortening factor when only a single transformer block is indicated (depth of one integer value)'

    if isinstance(depth, int):
        return Transformer(dim = dim, depth = depth, **kwargs)

    return HourglassTransformer(dim = dim, depth = depth, shorten_factor = shorten_factor, attn_resampling = attn_resampling, updown_sample_type = updown_sample_type, **kwargs)

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
        attn_resampling = True,
        updown_sample_type = 'naive',
        heads = 8,
        dim_head = 64,
        causal = False,
        norm_out = False
    ):
        super().__init__()
        assert len(depth) == 3, 'depth should be a tuple of length 3'
        assert updown_sample_type in {'naive', 'linear'}, 'downsample / upsample type must be either naive (average pool and repeat) or linear (linear projection and reshape)'

        pre_layers_depth, valley_depth, post_layers_depth = depth

        if isinstance(shorten_factor, (tuple, list)):
            shorten_factor, *rest_shorten_factor = shorten_factor
        elif isinstance(valley_depth, int):
            shorten_factor, rest_shorten_factor = shorten_factor, None
        else:
            shorten_factor, rest_shorten_factor = shorten_factor, shorten_factor

        transformer_kwargs = dict(
            dim = dim,
            heads = heads,
            dim_head = dim_head
        )

        self.causal = causal
        self.shorten_factor = shorten_factor

        if updown_sample_type == 'naive':
            self.downsample = NaiveDownsample(shorten_factor)
            self.upsample   = NaiveUpsample(shorten_factor)
        elif updown_sample_type == 'linear':
            self.downsample = LinearDownsample(dim, shorten_factor)
            self.upsample   = LinearUpsample(dim, shorten_factor)
        else:
            raise ValueError(f'unknown updown_sample_type keyword value - must be either naive or linear for now')

        self.valley_transformer = get_hourglass_transformer(
            shorten_factor = rest_shorten_factor,
            depth = valley_depth,
            attn_resampling = attn_resampling,
            updown_sample_type = updown_sample_type,
            causal = causal,
            **transformer_kwargs
        )

        self.attn_resampling_pre_valley = Transformer(depth = 1, **transformer_kwargs) if attn_resampling else None
        self.attn_resampling_post_valley = Transformer(depth = 1, **transformer_kwargs) if attn_resampling else None

        self.pre_transformer = Transformer(depth = pre_layers_depth, causal = causal, **transformer_kwargs)
        self.post_transformer = Transformer(depth = post_layers_depth, causal = causal, **transformer_kwargs)
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

        # pre-valley "attention resampling" - they have the pooled token in each bucket attend to the tokens pre-pooled

        if exists(self.attn_resampling_pre_valley):
            if exists(mask):
                attn_resampling_mask = rearrange(padded_mask, 'b (n s) -> (b n) s', s = s)
            else:
                attn_resampling_mask = None

            downsampled = self.attn_resampling_pre_valley(
                rearrange(downsampled, 'b n d -> (b n) () d'),
                rearrange(x, 'b (n s) d -> (b n) s d', s = s),
                mask = attn_resampling_mask
            )

            downsampled = rearrange(downsampled, '(b n) () d -> b n d', b = b)

        # the "valley" - either a regular transformer or another hourglass

        x = self.valley_transformer(downsampled, mask = downsampled_mask)

        valley_out = x.clone()

        # naive repeat upsample

        x = self.upsample(x)

        # add the residual

        x = x + x_residual

        # post-valley "attention resampling"

        if exists(self.attn_resampling_post_valley):
            x = self.attn_resampling_post_valley(
                rearrange(x, 'b (n s) d -> (b n) s d', s = s),
                rearrange(valley_out, 'b n d -> (b n) () d')
            )

            x = rearrange(x, '(b n) s d -> b (n s) d', b = b)

        # bring sequence back to original length, if it were padded for pooling

        x = x[:, :n]

        # post-valley transformers

        x = self.post_transformer(x, mask = mask)
        return self.norm_out(x)

def main():
    # configs
    shard_dir = "/homefs/home/lux70/storage/data/cath/shards/"
    pdb_dir = "/homefs/home/lux70/storage/data/cath/dompdb"
    # shard_dir = "/homefs/home/lux70/storage/data/cath/shards/"
    embedder = "esmfold"
    D = 1024 if embedder == "esmfold" else 320
    n_epochs = 100000
    device = "cuda"
    project = "plaid-hourglass-compression"
    # data
    # dm = CATHShardedDataModule(
    #     storage_type="hdf5",
    #     shard_dir=shard_dir,
    #     embedder=embedder,
    #     seq_len=256,
    #     batch_size=1024
    # )
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

    latent_scaler = LatentScaler(lm_embedder_type=embedder)

    sequence_constructor = LatentToSequence()
    sequence_constructor.to(device)
    seq_loss_fn = SequenceAuxiliaryLoss(sequence_constructor)

    structure_constructor = LatentToStructure()
    structure_constructor.to(device)
    structure_loss_fn = BackboneAuxiliaryLoss(structure_constructor)

    # model
    transformer = get_hourglass_transformer(
        dim = D,                     # feature dimension
        heads = 8,                      # attention heads
        dim_head = 64,                  # dimension per attention head
        shorten_factor = 2,             # shortening factor
        depth = (4, 2, 4),              # tuple of 3, standing for pre-transformer-layers, valley-transformer-layers (after downsample), post-transformer-layers (after upsample) - the valley transformer layers can be yet another nested tuple, in which case it will shorten again recursively
        attn_resampling = True,
        updown_sample_type = "naive",
        causal = True,
        norm_out = True
    )
    transformer = transformer.to(device)

    # train
    import wandb
    wandb.init(
        project=project,
        entity="lu-amy-al1"
    )
    optimizer = AdamW(transformer.parameters(), lr=1e-4)

    def run_batch(train_mode=True):
        prefix = "train" if train_mode else "val"
        x, sequences, gt_structures = batch
        # todo: might be easier to just save seqlen when processing for making masks
        # in this form, the sequences *must* have the correct length (trimmed, no pad)
        tokens, mask, _, _, _ = batch_encode_sequences(sequences)
        if mask.shape[1] != x.shape[1]:
            # pad with False
            mask = trim_or_pad_batch_first(mask, x.shape[1], pad_idx=0)
            tokens = trim_or_pad_batch_first(tokens, x.shape[1], pad_idx=0)

        if embedder != "esmfold":
            x, mask, tokens = x[:, 1:-1, :], mask[:, 1:-1], tokens[:, 1:-1]
        else:
            x = latent_scaler.scale(x)
        mask = mask.bool()
        x, mask = x.to(device), mask.to(device)
        
        output = transformer(x, mask)
        recons_loss = F.mse_loss(x, output)
        wandb.log({f"{prefix}/recons_loss": recons_loss})
    
        scaled_output = latent_scaler.unscale(output)
        seq_loss, seq_loss_dict, recons_strs = seq_loss_fn(scaled_output, tokens, mask, return_reconstructed_sequences=True)
        tbl = {"reconstructed": recons_strs, "original": sequences}
        seq_loss_dict = {f"{prefix}/{k}": v for k, v in seq_loss_dict.items()}
        wandb.log(seq_loss_dict)
        wandb.log({f"{prefix}/recons_strs_tbl": wandb.Table(dataframe=tbl)})

        struct_loss, struct_loss_dict = structure_loss_fn(scaled_output, gt_structures, sequences)
        struct_loss_dict = {f"{prefix}/{k}": v for k, v in struct_loss_dict.items()}
        wandb.log(struct_loss_dict)

        loss = recons_loss

        if train_mode: 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    ######### run ############
    for epoch in trange(n_epochs): 
        for i, batch in enumerate(train_dataloader):
            run_batch(train_mode=True)
        
        for i, batch in enumerate(val_dataloader):
            run_batch(train_mode=False)



if __name__ == "__main__":
    main()