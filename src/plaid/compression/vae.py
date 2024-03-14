import lightning as L
import numpy as np
import torch.nn as nn
import torch
from einops import rearrange, repeat


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f"attn_type {attn_type} unknown"
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv1d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool1d(x, kernel_size=2, stride=2)
        return x


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, l = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) l -> qkv b heads c l", heads=self.heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(out, "b heads c l -> b (heads c) l", heads=self.heads, l=l)
        return self.to_out(out)


class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""

    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv1d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv1d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv1d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv1d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        # temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        # if temb_channels > 0:
        #     self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv1d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        # if temb is not None:
        #     h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch,                         # internal channel
        out_ch,                     # latent channel dim
        ch_mult=(1, 2, 4, 8),       # expansion factor for each resolution 
        num_res_blocks,             # num resblocks at each resolution
        attn_resolutions,           # ?
        dropout=0.0,                # ?
        resamp_with_conv=True,      # ?
        in_channels,                # input channels, e.g. 3 for images, 1024 for ESMFold embed
        initial_seq_len,            # protein analog to image resolution
        z_channels,                 # latent channel dimension
        double_z=True,              # for VAE 
        use_linear_attn=False,
        attn_type="vanilla",
        **ignore_kwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = initial_seq_len
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv1d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = initial_seq_len 
        in_ch_mult = (1,) + tuple(ch_mult)  # why add 1?
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()

        # make resblocks for each resolution
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            for _ in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            # temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            # temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])  # , temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)  # , temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)  # , temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        tanh_out=False,
        use_linear_attn=False,
        attn_type="vanilla",
        **ignorekwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        # self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        # z to block_in
        self.conv_in = torch.nn.Conv1d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            # temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            # temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        # temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        # temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)  # , temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)  # , temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)  # , temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class AutoencoderKL(L.LightningModule):
    def __init__(
        self,
        z_channels,
        embed_dim,
        *args,
        **kwargs
    ):
        super().__init__()
        self.encoder = Encoder(...)
        self.decoder = Decoder(...)
        self.loss = ...
        self.quant_conv = torch.nn.Conv1d(2 * z_channels, 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv1d(embed_dim, z_channels, 1)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

#     def training_step(self, batch, batch_idx, optimizer_idx):
#         inputs = self.get_input(batch, self.image_key)
#         reconstructions, posterior = self(inputs)

#         if optimizer_idx == 0:
#             # train encoder+decoder+logvar
#             aeloss, log_dict_ae = self.loss(
#                 inputs,
#                 reconstructions,
#                 posterior,
#                 optimizer_idx,
#                 self.global_step,
#                 last_layer=self.get_last_layer(),
#                 split="train",
#             )
#             self.log(
#                 "aeloss",
#                 aeloss,
#                 prog_bar=True,
#                 logger=True,
#                 on_step=True,
#                 on_epoch=True,
#             )
#             self.log_dict(
#                 log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False
#             )
#             return aeloss

#         if optimizer_idx == 1:
#             # train the discriminator
#             discloss, log_dict_disc = self.loss(
#                 inputs,
#                 reconstructions,
#                 posterior,
#                 optimizer_idx,
#                 self.global_step,
#                 last_layer=self.get_last_layer(),
#                 split="train",
#             )

#             self.log(
#                 "discloss",
#                 discloss,
#                 prog_bar=True,
#                 logger=True,
#                 on_step=True,
#                 on_epoch=True,
#             )
#             self.log_dict(
#                 log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False
#             )
#             return discloss

#     def validation_step(self, batch, batch_idx):
#         inputs = self.get_input(batch, self.image_key)
#         reconstructions, posterior = self(inputs)
#         aeloss, log_dict_ae = self.loss(
#             inputs,
#             reconstructions,
#             posterior,
#             0,
#             self.global_step,
#             last_layer=self.get_last_layer(),
#             split="val",
#         )

#         discloss, log_dict_disc = self.loss(
#             inputs,
#             reconstructions,
#             posterior,
#             1,
#             self.global_step,
#             last_layer=self.get_last_layer(),
#             split="val",
#         )

#         self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
#         self.log_dict(log_dict_ae)
#         self.log_dict(log_dict_disc)
#         return self.log_dict

#     def configure_optimizers(self):
#         lr = self.learning_rate
#         opt_ae = torch.optim.Adam(
#             list(self.encoder.parameters())
#             + list(self.decoder.parameters())
#             + list(self.quant_conv.parameters())
#             + list(self.post_quant_conv.parameters()),
#             lr=lr,
#             betas=(0.5, 0.9),
#         )
#         opt_disc = torch.optim.Adam(
#             self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9)
#         )
#         return [opt_ae, opt_disc], []

#     def get_last_layer(self):
#         return self.decoder.conv_out.weight

#     @torch.no_grad()
#     def log_images(self, batch, only_inputs=False, **kwargs):
#         log = dict()
#         x = self.get_input(batch, self.image_key)
#         x = x.to(self.device)
#         if not only_inputs:
#             xrec, posterior = self(x)
#             if x.shape[1] > 3:
#                 # colorize with random projection
#                 assert xrec.shape[1] > 3
#                 x = self.to_rgb(x)
#                 xrec = self.to_rgb(xrec)
#             log["samples"] = self.decode(torch.randn_like(posterior.sample()))
#             log["reconstructions"] = xrec
#         log["inputs"] = x
#         return log

#     def to_rgb(self, x):
#         assert self.image_key == "segmentation"
#         if not hasattr(self, "colorize"):
#             self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
#         x = F.Conv1d(x, weight=self.colorize)
#         x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
#         return x
