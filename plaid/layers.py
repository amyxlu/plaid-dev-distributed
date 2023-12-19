import math

from dctorch import functional as df
from einops import rearrange, repeat
import torch
from torch import nn
from torch.nn import functional as F

# Karras et al. preconditioned denoiser



# class DenoiserWithVariance(Denoiser):
#     def loss(self, input, noise, sigma, return_model_output=False, **kwargs):
#         assert self.loss_distance == "mse"
#         c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
#         noised_input = input + noise * utils.append_dims(sigma, input.ndim)
#         model_output, logvar = self.inner_model(noised_input * c_in, sigma, return_variance=True, **kwargs)
#         logvar = utils.append_dims(logvar, model_output.ndim)
#         target = (input - c_skip * noised_input) / c_out
#         losses = ((model_output - target) ** 2 / logvar.exp() + logvar) / 2
#         loss = losses.flatten(1).mean(1)
#         if return_model_output:
#             return loss, model_output
#         else:
#             return loss


# class SimpleLossDenoiser(Denoiser):
#     """L_simple with the Karras et al. preconditioner."""

#     def loss(self, input, noise, sigma, return_model_output=False, **kwargs):
#         noised_input = input + noise * utils.append_dims(sigma, input.ndim)
#         denoised = self(noised_input, sigma, **kwargs)
#         eps = continuous.to_d(noised_input, sigma, denoised)
#         if self.loss_distance == "mse":
#             loss = (eps - noise).pow(2).flatten(1).mean(1)
#         elif self.loss_distance == "huber":
#             loss = F.huber_loss(eps, noise, reduction="mean")
#         else:
#             raise ValueError(f"Unknown loss type {self.loss_distance}")
        
#         if return_model_output:
#             return loss, denoised
#         else:
#             return loss
           
            
# class SimpleVanilla(Denoiser):
#     """ Vanilla L_simple without the Karras et al. preconditioner."""
#     def loss(self, input, noise, sigma, return_model_output=False, **kwargs):
#         noised_input = input + noise * utils.append_dims(sigma, input.ndim)
#         model_output = self.inner_model(noised_input, sigma, **kwargs)
#         if self.loss_distance == "mse":
#             loss = (model_output - noise).pow(2).flatten(1).mean(1)
#         elif self.loss_distance == "huber":
#             loss = F.huber_loss(model_output, noise, reduction="mean")
#         else:
#             raise ValueError(f"Unknown loss type {self.loss_distance}")
            
#         if return_model_output:
#             return loss, model_output
#         else:
#             return loss

#     def forward(self, input, sigma, **kwargs):
#         return self.inner_model(input, sigma, **kwargs)


# class DiscreteDenoiser:
#     def __init__(self, model_config, inner_model, **kwargs):
#         self.model_config = model_config
#         self.sd_config = model_config.sigma_sample_density
#         self.diffusion = get_default_diffusion(self.sd_config.type, self.sd_config.T, **kwargs)
#         self.inner_model = inner_model
    
#     def loss_epsilon(self, x, ts, noise = None, return_model_output=False, **model_kwargs):
#         if noise is None:
#             noise = torch.randn_like(x) * self.sd_config.noise_scale

#         noised_input = self.diffusion.q_sample(x, ts, noise)
#         epsilon = self.inner_model(noised_input, ts, **model_kwargs)
#         if self.model_config.loss_distance == "mse":
#             loss = (epsilon - noise).pow(2).flatten(1).mean(1)
#         elif self.model_config.loss_distance == "huber":
#             loss = F.huber_loss(epsilon, noise, reduction="mean")
#         else:
#             raise ValueError(f"Unknown loss type {self.loss_distance}")
#         if return_model_output:
#             return loss, epsilon
#         else:
#             return loss
    

# class ResidualBlock(nn.Module):
#     def __init__(self, *main, skip=None):
#         super().__init__()
#         self.main = nn.Sequential(*main)
#         self.skip = skip if skip else nn.Identity()

#     def forward(self, input):
#         return self.main(input) + self.skip(input)


# # Noise level (and other) conditioning

# class ConditionedModule(nn.Module):
#     pass


# class UnconditionedModule(ConditionedModule):
#     def __init__(self, module):
#         super().__init__()
#         self.module = module

#     def forward(self, input, cond=None):
#         return self.module(input)


# class ConditionedSequential(nn.Sequential, ConditionedModule):
#     def forward(self, input, cond):
#         for module in self:
#             if isinstance(module, ConditionedModule):
#                 input = module(input, cond)
#             else:
#                 input = module(input)
#         return input


# class ConditionedResidualBlock(ConditionedModule):
#     def __init__(self, *main, skip=None):
#         super().__init__()
#         self.main = ConditionedSequential(*main)
#         self.skip = skip if skip else nn.Identity()

#     def forward(self, input, cond):
#         skip = self.skip(input, cond) if isinstance(self.skip, ConditionedModule) else self.skip(input)
#         return self.main(input, cond) + skip


# class AdaGN(ConditionedModule):
#     def __init__(self, feats_in, c_out, num_groups, eps=1e-5, cond_key='cond'):
#         super().__init__()
#         self.num_groups = num_groups
#         self.eps = eps
#         self.cond_key = cond_key
#         self.mapper = nn.Linear(feats_in, c_out * 2)
#         nn.init.zeros_(self.mapper.weight)
#         nn.init.zeros_(self.mapper.bias)

#     def forward(self, input, cond):
#         weight, bias = self.mapper(cond[self.cond_key]).chunk(2, dim=-1)
#         input = F.group_norm(input, self.num_groups, eps=self.eps)
#         return torch.addcmul(utils.append_dims(bias, input.ndim), input, utils.append_dims(weight, input.ndim) + 1)


# # Attention


# class SelfAttention2d(ConditionedModule):
#     def __init__(self, c_in, n_head, norm, dropout_rate=0.):
#         super().__init__()
#         assert c_in % n_head == 0
#         self.norm_in = norm(c_in)
#         self.n_head = n_head
#         self.qkv_proj = nn.Conv2d(c_in, c_in * 3, 1)
#         self.out_proj = nn.Conv2d(c_in, c_in, 1)
#         self.dropout = nn.Dropout(dropout_rate)
#         nn.init.zeros_(self.out_proj.weight)
#         nn.init.zeros_(self.out_proj.bias)

#     def forward(self, input, cond):
#         n, c, h, w = input.shape
#         qkv = self.qkv_proj(self.norm_in(input, cond))
#         qkv = qkv.view([n, self.n_head * 3, c // self.n_head, h * w]).transpose(2, 3)
#         q, k, v = qkv.chunk(3, dim=1)
#         y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p)
#         y = y.transpose(2, 3).contiguous().view([n, c, h, w])
#         return input + self.out_proj(y)


# class CrossAttention2d(ConditionedModule):
#     def __init__(self, c_dec, c_enc, n_head, norm_dec, dropout_rate=0.,
#                  cond_key='cross', cond_key_padding='cross_padding'):
#         super().__init__()
#         assert c_dec % n_head == 0
#         self.cond_key = cond_key
#         self.cond_key_padding = cond_key_padding
#         self.norm_enc = nn.LayerNorm(c_enc)
#         self.norm_dec = norm_dec(c_dec)
#         self.n_head = n_head
#         self.q_proj = nn.Conv2d(c_dec, c_dec, 1)
#         self.kv_proj = nn.Linear(c_enc, c_dec * 2)
#         self.out_proj = nn.Conv2d(c_dec, c_dec, 1)
#         self.dropout = nn.Dropout(dropout_rate)
#         nn.init.zeros_(self.out_proj.weight)
#         nn.init.zeros_(self.out_proj.bias)

#     def forward(self, input, cond):
#         n, c, h, w = input.shape
#         q = self.q_proj(self.norm_dec(input, cond))
#         q = q.view([n, self.n_head, c // self.n_head, h * w]).transpose(2, 3)
#         kv = self.kv_proj(self.norm_enc(cond[self.cond_key]))
#         kv = kv.view([n, -1, self.n_head * 2, c // self.n_head]).transpose(1, 2)
#         k, v = kv.chunk(2, dim=1)
#         attn_mask = (cond[self.cond_key_padding][:, None, None, :]) * -10000
#         y = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p=self.dropout.p)
#         y = y.transpose(2, 3).contiguous().view([n, c, h, w])
#         return input + self.out_proj(y)


# # Downsampling/upsampling

# _kernels = {
#     'linear':
#         [1 / 8, 3 / 8, 3 / 8, 1 / 8],
#     'cubic': 
#         [-0.01171875, -0.03515625, 0.11328125, 0.43359375,
#         0.43359375, 0.11328125, -0.03515625, -0.01171875],
#     'lanczos3': 
#         [0.003689131001010537, 0.015056144446134567, -0.03399861603975296,
#         -0.066637322306633, 0.13550527393817902, 0.44638532400131226,
#         0.44638532400131226, 0.13550527393817902, -0.066637322306633,
#         -0.03399861603975296, 0.015056144446134567, 0.003689131001010537]
# }
# _kernels['bilinear'] = _kernels['linear']
# _kernels['bicubic'] = _kernels['cubic']


# class Downsample2d(nn.Module):
#     def __init__(self, kernel='linear', pad_mode='reflect'):
#         super().__init__()
#         self.pad_mode = pad_mode
#         kernel_1d = torch.tensor([_kernels[kernel]])
#         self.pad = kernel_1d.shape[1] // 2 - 1
#         self.register_buffer('kernel', kernel_1d.T @ kernel_1d)

#     def forward(self, x):
#         x = F.pad(x, (self.pad,) * 4, self.pad_mode)
#         weight = x.new_zeros([x.shape[1], x.shape[1], self.kernel.shape[0], self.kernel.shape[1]])
#         indices = torch.arange(x.shape[1], device=x.device)
#         weight[indices, indices] = self.kernel.to(weight)
#         return F.conv2d(x, weight, stride=2)


# class Upsample2d(nn.Module):
#     def __init__(self, kernel='linear', pad_mode='reflect'):
#         super().__init__()
#         self.pad_mode = pad_mode
#         kernel_1d = torch.tensor([_kernels[kernel]]) * 2
#         self.pad = kernel_1d.shape[1] // 2 - 1
#         self.register_buffer('kernel', kernel_1d.T @ kernel_1d)

#     def forward(self, x):
#         x = F.pad(x, ((self.pad + 1) // 2,) * 4, self.pad_mode)
#         weight = x.new_zeros([x.shape[1], x.shape[1], self.kernel.shape[0], self.kernel.shape[1]])
#         indices = torch.arange(x.shape[1], device=x.device)
#         weight[indices, indices] = self.kernel.to(weight)
#         return F.conv_transpose2d(x, weight, stride=2, padding=self.pad * 2 + 1)


# # Embeddings

# class FourierFeatures(nn.Module):
#     def __init__(self, in_features, out_features, std=1.):
#         super().__init__()
#         assert out_features % 2 == 0
#         self.register_buffer('weight', torch.randn([out_features // 2, in_features]) * std)

#     def forward(self, input):
#         f = 2 * math.pi * input @ self.weight.T
#         return torch.cat([f.cos(), f.sin()], dim=-1)


# # U-Nets

# class UNet(ConditionedModule):
#     def __init__(self, d_blocks, u_blocks, skip_stages=0):
#         super().__init__()
#         self.d_blocks = nn.ModuleList(d_blocks)
#         self.u_blocks = nn.ModuleList(u_blocks)
#         self.skip_stages = skip_stages

#     def forward(self, input, cond):
#         skips = []
#         for block in self.d_blocks[self.skip_stages:]:
#             input = block(input, cond)
#             skips.append(input)
#         for i, (block, skip) in enumerate(zip(self.u_blocks, reversed(skips))):
#             input = block(input, cond, skip if i > 0 else None)
#         return input


class FullyConnectedNetwork(nn.Module):
    def __init__(
        self,
        n_classes: int,
        mlp_hidden_dim: int = 1024,
        mlp_num_layers: int = 5,
        mlp_dropout_p: float = 0.1,
        add_sigmoid: bool = False,
    ):
        super().__init__()
        if mlp_num_layers == 1:
            layers = [nn.Linear(mlp_hidden_dim, n_classes)]

        elif mlp_num_layers == 2:
            first_layer = [
                nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(p=mlp_dropout_p),
            ]
            final_layer = [
                nn.Linear(mlp_hidden_dim // 4, n_classes),
            ]
            layers = first_layer + final_layer

        else:
            assert mlp_num_layers >= 3
            num_hidden_layers = mlp_num_layers - 3

            first_layer = [
                nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(p=mlp_dropout_p),
            ]

            second_layer = [
                nn.Linear(mlp_hidden_dim // 2, mlp_hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(p=mlp_dropout_p),
            ]

            hidden_layer = [
                nn.Linear(mlp_hidden_dim // 4, mlp_hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(p=mlp_dropout_p),
            ]

            final_layer = [
                nn.Linear(mlp_hidden_dim // 4, n_classes),
            ]

            layers = (
                first_layer
                + second_layer
                + hidden_layer * num_hidden_layers
                + final_layer
            )

        if add_sigmoid:
            layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)