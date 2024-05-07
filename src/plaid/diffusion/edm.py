from functools import lru_cache, reduce
import logging
from copy import deepcopy
import time
import typing as T
import random

from dctorch import functional as df
import torch

from plaid.compression.uncompress import UncompressContinuousLatent
from plaid.utils import LatentScaler, get_lr_scheduler
from plaid.ema import PostHocEMA, inplace_copy 

from plaid.losses.modules import SequenceAuxiliaryLoss, BackboneAuxiliaryLoss
from plaid.proteins import LatentToSequence, LatentToStructure
from plaid.losses.functions import masked_mse_loss, masked_huber_loss
# Helper functions


def cast_dtype(tensor, parameter):
    return tensor.to(dtype=parameter.dtype)


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]


def dct(x):
    if x.ndim == 3:
        return df.dct(x)
    if x.ndim == 4:
        return df.dct2(x)
    if x.ndim == 5:
        return df.dct3(x)
    raise ValueError(f'Unsupported dimensionality {x.ndim}')


@lru_cache
def freq_weight_1d(n, scales=0, dtype=None, device=None):
    ramp = torch.linspace(0.5 / n, 0.5, n, dtype=dtype, device=device)
    weights = -torch.log2(ramp)
    if scales >= 1:
        weights = torch.clamp_max(weights, scales)
    return weights


@lru_cache
def freq_weight_nd(shape, scales=0, dtype=None, device=None):
    indexers = [[slice(None) if i == j else None for j in range(len(shape))] for i in range(len(shape))]
    weights = [freq_weight_1d(n, scales, dtype, device)[ix] for n, ix in zip(shape, indexers)]
    return reduce(torch.minimum, weights)


def ema_update_dict(values, updates, decay):
    for k, v in updates.items():
        if k not in values:
            values[k] = v
        else:
            values[k] *= decay
            values[k] += (1 - decay) * v
    return values

# Karras et al. preconditioned denoiser

import lightning as L


class ElucidatedDiffusion(L.LightningModule):
    """A Karras et al. preconditioner for denoising diffusion models."""

    def __init__(
        self,
        denoiser,
        sigma_density_generator,
        sigma_data=1.,
        loss_weighting='soft-min-snr',
        # post hoc EMA
        ema_sigma_rels = (0.05, 0.3),           # a tuple with the hyperparameter for the multiple EMAs. you need at least 2 here to synthesize a new one
        ema_gammas = None,
        ema_update_every = 10,                  # how often to actually update, to save on compute (updates every 10th .update() call). -1 disables EMA
        ema_checkpoint_every_num_steps = 1000,
        ema_checkpoint_folder = './post-hoc-ema-checkpoints',   # the folder of saved checkpoints for each sigma_rel (gamma) across timesteps with the hparam above, used to synthesizing a new EMA model after training
        # compression and architecture
        shorten_factor=1.0,
        unscaler: LatentScaler = LatentScaler("identity"),
        uncompressor: T.Optional[UncompressContinuousLatent] = None,
        # optimization,
        gradient_accumulation_steps: int = 1,
        gradient_clip_val: float = 0.5,
        lr=1e-4,
        adam_betas=(0.9, 0.999),
        lr_sched_type: str = "constant",
        lr_num_warmup_steps: int = 0,
        lr_num_training_steps: int = 10_000_000,
        lr_num_cycles: int = 1,
    ):
        super().__init__()

        self.sigma_density_generator = sigma_density_generator
        self.shorten_factor = shorten_factor
        self.unscaler = unscaler
        self.uncompressor = uncompressor

        # self conditioning projection layers is configured within the denoiser class
        # we need to know if we should randomly generate self-conditioning or not
        self.use_self_conditioning = denoiser.use_self_conditioning

        # learning rates and optimization
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip_val = gradient_clip_val
        self.lr = lr
        self.adam_betas = adam_betas
        self.lr_sched_type = lr_sched_type
        self.lr_num_warmup_steps = lr_num_warmup_steps
        self.lr_num_training_steps = lr_num_training_steps
        self.lr_num_cycles = lr_num_cycles

        # turn off lightning's automatic optimization to handle EMA after optimizer
        self.automatic_optimization = False

        if ema_update_every <= 0: 
            self.ema_wrapper = None
        else:
            self.ema_wrapper = PostHocEMA(
                denoiser,
                sigma_rels = ema_sigma_rels,                # a tuple with the hyperparameter for the multiple EMAs. you need at least 2 here to synthesize a new one
                gammas = ema_gammas,
                update_every = ema_update_every,            # how often to actually update, to save on compute (updates every 10th .update() call)
                checkpoint_every_num_steps = ema_checkpoint_every_num_steps,
                checkpoint_folder = ema_checkpoint_folder,  # the folder of saved checkpoints for each sigma_rel (gamma) across timesteps with the hparam above, used to synthesizing a new EMA model after training
            )

        # denoising and loss settings
        self.denoiser = denoiser
        self.sigma_data = sigma_data
        if callable(loss_weighting):
            self.weighting = loss_weighting
        if loss_weighting == 'karras':
            self.weighting = torch.ones_like
        elif loss_weighting == 'soft-min-snr':
            self.weighting = self._weighting_soft_min_snr
        elif loss_weighting == 'snr':
            self.weighting = self._weighting_snr
        else:
            raise ValueError(f'Unknown weighting type {loss_weighting}')
        
        self.save_hyperparameters()

    """Loss scalings and preconditioning"""
    def _weighting_soft_min_snr(self, sigma):
        return (sigma * self.sigma_data) ** 2 / (sigma ** 2 + self.sigma_data ** 2) ** 2

    def _weighting_snr(self, sigma):
        return self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)

    def get_scalings(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def loss_preconditioned(self, x, sigma, label=None, mask=None, x_self_cond=None, **kwargs):
        c_skip, c_out, c_in = [append_dims(s, x.ndim) for s in self.get_scalings(sigma)]
        c_weight = self.weighting(sigma)
        noise = torch.randn_like(x)
        noised_input = x + noise * append_dims(sigma, x.ndim)
        model_output = self.denoiser(
            x=noised_input * c_in, 
            sigma=sigma,
            y=label,
            mask=mask,
            x_self_cond=x_self_cond
        )
        target = (x - c_skip * noised_input) / c_out 
        losses = (masked_mse_loss(model_output, target, mask) * c_weight)
        return losses.mean()

    def forward(self, x, sigma, label=None, mask=None, x_self_cond=None, **kwargs):
        c_skip, c_out, c_in = [append_dims(s, x.ndim) for s in self.get_scalings(sigma)]
        model_output = self.denoiser(
            x=x * c_in,
            sigma=sigma,
            y=label,
            mask=mask,
            x_self_cond=x_self_cond
        )
        return model_output * c_out + x * c_skip
    
    """Lightning set up"""
    def training_step(self, batch, batch_idx):
        start = time.time()
        x, mask, clan = batch
        clan = clan.long().squeeze()   # (N,)
        mask = mask.bool()
        sigma = self.sigma_density_generator((x.shape[0], 1))  # (N, 1)
        sigma = sigma.to(x.device)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        x_self_cond = None
        if self.use_self_conditioning and random.random() < 0.5:
            with torch.no_grad():
                x_self_cond = self(x=x, sigma=sigma, label=clan, mask=mask, x_self_cond=x_self_cond)
                x_self_cond.detach_()
                x_self_cond = cast_dtype(x_self_cond, next(self.parameters()))

        # TODO: potentially add other loss terms; currently this would also require reprocessing the dataset
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        loss = self.loss(
            x=x,
            sigma=sigma,
            label=clan,
            mask=mask,
            x_self_cond=x_self_cond
        )
        self.log("train/diffusion_loss", loss, batch_size=x.shape[0], on_step=True, on_epoch=True)

        # manual optimization
        optimizer.zero_grad()
        self.manual_backward(loss)
        self.clip_gradients(optimizer, self.gradient_clip_val, gradient_clip_algorithm="norm")

        # accumulate gradients of N batches
        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
            optimizer.step()
        
        # update lr every step
        scheduler.step()

        # EMA updates
        self.ema_wrapper.update()

        end = time.time()
        self.log("train/batch_runtime", (end - start) / 60, on_step=False, on_epoch=True)

        return loss
    
    def validation_step(self, batch):
        start = time.time()
        x, mask, clan = batch
        sigma = self.sigma_density_generator((x.shape[0], 1))  # (N, 1)
        sigma = sigma.to(x.device)
        x, sigma = tuple(map(lambda tensor: cast_dtype(tensor, next(self.parameters())), (x, sigma)))
        mask = mask.bool()
        clan = clan.long().squeeze()   # (N,)

        loss = self.loss(
            x=x,
            label=clan,
            sigma=sigma,
            mask=mask,
            x_self_cond=None
        )
        end = time.time()
        self.log("val/diffusion_loss", loss, batch_size=x.shape[0], on_step=False, on_epoch=True)
        self.log("val/batch_runtime", (end - start) / 60, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, betas=self.adam_betas
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