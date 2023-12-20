"""
Roughly follows the iDDPM formulation with min-SNR weighting, etc.
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/guided_diffusion.py
"""

import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
from torchvision import transforms as T, utils

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm

# from ema_pytorch import EMA
import lightning as L

from plaid.denoisers import BaseDenoiser
from plaid.utils import LatentScaler
from plaid.diffusion.beta_schedulers import BetaScheduler, SigmoidBetaScheduler


ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model: BaseDenoiser,
        latent_scaler: LatentScaler = LatentScaler(),
        beta_scheduler: BetaScheduler = SigmoidBetaScheduler(),
        *,
        timesteps=1000,
        objective="pred_noise",
        min_snr_loss_weight=False,
        min_snr_gamma=5,
        ddim_sampling_eta=0.0,  # 0 is DDIM and 1 is DDPM
        sampling_timesteps=None,
        sampling_seq_len=64,
    ):
        super().__init__()
        self.model = model
        self.latent_scaler = latent_scaler
        self.self_condition = self.model.use_self_conditioning
        self.objective = objective
        assert objective in {
            "pred_noise",
            "pred_x0",
            "pred_v",
        }, "objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])"

        betas = beta_scheduler(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps
        )  # default num sampling timesteps to number of timesteps at training
        assert self.sampling_timesteps <= timesteps
        self.sampling_seq_len = sampling_seq_len
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer to bfloat16
        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.bfloat16)
        )
        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        register_buffer("posterior_variance", posterior_variance)
        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # loss weight
        snr = alphas_cumprod / (1 - alphas_cumprod)
        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        if objective == "pred_noise":
            loss_weight = maybe_clipped_snr / snr
        elif objective == "pred_x0":
            loss_weight = maybe_clipped_snr
        elif objective == "pred_v":
            loss_weight = maybe_clipped_snr / (snr + 1)
        register_buffer("loss_weight", loss_weight)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, model_kwargs={}, clip_x_start=False):
        model_output = self.model(x, t, **model_kwargs)
        maybe_clip = (
            partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        )
        # TODO: add dynamic thresholding from Imagen

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == "pred_x0":
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(
        self, x, t, model_kwargs={}, clip_denoised=True
    ):
        preds = self.model_predictions(x, t, **model_kwargs)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def condition_mean(self, cond_fn, mean, variance, x, t, guidance_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.
        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, t, **guidance_kwargs)
        new_mean = mean + variance * gradient
        print("gradient: ", (variance * gradient).mean())
        return new_mean

    @torch.no_grad()
    def p_sample(self, x, t: int, model_kwargs={}, cond_fn=None, guidance_kwargs=None, clip_denoised=True):
        B, L, C = x.shape
        batched_times = torch.full((B,), t, device=x.device, dtype=torch.long)
        model_mean, variance, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, model_kwargs=model_kwargs, clip_denoised=clip_denoised
        )
        if exists(cond_fn) and exists(guidance_kwargs):
            model_mean = self.condition_mean(
                cond_fn, model_mean, variance, x, batched_times, guidance_kwargs
            )

        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        return_all_timesteps=False,
        model_kwargs={},
        cond_fn=None,
        guidance_kwargs=None,
        clip_denoised=True
    ):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, model_kwargs, cond_fn, guidance_kwargs, clip_denoised)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        ret = self.latent_scaler.unscale(ret)
        return ret

    @torch.no_grad()
    def ddim_sample(
        self,
        shape,
        return_all_timesteps=False,
        model_kwargs={},
        cond_fn=None,
        guidance_kwargs=None,
        clip_denoised=True,
    ):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = (
            shape[0],
            self.betas.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
            self.objective,
        )

        times = torch.linspace(
            -1, total_timesteps - 1, steps=sampling_timesteps + 1
        )  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(
                img, time_cond, model_kwargs=model_kwargs, clip_x_start=True
            )

            imgs.append(img)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        ret = self.latent_scaler.unscale(ret)
        return ret

    @torch.no_grad()
    def sample(
        self,
        batch_size=16,
        return_all_timesteps=False,
        model_kwargs={},
        cond_fn=None,
        guidance_kwargs=None,
        clip_denoised=True
    ):
        sample_fn = (
            self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        )
        shape = (batch_size, self.sampling_seq_len, self.model.hid_dim)
        return sample_fn(
            shape,
            return_all_timesteps=return_all_timesteps,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            guidance_kwargs=guidance_kwargs,
            clip_denoised=clip_denoised
        )

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device=device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(
            reversed(range(0, t)), desc="interpolation sample time step", total=t
        ):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast(enabled=False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, model_kwargs={}, noise=None):
        B, L, C = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        model_kwargs["x_self_cond"] = x_self_cond

        # predict and take gradient step
        model_out = self.model(x, t, **model_kwargs)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        elif self.objective == "pred_v":
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f"unknown objective {self.objective}")

        loss = F.mse_loss(model_out, target, reduction="none")
        loss = reduce(loss, "b ... -> b", "mean")

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()
    
    def forward(self, x_unnormalized, mask, model_kwargs={}, noise=None):
        x = self.latent_scaler.scale(x_unnormalized)
        t = torch.randint(0, self.num_timesteps, (x.shape[0],)).long().to(x.device)
        model_kwargs['mask'] = mask
        return self.p_losses(x, t, model_kwargs, noise)


class GaussianDiffusionLightningModule(L.LightningModule):
    def __init__(
        self,
        model: BaseDenoiser,
        latent_scaler: LatentScaler = LatentScaler(),
        beta_scheduler: BetaScheduler = SigmoidBetaScheduler(),
        timesteps=1000,
        objective="pred_noise",
        min_snr_loss_weight=False,
        min_snr_gamma=5,
        ddim_sampling_eta=0.0,  # 0 is DDIM and 1 is DDPM
        sampling_timesteps=None,
        sampling_seq_len=64,
        # optimization
        lr = 1e-4,
        adam_betas = (0.9, 0.999),
    ):
        super().__init__()
        self.diffusion = GaussianDiffusion(
            model=model,
            latent_scaler=latent_scaler,
            beta_scheduler=beta_scheduler,
            timesteps=timesteps,
            objective=objective,
            min_snr_loss_weight=min_snr_loss_weight,
            min_snr_gamma=min_snr_gamma,
            ddim_sampling_eta=ddim_sampling_eta,
            sampling_timesteps=sampling_timesteps,
            sampling_seq_len=sampling_seq_len
        )
        self.lr = lr
        self.adam_betas = adam_betas

    def compute_loss(self, batch):
        # batch: embedding, mask
        return self.diffusion(*batch)

    def training_step(self, batch):
        loss = self.compute_loss(batch)
        self.log_dict({'train_loss' : loss}, logger = True, on_step = True, sync_dist = True)
        return loss

    def validation_step(self, batch):
        # Extract the starting images from data batch
        loss = self.compute_loss(batch)
        self.log_dict({'val_loss' : loss}, logger = True, on_step = True, sync_dist = True)
        return loss 

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=self.adam_betas)
        return optimizer

#     # def configure_optimizers(self) -> None:
#     #     optim = torch.optim.AdamW(self.parameters(), betas=self.adam_betas, lr=self.lr)
#     #     scheduler = TanhLRScheduler(optimizer, ...)
#     #     return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

#     # def lr_scheduler_step(self, scheduler, metric):
#     #     scheduler.step()


if __name__ == "__main__":
    from plaid.denoisers import UTriSelfAttnDenoiser
    denoiser = UTriSelfAttnDenoiser(num_blocks=7, hid_dim=1024)
    diffusion = GaussianDiffusion(denoiser)
    N, L, C = 4, 64, 1024
    x_start = torch.randn(N, L, C)
    t = torch.randint(0, diffusion.num_timesteps, (N,)).long()
    loss = diffusion.p_losses(x_start, t)
    # sampled = diffusion.sample()
    print(loss)
    # print(sampled)



    
    from plaid.datasets import CATHShardedDataModule

    torch.set_default_dtype(torch.bfloat16)
    device = torch.device("cuda:0")

    model = UTriSelfAttnDenoiser(num_blocks=7, hid_dim=1024)
    model.to(device)
    dm = CATHShardedDataModule()
    dm.setup("fit")
    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))

    module = GaussianDiffusionLightningModule(denoiser)
    loss = module.training_step(batch)
    # x, seqlens = batch