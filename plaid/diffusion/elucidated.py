"""
From:
Karras et al.: https://arxiv.org/pdf/2206.00364.pdf
DPM++ implementation by Kathryn Crowson: https://arxiv.org/abs/2211.01095
Abstractions by Phil Wang: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/elucidated_diffusion.py
"""

from math import sqrt
from random import random
import torch
from torch import nn, einsum
import torch.nn.functional as F

from tqdm import tqdm
from einops import rearrange, repeat, reduce

import lightning as L

from plaid.denoisers import BaseDenoiser
from plaid.utils import LatentScaler, get_lr_scheduler, sequences_to_secondary_structure_fracs
from plaid.diffusion.beta_schedulers import BetaScheduler, SigmoidBetaScheduler


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))


class ElucidatedDiffusion(L.LightningModule):
    def __init__(
        self,
        denoiser: BaseDenoiser,
        latent_scaler: LatentScaler = LatentScaler(),
        beta_scheduler: BetaScheduler = SigmoidBetaScheduler(),
        # sampling_seq_len=64,
        num_sample_steps = 32, # number of sampling steps
        sigma_min = 0.002,     # min noise level
        sigma_max = 80,        # max noise level
        sigma_data = 0.5,      # standard deviation of data distribution
        rho = 7,               # controls the sampling schedule
        P_mean = -1.2,         # mean of log-normal distribution from which noise is drawn for training
        P_std = 1.2,           # standard deviation of log-normal distribution from which noise is drawn for training
        S_churn = 80,          # parameters for stochastic sampling - depends on dataset, Table 5 in apper
        S_tmin = 0.05,
        S_tmax = 50,
        S_noise = 1.003,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.self_condition = self.denoiser.use_self_conditioning

        # parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.rho = rho

        self.P_mean = P_mean
        self.P_std = P_std

        self.num_sample_steps = num_sample_steps  # otherwise known as N in the paper

        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

    @property
    def device(self):
        return next(self.net.parameters()).device

    # derived preconditioning params - Table 1

    def c_skip(self, sigma):
        return (self.sigma_data ** 2) / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma):
        return 1 * (sigma ** 2 + self.sigma_data ** 2) ** -0.5

    def c_noise(self, sigma):
        return log(sigma) * 0.25

    # preconditioned network output
    # equation (7) in the paper

    def preconditioned_network_forward(self, x_noised, sigma, self_cond = None, clamp = False):
        N, L, _ = x_noised.shape
        device = x_noised.device

        if isinstance(sigma, float):
            sigma = torch.full((N,), sigma, device = device)

        padded_sigma = rearrange(sigma, 'b -> b 1 1')
        net_out = self.net(
            self.c_in(padded_sigma) * x_noised,
            self.c_noise(sigma),
            self_cond
        )
        out = self.c_skip(padded_sigma) * x_noised + self.c_out(padded_sigma) * net_out

        if clamp:
            out = out.clamp(-1., 1.)

        return out

    # sampling

    # sample schedule
    # equation (5) in the paper

    def sample_schedule(self, num_sample_steps = None):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        N = num_sample_steps
        inv_rho = 1 / self.rho

        steps = torch.arange(num_sample_steps, device = self.device)
        sigmas = (self.sigma_max ** inv_rho + steps / (N - 1) * (self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho

        sigmas = F.pad(sigmas, (0, 1), value = 0.) # last step is sigma value of 0.
        return sigmas

    @torch.no_grad()
    def sample(self, shape, num_sample_steps = None, clamp = True):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        sigmas = self.sample_schedule(num_sample_steps)

        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, sqrt(2) - 1),
            0.
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # init noise
        init_sigma = sigmas[0]
        x = init_sigma * torch.randn(shape, device = self.device)

        # for self conditioning
        x_start = None

        # gradually denoise
        for sigma, sigma_next, gamma in tqdm(sigmas_and_gammas, desc = 'sampling time step'):
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            eps = self.S_noise * torch.randn(shape, device = self.device) # stochastic sampling

            sigma_hat = sigma + gamma * sigma
            x_hat = x + sqrt(sigma_hat ** 2 - sigma ** 2) * eps

            self_cond = x_start if self.self_condition else None

            model_output = self.preconditioned_network_forward(x_hat, sigma_hat, self_cond, clamp = clamp)
            denoised_over_sigma = (x_hat - model_output) / sigma_hat

            x_next = x_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            # second order correction, if not the last timestep

            if sigma_next != 0:
                self_cond = model_output if self.self_condition else None

                model_output_next = self.preconditioned_network_forward(x_next, sigma_next, self_cond, clamp = clamp)
                denoised_prime_over_sigma = (x_next - model_output_next) / sigma_next
                x_next = x_hat + 0.5 * (sigma_next - sigma_hat) * (denoised_over_sigma + denoised_prime_over_sigma)

            x = x_next
            x_start = model_output_next if sigma_next != 0 else model_output

        if clamp:
            x = x.clamp(-1., 1.)
        
        return self.latent_scaler.unscale(x) 

    @torch.no_grad()
    def sample_using_dpmpp(self, batch_size = 16, num_sample_steps = None, clamp=False):
        """
        thanks to Katherine Crowson (https://github.com/crowsonkb) for figuring it all out!
        https://arxiv.org/abs/2211.01095
        """

        device, num_sample_steps = self.device, default(num_sample_steps, self.num_sample_steps)

        sigmas = self.sample_schedule(num_sample_steps)

        shape = (batch_size, self.channels, self.image_size, self.image_size)
        x  = sigmas[0] * torch.randn(shape, device = device)

        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()

        old_denoised = None
        for i in tqdm(range(len(sigmas) - 1)):
            denoised = self.preconditioned_network_forward(x, sigmas[i].item())
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t

            if not exists(old_denoised) or sigmas[i + 1] == 0:
                denoised_d = denoised
            else:
                h_last = t - t_fn(sigmas[i - 1])
                r = h_last / h
                gamma = - 1 / (2 * r)
                denoised_d = (1 - gamma) * denoised + gamma * old_denoised

            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
            old_denoised = denoised

        if clamp:
            x = x.clamp(-1., 1.)
        return self.latent_scaler.unscale(x) 

    # training

    def loss_weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2

    def noise_distribution(self, batch_size):
        return (self.P_mean + self.P_std * torch.randn((batch_size,), device = self.device)).exp()

    def forward(self, x_unnormalized, mask, model_kwargs={}, noise=None):
        x = self.latent_scaler.scale(x_unnormalized)
        B, L, _ = x.shape

        sigmas = self.noise_distribution(B)
        padded_sigmas = rearrange(sigmas, 'b -> b 1 1')

        noise = torch.randn_like(x)

        noised_x = x + padded_sigmas * noise  # alphas are 1. in the paper

        self_cond = None

        if self.self_condition and random() < 0.5:
            # from hinton's group's bit diffusion paper
            with torch.no_grad():
                self_cond = self.preconditioned_network_forward(noised_x, sigmas)
                self_cond.detach_()

        denoised = self.preconditioned_network_forward(noised_x, sigmas, self_cond)

        losses = F.mse_loss(denoised, x, reduction = 'none')
        losses = reduce(losses, 'b ... -> b', 'mean')

        losses = losses * self.loss_weight(sigmas)

        return losses.mean()