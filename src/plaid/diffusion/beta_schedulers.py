import torch
import threading
import abc
import math
import numpy as np


import torch
import abc
import math


######
# The following is adapted from pseudocode in Chen et al.,
# https://arxiv.org/abs/2301.10972
######


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def simple_linear_schedule(t, clip_min=1e-9):
    # A gamma function that simply is 1-t.
    return np.clip(1 - t, clip_min, 1.0)


def sigmoid_schedule(t, start=-3, end=3, tau=1.0, clip_min=1e-9):
    # A gamma function based on sigmoid function.
    v_start = sigmoid(start / tau)
    v_end = sigmoid(end / tau)
    output = sigmoid((t * (end - start) + start) / tau)
    output = (v_end - output) / (v_end - v_start)
    return np.clip(output, clip_min, 1.0)


def modified_cosine_schedule(t, start=0, end=1, tau=1, clip_min=1e-9):
    # A gamma function based on cosine function.
    v_start = math.cos(start * math.pi / 2) ** (2 * tau)
    v_end = math.cos(end * math.pi / 2) ** (2 * tau)
    output = math.cos((t * (end - start) + start) * math.pi / 2) ** (2 * tau)
    output = (v_end - output) / (v_end - v_start)
    return np.clip(output, clip_min, 1.0)


######
# openai guided diffusion codebase
######


def adm_cosine_schedule(t):
    return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class BetaScheduler(abc.ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def __call__(self, timesteps):
        raise NotImplementedError


class LinearBetaScheduler(BetaScheduler):
    def __init__(self, beta_start=0.0001, beta_end=0.02):
        self.beta_start = beta_start
        self.beta_end = beta_end

    def __call__(self, timesteps):
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)


class SigmoidBetaScheduler(BetaScheduler):
    def __init__(self, start=-3, end=3, tau=1.0):
        super().__init__()
        self.start = start
        self.end = end
        self.tau = tau

    def __call__(self, timesteps):
        return betas_for_alpha_bar(
            timesteps,
            lambda t: sigmoid_schedule(t, start=self.start, end=self.end, tau=self.tau),
        )


class CosineBetaScheduler(BetaScheduler):
    def __init__(self, start=0, end=1, tau=1):
        super().__init__()
        self.start = start
        self.end = end
        self.tau = tau

    def __call__(self, timesteps):
        return betas_for_alpha_bar(
            timesteps,
            lambda t: modified_cosine_schedule(
                t, start=self.start, end=self.end, tau=self.tau
            ),
        )


class ADMCosineBetaScheduler(BetaScheduler):
    def __init__(self):
        super().__init__()

    def __call__(self, timesteps):
        return betas_for_alpha_bar(timesteps, lambda t: adm_cosine_schedule(t))



# ############# k diffusion sigmas

def stratified_uniform(shape, group=0, groups=1, dtype=None, device=None):
    """Draws stratified samples from a uniform distribution."""
    if groups <= 0:
        raise ValueError(f"groups must be positive, got {groups}")
    if group < 0 or group >= groups:
        raise ValueError(f"group must be in [0, {groups})")
    n = shape[-1] * groups
    offsets = torch.arange(group, n, groups, dtype=dtype, device=device)
    u = torch.rand(shape, dtype=dtype, device=device)
    return (offsets + u) / n


stratified_settings = threading.local()

def stratified_with_settings(shape, dtype=None, device=None):
    """Draws stratified samples from a uniform distribution, using settings from a context
    manager."""
    if not hasattr(stratified_settings, 'disable') or stratified_settings.disable:
        return torch.rand(shape, dtype=dtype, device=device)
    return stratified_uniform(
        shape, stratified_settings.group, stratified_settings.groups, dtype=dtype, device=device
    )


def rand_v_diffusion(shape, sigma_data=1., min_value=0., max_value=float('inf'), device='cpu', dtype=torch.float32):
    """Draws samples from a truncated v-diffusion training timestep distribution."""
    min_cdf = math.atan(min_value / sigma_data) * 2 / math.pi
    max_cdf = math.atan(max_value / sigma_data) * 2 / math.pi
    u = stratified_with_settings(shape, device=device, dtype=dtype) * (max_cdf - min_cdf) + min_cdf
    return torch.tan(u * math.pi / 2) * sigma_data


def rand_cosine_interpolated(shape, image_d, noise_d_low, noise_d_high, sigma_data=1., min_value=1e-3, max_value=1e3, device='cpu', dtype=torch.float32):
    """Draws samples from an interpolated cosine timestep distribution (from simple diffusion)."""

    def logsnr_schedule_cosine(t, logsnr_min, logsnr_max):
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))

    def logsnr_schedule_cosine_shifted(t, image_d, noise_d, logsnr_min, logsnr_max):
        shift = 2 * math.log(noise_d / image_d)
        return logsnr_schedule_cosine(t, logsnr_min - shift, logsnr_max - shift) + shift

    def logsnr_schedule_cosine_interpolated(t, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max):
        logsnr_low = logsnr_schedule_cosine_shifted(t, image_d, noise_d_low, logsnr_min, logsnr_max)
        logsnr_high = logsnr_schedule_cosine_shifted(t, image_d, noise_d_high, logsnr_min, logsnr_max)
        return torch.lerp(logsnr_low, logsnr_high, t)

    logsnr_min = -2 * math.log(min_value / sigma_data)
    logsnr_max = -2 * math.log(max_value / sigma_data)
    u = stratified_with_settings(shape, device=device, dtype=dtype)
    logsnr = logsnr_schedule_cosine_interpolated(u, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max)
    return torch.exp(-logsnr / 2) * sigma_data


class VDiffusionSigmas(BetaScheduler):
    def __init__(self, max_value=1e3, min_value=1e-3, sigma_data=1):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.sigma_data = sigma_data
    
    def __call__(self, shape):
        return rand_v_diffusion(
            shape=shape,
            sigma_data=self.sigma_data,
            min_value=self.min_value,
            max_value=self.max_value
        )


class CosineInterpolatedSigmas(BetaScheduler):
    def __init__(self, max_value=1e3, min_value=1e-3, sigma_data=1, input_size=512, noise_d_low=32, noise_d_high=512):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.sigma_data = sigma_data
        self.input_size = input_size
        self.noise_d_low = noise_d_low
        self.noise_d_high = noise_d_high

    def __call__(self, shape):
        return rand_cosine_interpolated(
            shape=shape,
            image_d=self.input_size,
            noise_d_low = self.noise_d_low,
            noise_d_high = self.noise_d_high,
            sigma_data=self.sigma_data,
            min_value=self.min_value,
            max_value=self.max_vaue,
        ) 

        


#     if sd_config['type'] in {'v-diffusion', 'cosine'}:
#         min_value = sd_config['min_value'] if 'min_value' in sd_config else 1e-3
#         max_value = sd_config['max_value'] if 'max_value' in sd_config else 1e3
#         return partial(utils.rand_v_diffusion, sigma_data=sigma_data, min_value=min_value, max_value=max_value)
#     if sd_config['type'] == 'split-lognormal':
#         loc = sd_config['mean'] if 'mean' in sd_config else sd_config['loc']
#         scale_1 = sd_config['std_1'] if 'std_1' in sd_config else sd_config['scale_1']
#         scale_2 = sd_config['std_2'] if 'std_2' in sd_config else sd_config['scale_2']
#         return partial(utils.rand_split_log_normal, loc=loc, scale_1=scale_1, scale_2=scale_2)
#     if sd_config['type'] == 'cosine-interpolated':
#         min_value = sd_config.get('min_value', min(config['sigma_min'], 1e-3))
#         max_value = sd_config.get('max_value', max(config['sigma_max'], 1e3))
#         image_d = sd_config.get('image_d', max(config['input_size']))
#         noise_d_low = sd_config.get('noise_d_low', 32)
#         noise_d_high = sd_config.get('noise_d_high', max(config['input_size']))
#         return partial(utils.rand_cosine_interpolated, image_d=image_d, noise_d_low=noise_d_low, noise_d_high=noise_d_high, sigma_data=sigma_data, min_value=min_value, max_value=max_value)




