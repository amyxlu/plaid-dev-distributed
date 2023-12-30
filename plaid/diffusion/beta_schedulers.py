import torch
import abc
import math


class BetaScheduler(abc.ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def __call__(self, timesteps):
        raise NotImplementedError


class LinearBetaScheduler(BetaScheduler):
    def __init__(self, beta_start = 0.0001, beta_end = 0.02):
        self.beta_start = beta_start
        self.beta_end = beta_end
    
    def __call__(self, timesteps):
        scale = 1000 / timesteps
        beta_start = scale * self.beta_start
        beta_end = scale * self.beta_end
        return torch.linspace(beta_start, beta_end, timesteps)


class CosineBetaScheduler(BetaScheduler):
    def __init__(self, s = 0.008):
        self.s = s
    
    def __call__(self, timesteps):
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps) / timesteps
        alphas_cumprod = torch.cos((t + self.s) / (1 + self.s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    

class SigmoidBetaScheduler(BetaScheduler):
    def __init__(self, start = -3, end = 3, tau = 0.5, clamp_min = 1e-5):
        self.start = start
        self.end = end
        self.tau = tau
        self.clamp_min = clamp_min
    
    def __call__(self, timesteps):
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps) / timesteps
        v_start = torch.tensor(self.start / self.tau).sigmoid()
        v_end = torch.tensor(self.end / self.tau).sigmoid()
        alphas_cumprod = (-((t * (self.end - self.start) + self.start) / self.tau).sigmoid() + v_end) / (v_end - v_start)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
