import torch.nn as nn
import torch



class Denoiser(nn.Module):
    """A Karras et al. preconditioner for denoising diffusion models."""

    def __init__(self, inner_model, sigma_data=1., loss_distance="mse", weighting='karras', scales=1):
        super().__init__()
        assert loss_distance in ['mse', 'huber']
        self.inner_model = inner_model
        self.loss_distance = loss_distance
        self.sigma_data = sigma_data
        self.scales = scales
        if callable(weighting):
            self.weighting = weighting
        if weighting == 'karras':
            self.weighting = torch.ones_like
        elif weighting == 'soft-min-snr':
            self.weighting = self._weighting_soft_min_snr
        elif weighting == 'snr':
            self.weighting = self._weighting_snr
        else:
            raise ValueError(f'Unknown weighting type {weighting}')

    def _weighting_soft_min_snr(self, sigma):
        return (sigma * self.sigma_data) ** 2 / (sigma ** 2 + self.sigma_data ** 2) ** 2

    def _weighting_snr(self, sigma):
        return self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)

    def get_scalings(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def loss(self, input, noise, sigma, return_model_output=False, **kwargs):
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        c_weight = self.weighting(sigma)
        noised_input = input + noise * utils.append_dims(sigma, input.ndim)
        model_output = self.inner_model(noised_input * c_in, sigma, **kwargs)
        target = (input - c_skip * noised_input) / c_out
        
        if self.loss_distance == "huber":
            loss = F.huber_loss(model_output, target, reduction="mean") * c_weight
        else:
            if self.scales == 1:
                loss = ((model_output - target) ** 2).flatten(1).mean(1) * c_weight
            else:
                sq_error = dct(model_output - target) ** 2
                f_weight = freq_weight_nd(sq_error.shape[2:], self.scales, dtype=sq_error.dtype, device=sq_error.device)
                loss = (sq_error * f_weight).flatten(1).mean(1) * c_weight
            
        if return_model_output:
            return loss, model_output
        else:
            return loss

    def forward(self, input, sigma, **kwargs):
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        return self.inner_model(input * c_in, sigma, **kwargs) * c_out + input * c_skip

