"""
Roughly follows the iDDPM formulation with min-SNR weighting, etc.
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/guided_diffusion.py
"""

import typing as T
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import pandas as pd
import wandb

from einops import reduce

from tqdm.auto import tqdm

# from ema_pytorch import EMA
import lightning as L

from plaid.denoisers import BaseDenoiser
from plaid.utils import (
    LatentScaler,
    get_lr_scheduler,
    sequences_to_secondary_structure_fracs,
    to_tensor,
    print_cuda_info
)
from plaid.diffusion.beta_schedulers import BetaScheduler, ADMCosineBetaScheduler
from plaid.losses.functions import masked_mse_loss, masked_huber_loss
from plaid.losses.modules import SequenceAuxiliaryLoss, BackboneAuxiliaryLoss
from plaid.esmfold.misc import batch_encode_sequences
from plaid.proteins import LatentToSequence, LatentToStructure
from plaid.transforms import trim_or_pad_batch_first
from plaid.compression.uncompress import UncompressContinuousLatent


ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def maybe_clip(x, clip_val):
    if clip_val is None:
        return x
    else:
        return torch.clamp(x, min=-1 * clip_val, max=clip_val)


def extract(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def maybe_pad(tensor, length):
    if tensor.shape[1] != length:
        return trim_or_pad_batch_first(tensor, length, pad_idx=0)
    else:
        return tensor


class GaussianDiffusion(L.LightningModule):
    """
    Adapted from OpenAI ADM implementation, restructured as a lightning module.

    Additional considerations specific to our architecture:
    * shorten factor
    * uncompressor (on the normalized x; for auxiliary losses only)
    * unscaler (to undo the normalization; for auxiliary losses only)

    The sampling loop returns x_sampled, where x is the normalized-and-compressed
    version of the input.

    Additional features, adapted from Lucidrains and elsewhere:
    * loss weighting by SNR
    """
    def __init__(
        self,
        model: torch.nn.Module,  # denoiser
        beta_scheduler: BetaScheduler = ADMCosineBetaScheduler(),
        *,
        x_downscale_factor: float = 1.0,
        timesteps=1000,
        objective="pred_v",
        min_snr_loss_weight=False,
        min_snr_gamma=5,
        x_clip_val: T.Optional[float] = 1.0,  # determines behavior during training, and value to use during sampling
        # compression and architecture
        shorten_factor=1.0,
        unscaler: LatentScaler = LatentScaler("identity"),
        uncompressor: T.Optional[UncompressContinuousLatent] = None,
        sequence_to_latent_fn: T.Optional[T.Callable] = lambda x: x,
        pfam_to_clan_fpath: T.Optional[T.Union[str, Path]] = None,
        # sampling
        sampling_timesteps=500,  # None,
        sampling_seq_len=64,
        # optimization
        lr=1e-4,
        adam_betas=(0.9, 0.999),
        lr_sched_type: str = "constant",
        lr_num_warmup_steps: int = 0,
        lr_num_training_steps: int = 10_000_000,
        lr_num_cycles: int = 1,
        add_secondary_structure_conditioning: bool = False,
        # auxiliary losses
        sequence_constructor: T.Optional[LatentToSequence] = None,
        structure_constructor: T.Optional[LatentToStructure] = None,
        sequence_decoder_weight: float = 0.0,
        structure_decoder_weight: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.beta_scheduler = beta_scheduler
        self.self_condition = self.model.use_self_conditioning
        self.x_downscale_factor = x_downscale_factor
        self.objective = objective
        self.shorten_factor = shorten_factor
        self.hourglass_model = uncompressor
        self.sequence_to_latent_fn = sequence_to_latent_fn
        if pfam_to_clan_fpath is not None:
            self.pfam_to_clan = pd.read_csv()
        assert objective in {
            "pred_noise",
            "pred_x0",
            "pred_v",
        }, "objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])"


        """
        If latent scaler is using minmaxnorm method, it uses precomputed stats to clamp the latent space to
        roughly between (-1, 1). If self.x_clip_val is not None, at each p_sample loop, the value will
        be clipped so that we can get a cleaner range when doing the final calculation.
        """
        self.unscaler = unscaler
        self.x_clip_val = x_clip_val

        # Use float64 for accuracy.
        betas = self.beta_scheduler(timesteps)
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        # sampling specific specifications
        self.sampling_timesteps = default(
            sampling_timesteps, timesteps
        )
        assert self.sampling_timesteps <= timesteps
        self.sampling_seq_len = sampling_seq_len

        # loss weight by SNR
        self.snr = self.alphas_cumprod / (1 - self.alphas_cumprod)
        maybe_clipped_snr = self.snr.copy()
        if min_snr_loss_weight:
            maybe_clipped_snr = maybe_clipped_snr.clip(max=min_snr_gamma)

        # model prediction 
        if objective == "pred_noise":
            self.loss_weight = maybe_clipped_snr / self.snr
        elif objective == "pred_x0":
            self.loss_weight = maybe_clipped_snr
        elif objective == "pred_v":
            self.loss_weight = maybe_clipped_snr / (self.snr + 1)

        # auxiliary losses
        self.need_to_setup_sequence_decoder = sequence_decoder_weight > 0.0
        self.need_to_setup_structure_decoder = structure_decoder_weight > 0.0

        self.sequence_constructor = sequence_constructor
        self.structure_constructor = structure_constructor
        self.sequence_decoder_weight = sequence_decoder_weight
        self.structure_decoder_weight = structure_decoder_weight
        self.sequence_loss_fn = None
        self.structure_loss_fn = None

        # learning rates and optimization
        self.lr = lr
        self.adam_betas = adam_betas
        self.lr_sched_type = lr_sched_type
        self.lr_num_warmup_steps = lr_num_warmup_steps
        self.lr_num_training_steps = lr_num_training_steps
        self.lr_num_cycles = lr_num_cycles

        # other hyperparameters
        self.add_secondary_structure_conditioning = add_secondary_structure_conditioning
        self.save_hyperparameters(ignore=["model"])
    
    """
    Converting between model output, x_start, and eps
    """
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def model_output_conversion_wrapper(self, model_output, x, t):
        """Based on the objective, converts model outputs to noise or start"""        
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start, self.x_clip_val)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start, self.x_clip_val)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start, self.x_clip_val)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        return ModelPrediction(pred_noise=pred_noise, pred_x_start=x_start)
    
    def model_predictions(self, x, t, y=None, mask = None, x_self_cond = None, model_kwargs = {}):
        """Based on a noised sample and self-conditioning, denoise as x_start and eps."""
        if model_kwargs is None:
            model_kwargs = {}
        model_output = self.model(x=x, t=t, y=y, mask=mask, x_self_cond=x_self_cond, **model_kwargs)
        return self.model_output_conversion_wrapper(model_output, x, t)
    
    """
    Forward process functions
    """
    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    """Backward process functions
    """
    def p_mean_variance(
        self, x, t, y=None, mask=None, x_self_cond=None, clip_denoised=True, model_kwargs={}
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        predictions = self.model_predictions(
            x=x,
            t=t,
            y=y,
            mask=mask,
            x_self_cond=x_self_cond,
            model_kwargs=model_kwargs,
        )
        pred_xstart = predictions.pred_x_start

        if clip_denoised:
            pred_xstart = maybe_clip(pred_xstart, self.x_clip_val)

        # FIXED_LARGE:
        model_variance = np.append(self.posterior_variance[1], self.betas[1:])
        model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))

        model_variance = extract(model_variance, t, x.shape)
        model_log_variance = extract(model_log_variance, t, x.shape)

        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        return {
            "mean": to_tensor(model_mean, self.device),
            "variance": to_tensor(model_variance, self.device),
            "log_variance": to_tensor(model_log_variance, self.device),
            "pred_xstart": to_tensor(pred_xstart, self.device),
        }

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).

        Note: original implementation had a bug, see
        https://github.com/openai/guided-diffusion/issues/51
        """
        gradient = cond_fn(x, t, **model_kwargs)
        
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def p_sample(
        self,
        x,
        t,
        y=None,
        mask=None,
        x_self_cond = None,
        clip_denoised = True,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            x,
            t,
            y=y,
            mask=mask,
            x_self_cond=x_self_cond,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs
        )
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        shape,
        y=None,
        mask=None,
        noise=None,
        clip_denoised=True,
        cond_fn=None,
        model_kwargs=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            shape,
            noise=noise,
            y=y,
            mask=mask,
            clip_denoised=clip_denoised,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        shape,
        y=None,
        mask=None,
        noise=None,
        clip_denoised=True,
        cond_fn=None,
        model_kwargs=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=self.device)
        indices = list(range(self.sampling_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=self.device)
            with torch.no_grad():
                out = self.p_sample(
                    x=img,
                    t=t,
                    y=y,
                    mask=mask,
                    clip_denoised=clip_denoised,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    """
    Lightning set up
    """
    def setup_sequence_decoder(self):
        """If a reference pointer to the auxiliary sequence decoder wasn't already passed in
        at the construction of the class, load the sequence decoder onto the GPU now.
        """
        assert self.need_to_setup_sequence_decoder
        if self.sequence_constructor is None:
            self.sequence_constructor = LatentToSequence()

        self.sequence_loss_fn = SequenceAuxiliaryLoss(
            self.sequence_constructor, weight=self.sequence_decoder_weight
        )
        self.need_to_setup_sequence_decoder = False

    def setup_structure_decoder(self):
        assert self.need_to_setup_structure_decoder
        if self.structure_constructor is None:
            # Note: this will make ESMFold in the function and might be expensive
            self.structure_constructor = LatentToStructure()

        self.structure_loss_fn = BackboneAuxiliaryLoss(
            self.structure_constructor, weight=self.structure_decoder_weight
        )
        self.need_to_setup_structure_decoder = False
    
    def _move_auxiliary_models_to_device(self):
        if self.hourglass_model is not None:
            self.hourglass_model.to(self.device)
        if self.structure_constructor is not None:
            self.structure_constructor.to(self.device)
        if self.sequence_constructor is not None:
            self.sequence_constructor.to(self.device)

    def training_step(self, batch):
        # print_cuda_info()
        self._move_auxiliary_models_to_device()
        
        loss, log_dict = self.run_step(
            batch, model_kwargs={}, noise=None, clip_x_start=True
        )
        N = self._get_batch_size(batch)
        self.log_dict(
            {f"train/{k}": v for k, v in log_dict.items()}, on_step=True, on_epoch=False, batch_size=N
        )
        return loss

    def validation_step(self, batch):
        # print_cuda_info()
        self._move_auxiliary_models_to_device()

        # Extract the starting images from data batch
        loss, log_dict = self.run_step(
            batch, model_kwargs={}, noise=None, clip_x_start=True
        )
        N = self._get_batch_size(batch)
        self.log_dict(
            {f"val/{k}": v for k, v in log_dict.items()}, on_step=False, on_epoch=True, batch_size=N, sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, betas=self.adam_betas
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

    """
    Forward pass with loss calculations
    """

    def forward(
        self,
        x_start,
        mask,
        clan_idx=None,
        sequences=None,
        model_kwargs={},
        noise=None,
    ):
        # x_start was already compressed and bounded to -1 and 1 during dataloader preprocessing
        B, L, _ = x_start.shape
        t = torch.randint(0, self.num_timesteps, (B,))
        t = t.long().to(self.device)

        # potentially unscale
        x_start *= self.x_downscale_factor

        # noise sample
        B, L, C = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(
                    x_t=x_t,
                    t=t,
                    mask=mask,
                    y=clan_idx,
                    x_self_cond=x_self_cond,
                    model_kwargs=model_kwargs,
                ).pred_x_start
                x_self_cond.detach_()
        model_kwargs["x_self_cond"] = x_self_cond

        # add conditioning information here
        if self.add_secondary_structure_conditioning:
            assert not sequences is None
            model_kwargs["cond_dict"] = self.get_secondary_structure_fractions(
                sequences
            )

        # main inner model forward pass
        model_out = self.model(x_t, t, y=clan_idx, mask=mask, **model_kwargs)
        
        # reconstruction / "main diffusion loss"
        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        elif self.objective == "pred_v":
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f"unknown objective {self.objective}")

        diffusion_loss = masked_mse_loss(model_out, target, mask, reduce="batch")
        diffusion_loss = diffusion_loss * extract(
            self.loss_weight, t, diffusion_loss.shape
        )
        return model_out, x_t, t, diffusion_loss.mean() 

    def sequence_loss(self, latent, aatype, mask, cur_weight=None):
        if self.need_to_setup_sequence_decoder:
            self.setup_sequence_decoder()
        # sequence should be the one generated when saving the latents,
        # i.e. lengths are already trimmed to self.max_seq_len
        # if cur_weight is None, no annealing is done except for the weighting
        # specified when specifying the class
        _, L, _ = latent.shape
        aatype = maybe_pad(aatype, L)
        mask = maybe_pad(mask, L)
        return self.sequence_loss_fn(latent, aatype, mask, return_reconstructed_sequences=True)

    def structure_loss(self, latent, gt_structures, sequences, cur_weight=None):
        # TODO: does this work for the rocklin mini-proteins dataset?
        if self.need_to_setup_structure_decoder:
            self.setup_structure_decoder()
        return self.structure_loss_fn(latent, gt_structures, sequences, cur_weight)

    # def get_secondary_structure_fractions(
    #     self, sequences: T.List[str], origin_dataset: str = "uniref"
    # ):
    #     # currently only does secondary structure
    #     sec_struct_fracs = sequences_to_secondary_structure_fracs(
    #         sequences, quantized=True, origin_dataset=origin_dataset
    #     )
    #     sec_struct_fracs = torch.tensor(sec_struct_fracs).to(self.device)
    #     cond_dict = {"secondary_structure": sec_struct_fracs}
    #     return cond_dict

    def process_x_to_latent(self, x):
        """Decompress and undo channel-wise scaling"""
        if self.hourglass_model is not None:
            x = self.hourglass_model.uncompress(x)
        return self.unscaler.unscale(x)

    def run_step(self, batch, model_kwargs={}, noise=None, clip_x_start=True):
        # use different batch unpacking strategies
        if isinstance(batch, dict):
            # hack: embedding batch output is a dict
            out = self._run_diffusion_step_saved_embeddings(batch, model_kwargs, noise)
        elif isinstance(batch, list) or isinstance(batch, tuple):
            # hack: fasta datamodule outputs (header, sequence) 
            out = self._run_diffusion_step_embed_on_the_fly(batch)

        # unpack outputs 
        diffusion_loss = out.get("diffusion_loss")
        model_output = out.get("model_output")
        sequences = out.get("sequences")
        x_t = out.get("x_t")
        t = out.get("t")

        log_dict = {"diffusion_loss": diffusion_loss}
        using_seq_loss = self.sequence_decoder_weight > 0.0 
        using_struct_loss = self.structure_decoder_weight > 0.0

        ###############
        # Auxiliary losses: structure and sequence constraint
        ###############
        if using_seq_loss or using_struct_loss:
            pred_x_start = self.model_output_conversion_wrapper(model_output, x_t, t).pred_x_start
            pred_latent = self.process_x_to_latent(pred_x_start)
        
        if self.sequence_decoder_weight > 0.0:
            true_aatype, seq_mask, _, _, _ = batch_encode_sequences(sequences)
            seq_loss, seq_loss_dict, recons_strs = self.sequence_loss(
                pred_latent, true_aatype, seq_mask, cur_weight=None
            )
            log_dict = (
                log_dict | seq_loss_dict
            )  # shorthand for combining dictionaries, requires python >= 3.9
            tbl = pd.DataFrame({"reconstructed": recons_strs, "original": sequences})
            wandb.log({"recons_strs_tbl": wandb.Table(dataframe=tbl)})
        else:
            seq_loss = 0.0

        # if self.structure_decoder_weight > 0.0:
        #     assert (
        #         not gt_structures is None
        #     ), "If using structure as an auxiliary loss, ground truth structures must be provided"
        #     # structure loss implicitly calls tokenizer again
        #     struct_loss, struct_loss_dict = self.structure_loss(
        #         latent, gt_structures, sequences, cur_weight=None
        #     )
        #     log_dict = log_dict | struct_loss_dict
        # else:
        #     struct_loss = 0.0
        struct_loss = 0.0
        loss = diffusion_loss + seq_loss + struct_loss
        log_dict["loss"] = loss
        return loss, log_dict

    def _run_diffusion_step_saved_embeddings(self, batch, model_kwargs={}, noise=None):
        """
        Runs the diffusion step (without auxiliary losses) for a batch of pre-computed embeddings & clan.
        """
        embs = batch.get("emb")
        mask = batch.get("mask")
        clan = batch.get("clan")
        sequences = batch.get("sequence")

        model_output, x_t, t, diffusion_loss = self(
            x_start=embs,
            mask=mask,
            clan_idx=clan,
            sequences=sequences,
            model_kwargs=model_kwargs,
            noise=noise
        )
        return {
            "diffusion_loss": diffusion_loss,
            "model_output": model_output,
            "sequences": sequences,
            "x_t": x_t,
            "clan": clan,
            "t": t
        }

    def _run_diffusion_step_embed_on_the_fly(self, batch, model_kwargs={}, noise=None):
        """
        Runs the diffusion step assuming a batch of header and sequence.
        In this case, the sequence must be embedded and compressed, and the
        header parsed to map to the clan. If these models aren't already created,
        the method will trigger its creation.
        """
        headers, sequences = batch
        embedding, mask = self.sequence_to_latent_fn(sequences)  # possibly already scaled and compressed
        clan_idxs = list(map(lambda header: self._header_to_clan_idx(header), headers))

        model_output, x_t, t, diffusion_loss = self(
            x_start=embedding,
            mask=mask,
            clan_idx=clan_idxs,
            sequences=sequences,
            clan_idxs=clan_idxs,
            model_kwargs=model_kwargs,
            noise=noise,
        )
        return {
            "diffusion_loss": diffusion_loss,
            "model_output": model_output,
            "sequences": sequences,
            "x_t": x_t,
            "clan": clan_idxs,
            "t": t
        }

    def _get_batch_size(self, batch):
        if isinstance(batch, dict):
            return len(batch['emb'])
        elif isinstance(batch, tuple):
            return len(batch[0])
        else:
            raise TypeError(f"Expected batch to be dict or tuple, got {type(batch)}.")
        
    def _make_accession_to_clan_data_structures(self):
        fam_to_clan_df = pd.read_csv(
            self.accession_to_clan_file,
            sep="\t",
            header=None
    )
        # read accession to clan dataframe and grab the first clan for each pfam family accession
        header = ["accession", "clan", "short_name", "gene_name", "description"]
        fam_to_clan_df.columns = header
        accession_to_clan = fam_to_clan_df.groupby("accession").first().filter(['accession','clan'], axis=1)
        accession_to_clan = accession_to_clan.to_dict()['clan']

        # create an unique index representer for each clan
        clans = fam_to_clan_df.clan.dropna().unique()
        clans.sort()

        # store mapping
        self.clans_to_idx = dict(zip(clans, np.arange(len(clans))))
        self.accession_to_clan = accession_to_clan
        self.clans = clans
    
    def _header_to_clan_idx(self, header):
        subid = header.split(" ")[-1]
        accession = subid.split(".")[0]
        clan_id = self.accession_to_clan[accession]
        if clan_id is None:
            return len(self.clans)  # dummy idx for unknown clan
        else:
            return self.clans_to_idx[clan_id]
    