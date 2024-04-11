"""
Roughly follows the iDDPM formulation with min-SNR weighting, etc.
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/guided_diffusion.py
"""

import typing as T
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


def extract_into_tensor(arr, timesteps, broadcast_shape):
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


class GaussianDiffusion(L.LightningModule):
    def __init__(
        self,
        model: BaseDenoiser,
        latent_scaler: LatentScaler = LatentScaler(),
        beta_scheduler: BetaScheduler = ADMCosineBetaScheduler(),
        *,
        x_downscale_factor: float = 1.0,
        timesteps=1000,
        objective="pred_v",
        min_snr_loss_weight=False,
        min_snr_gamma=5,
        soft_clip_x_start_to: T.Optional[float] = 1.0,  # determines behavior during training, and value to use during sampling
        # compression and architecture
        shorten_factor=1.0,
        uncompressor: T.Optional[UncompressContinuousLatent] = None,
        # sampling
        ddim_sampling_eta=0.0,  # 0 is DDIM and 1 is DDPM
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
        # latent_reconstruction_method: str = "unnormalized_x_recons",
    ):
        super().__init__()
        self.model = model
        self.beta_scheduler = beta_scheduler
        self.self_condition = self.model.use_self_conditioning
        self.x_downscale_factor = x_downscale_factor
        self.objective = objective
        self.shorten_factor = shorten_factor
        self.uncompressor = uncompressor
        assert objective in {
            "pred_noise",
            "pred_x0",
            "pred_v",
        }, "objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])"

        # self.latent_recons_method = latent_reconstruction_method

        """
        If latent scaler is using minmaxnorm method, it uses precomputed stats to clamp the latent space to
        roughly between (-1, 1). If self.soft_clip_x_start_to is not None, at each p_sample loop, the value will
        be clipped so that we can get a cleaner range when doing the final calculation.
        """
        self.latent_scaler = latent_scaler
        self.soft_clip_x_start_to = soft_clip_x_start_to

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

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps
        )  # default num sampling timesteps to number of timesteps at training
        assert self.sampling_timesteps <= timesteps
        self.sampling_seq_len = sampling_seq_len
        self.ddim_sampling_eta = ddim_sampling_eta

        # loss weight
        self.snr = self.alphas_cumprod / (1 - self.alphas_cumprod)
        maybe_clipped_snr = self.snr.copy()
        if min_snr_loss_weight:
            maybe_clipped_snr = maybe_clipped_snr.clip(max=min_snr_gamma)

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

        # other hyperparameteres
        self.add_secondary_structure_conditioning = add_secondary_structure_conditioning
        self.save_hyperparameters(ignore=["model"])

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

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
            * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_v(self, x_start, t, noise):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def maybe_clip(self, x):
        if self.soft_clip_x_start_to is None:
            return x
        else:
            return torch.clamp(x, min=-1 * self.soft_clip_x_start_to, max=self.soft_clip_x_start_to)

    def model_predictions(self, x, t, mask=None, model_kwargs={}):
        model_output = self.model(x, t, mask, **model_kwargs)
        # do a soft clip since we're in latent space and min/max stats are not global

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = self.maybe_clip(x_start)

        elif self.objective == "pred_x0":
            x_start = model_output
            x_start = self.maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = self.maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        
        if self.uncompressor is not None:
            x_start = self.uncompressor.uncompress(x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, model_kwargs={}, clip_denoised=True):
        preds = self.model_predictions(x, t, model_kwargs=model_kwargs)
        x_start = preds.pred_x_start

        if clip_denoised:
            assert not self.soft_clip_x_start_to is None
            x_start = self.maybe_clip(x_start)

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
        # print("gradient: ", (variance * gradient).mean())
        return new_mean

    @torch.no_grad()
    def p_sample(
        self,
        x,
        t: int,
        model_kwargs={},
        cond_fn=None,
        guidance_kwargs=None,
        clip_denoised=True,
    ):
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
        clip_denoised=True,
        unscale=False
    ):
        batch, device = shape[0], self.device

        xt = torch.randn(shape, device=device)
        xts = [xt]
        uncompressed_xts = []
        x_start = None

        for t in tqdm(
            reversed(range(0, self.sampling_timesteps)),
            desc="sampling loop time step",
            total=self.sampling_timesteps,
            leave=False
        ):
            # maybe add self conditioning to model input
            self_cond = x_start if self.self_condition else None
            model_kwargs["x_self_cond"] = self_cond

            # sample
            xt, x_start = self.p_sample(
                xt, t, model_kwargs, cond_fn, guidance_kwargs, clip_denoised
            )
            xts.append(xt)

            # if diffusing in compressed space, also return an uncompressed version of the latent
            if self.uncompressor is not None:
                uncompressed_xt = self.uncompressor.uncompress(xt) 
                uncompressed_xts.append(uncompressed_xt)

        # return either compressed or uncompressed 
        ret_list = xts if self.uncompressor is None else uncompressed_xts

        # return either all timesteps or final only
        ret = ret_list[-1] if not return_all_timesteps else torch.stack(ret, dim=1)

        # maybe unscale, if working with a normalized pre-compression space
        if unscale:
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
        unscale=False,
    ):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = (
            shape[0],
            self.device,
            self.sampling_timesteps,
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
            model_kwargs["x_self_cond"] = self_cond
            pred_noise, x_start, *_ = self.model_predictions(
                img, time_cond, model_kwargs=model_kwargs
            )

            if self.uncompressor is not None:
               img = self.uncompressor.uncompress(img)

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
        if unscale:
            ret = self.latent_scaler.unscale(ret)
        return ret

    @torch.no_grad()
    def sample(
        self,
        shape,
        return_all_timesteps=False,
        model_kwargs={},
        cond_fn=None,
        guidance_kwargs=None,
        clip_denoised=True,
        unscale=False,
        use_ddim_sampling=None,
    ):
        sample_fn = (
            self.p_sample_loop if not use_ddim_sampling else self.ddim_sample
        )
        return sample_fn(
            shape,
            return_all_timesteps=return_all_timesteps,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            guidance_kwargs=guidance_kwargs,
            clip_denoised=clip_denoised,
            unscale=unscale
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
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def _make_mask(self, latent: torch.Tensor, sequences: T.List[str]) -> torch.Tensor:
        """Make the mask from the input latent and the string of original sequences.
        The latent might have been shortened, so the mask always takes the same length as the latent,
        but we use the string of original sequences to compute which positions should be masked.
        This requires the shortening factor used in the dataloader to be provided a priori."""
        B, L, _ = latent.shape
        sequence_lengths = torch.tensor([len(s) for s in sequences], device=latent.device)[:, None]
        idxs = torch.tile(torch.arange(L, device=latent.device), (B, 1))
        fractional_lengths = torch.tile(sequence_lengths / self.shorten_factor, (1, L))
        return (idxs < fractional_lengths).long()

    def forward(
        self,
        x_unnormalized,
        sequences,
        model_kwargs={},
        noise=None,
    ):
        x_start = self.latent_scaler.scale(x_unnormalized).to(self.device)
        B, L, _ = x_start.shape
        t = (
            torch.randint(0, self.num_timesteps, (B,))
            .long()
            .to(self.device)
        )

        """
        sequence strings must be the trimmed version that matches latent
        if the length of the structure doesn't match, we prioritize choosing a sequence string that
        matches the latent 
        """
        mask = self._make_mask(x_start, sequences)

        # potentially unscale
        x_start *= self.x_downscale_factor

        # noise sample
        B, L, C = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t, mask).pred_x_start
                x_self_cond.detach_()
        model_kwargs["x_self_cond"] = x_self_cond

        # add conditioning information here
        if self.add_secondary_structure_conditioning:
            model_kwargs["cond_dict"] = self.get_secondary_structure_fractions(
                sequences
            )

        # main inner model forward pass
        model_out = self.model(x, t, mask=mask, **model_kwargs)

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

        recons_loss = masked_mse_loss(model_out, target, mask, reduce="batch")
        recons_loss = recons_loss * extract_into_tensor(
            self.loss_weight, t, recons_loss.shape
        )
        recons_loss = recons_loss.mean()

        # auxiliary losses
        x_recons = self.model_predictions(
            x, t, mask, model_kwargs
        ).pred_x_start

        log_dict = {
            "recons_loss": recons_loss,
        }
        return recons_loss, x_recons, log_dict

    def sequence_loss(self, latent, aatype, mask, cur_weight=None):
        if self.need_to_setup_sequence_decoder:
            self.setup_sequence_decoder()
        # sequence should be the one generated when saving the latents,
        # i.e. lengths are already trimmed to self.max_seq_len
        # if cur_weight is None, no annealing is done except for the weighting
        # specified when specifying the class
        return self.sequence_loss_fn(latent, aatype, mask, return_reconstructed_sequences=True)

    def structure_loss(self, latent, gt_structures, sequences, cur_weight=None):
        if self.need_to_setup_structure_decoder:
            self.setup_structure_decoder()
        return self.structure_loss_fn(latent, gt_structures, sequences, cur_weight)

    def get_secondary_structure_fractions(
        self, sequences: T.List[str], origin_dataset: str = "uniref"
    ):
        # currently only does secondary structure
        sec_struct_fracs = sequences_to_secondary_structure_fracs(
            sequences, quantized=True, origin_dataset=origin_dataset
        )
        sec_struct_fracs = torch.tensor(sec_struct_fracs).to(self.device)
        cond_dict = {"secondary_structure": sec_struct_fracs}
        return cond_dict

    def compute_loss(self, batch, model_kwargs={}, noise=None, clip_x_start=True):
        """
        loss logic is in the forward function, make wrapper for pytorch lightning
        sequence must be the same length as the embs (ie. saved during emb caching)
        -------
        small hack: if using a structure wrapper, the last batch element is a structure feature dict
        otherwise it's just pdb_ids which we don't need.
        2024/02/06 this is currently only for H5ShardDataset and CATHStructureDataset
        """
        embs, sequences, _ = batch
        recons_loss, x_recons, log_dict = self(embs, sequences)
        return recons_loss, log_dict

        # if isinstance(batch[-1], dict):
        #     # dictionary of structure features
        #     assert "backbone_rigid_tensor" in batch[-1].keys()
        #     embs, sequences, gt_structures = batch
        #     recons_loss, x_recons, log_dict = self(embs, sequences, gt_structures)
        # elif isinstance(batch[-1][0], str):
        #     embs, sequences, _ = batch
        #     recons_loss, x_recons, log_dict = self(embs, sequences)
        # else:
        #     raise Exception(
        #         f"Batch tuple not understood. Data type of last element of batch tuple is {type(batch[-1])}."
        #     )
        
        # # TODO: anneal losses

        # if self.sequence_decoder_weight > 0.0:
        #     # mask has same length as sequence even if latent was shortened!
        #     true_aatype, seq_mask, _, _, _ = batch_encode_sequences(sequences)
        #     seq_loss, seq_loss_dict, recons_strs = self.sequence_loss(
        #         latent, true_aatype, seq_mask, cur_weight=None
        #     )
        #     log_dict = (
        #         log_dict | seq_loss_dict
        #     )  # shorthand for combining dictionaries, requires python >= 3.9
        #     tbl = pd.DataFrame({"reconstructed": recons_strs, "original": sequences})
        #     wandb.log({"recons_strs_tbl": wandb.Table(dataframe=tbl)})
        #     # wandb.log({f"{prefix}/recons_strs_tbl": wandb.Table(dataframe=tbl)})
        # else:
        #     seq_loss = 0.0

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

        # loss = recons_loss + seq_loss + struct_loss
        # log_dict["loss"] = loss
        # return loss, log_dict


    def training_step(self, batch):
        loss, log_dict = self.compute_loss(
            batch, model_kwargs={}, noise=None, clip_x_start=True
        )
        N = len(batch[0])
        self.log_dict(
            {f"train/{k}": v for k, v in log_dict.items()}, on_step=True, on_epoch=True, batch_size=N
        )
        return loss

    def validation_step(self, batch):
        # Extract the starting images from data batch
        loss, log_dict = self.compute_loss(
            batch, model_kwargs={}, noise=None, clip_x_start=True
        )
        N = len(batch[0])
        self.log_dict(
            {f"val/{k}": v for k, v in log_dict.items()}, on_step=False, on_epoch=True, batch_size=N
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
