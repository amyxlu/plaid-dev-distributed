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
from plaid.losses import masked_mse_loss, masked_huber_loss, SequenceAuxiliaryLoss, BackboneAuxiliaryLoss
from plaid.decoder import FullyConnectedNetwork
from plaid.esmfold.trunk import FoldingTrunk

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
        # sampling
        ddim_sampling_eta=0.0,  # 0 is DDIM and 1 is DDPM
        sampling_timesteps=500, # None,
        sampling_seq_len=64,
        use_ddim=False,
        # optimization
        lr=1e-4,
        adam_betas=(0.9, 0.999),
        lr_sched_type: str = "constant",
        lr_num_warmup_steps: int = 0,
        lr_num_training_steps: int = 10_000_000,
        lr_num_cycles: int = 1,
        add_secondary_structure_conditioning: bool = False,
        # auxiliary losses
        sequence_decoder: torch.nn.Module = None,
        structure_decoder: torch.nn.Module = None,
        sequence_decoder_weight: float = 0.0,
        structure_decoder_weight: float = 0.0,
        latent_reconstruction_method: str = "unnormalized_x_recons"
    ):
        super().__init__()
        self.model = model
        self.beta_scheduler = beta_scheduler
        self.latent_scaler = latent_scaler
        self.self_condition = self.model.use_self_conditioning
        self.x_downscale_factor = x_downscale_factor
        self.objective = objective
        assert objective in {
            "pred_noise",
            "pred_x0",
            "pred_v",
        }, "objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])"

        self.latent_recons_method = latent_reconstruction_method
        assert latent_reconstruction_method in {
            "model_out",
            "x_recons",
            "unnormalized_x_recons",
            "unnormalized_upscale_x_recons"
        }

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
        self.is_ddim_sampling = use_ddim
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
        self.need_to_setup_sequence_decoder = sequence_decoder_weight > 0.
        self.need_to_setup_structure_decoder = structure_decoder_weight > 0.

        self.sequence_decoder = sequence_decoder
        self.structure_decoder = structure_decoder
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
        """ If a reference pointer to the auxiliary sequence decoder wasn't already passed in
        at the construction of the class, load the sequence decoder onto the GPU now. 
        """
        assert self.need_to_setup_sequence_decoder
        if self.sequence_decoder is None:
            self.sequence_decoder = FullyConnectedNetwork.from_pretrained(device=self.device, eval_mode=True)
        self.sequence_loss_fn = SequenceAuxiliaryLoss(self.sequence_decoder, weight=self.sequence_decoder_weight)
        self.need_to_setup_sequence_decoder = False
    
    def setup_structure_decoder(self):
        assert self.need_to_setup_structure_decoder
        if self.structure_decoder is None:
            self.structure_decoder = FoldingTrunk.from_pretrained(device=self.device, eval_mode=True) 
        self.structure_loss_fn = BackboneAuxiliaryLoss(esmfold_trunk=self.structure_decoder, weight=self.structure_decoder_weight)
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

    def model_predictions(self, x, t, mask=None, model_kwargs={}, clip_x_start=False):
        model_output = self.model(x, t, mask, **model_kwargs)
        maybe_clip = (
            partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        )

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

    def p_mean_variance(self, x, t, model_kwargs={}, clip_denoised=True):
        preds = self.model_predictions(x, t, model_kwargs=model_kwargs)
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
    ):
        batch, device = shape[0], self.device

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for t in tqdm(
            reversed(range(0, self.sampling_timesteps)),
            desc="sampling loop time step",
            total=self.sampling_timesteps,
        ):
            self_cond = x_start if self.self_condition else None
            model_kwargs["x_self_cond"] = self_cond
            img, x_start = self.p_sample(
                img, t, model_kwargs, cond_fn, guidance_kwargs, clip_denoised
            )
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
        shape,
        return_all_timesteps=False,
        model_kwargs={},
        cond_fn=None,
        guidance_kwargs=None,
        clip_denoised=True,
    ):
        sample_fn = (
            self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        )
        return sample_fn(
            shape,
            return_all_timesteps=return_all_timesteps,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            guidance_kwargs=guidance_kwargs,
            clip_denoised=clip_denoised,
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

    def forward(self, x_unnormalized, mask, model_kwargs={}, gt_structures=None, noise=None):
        x_start = self.latent_scaler.scale(x_unnormalized)
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],)).long().to(x.device)

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
            model_kwargs["cond_dict"] = self.get_secondary_structure_fractions(sequence)

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
        recons_loss = recons_loss * extract_into_tensor(self.loss_weight, t, recons_loss.shape)

        # auxiliary losses
        x_recons = self.model_predictions(x, t, noise).pred_x_start
        latent_forms = {
            "model_out": model_out,
            "x_recons": x_recons,
            "unnormalized_x_recons": self.latent_scaler.unscale(x_recons), 
            "unnormalized_upscale_x_recons": self.latent_scaler.unscale(x_recons / self.x_downscale_factor)
        }

        log_dict = {
            "means/model_out": model_out.mean(),
            "means/x_recons": x_recons.mean(),
            "means/unnormalized_x_recons": latent_forms['unnormalized_x_recons'].mean(),
            "means/unnormalized_upscale_x_recons": latent_forms['unnormalized_upscale_x_recons'].mean(),
            "stds/model_out": model_out.std(),
            "stds/x_recons": x_recons.std(),
            "stds/unnormalized_x_recons": latent_forms['unnormalized_x_recons'].std(),
            "stds/unnormalized_upscale_x_recons": latent_forms['unnormalized_upscale_x_recons'].std(),
            "recons_loss": recons_loss
        }
        latent = latent_forms[self.latent_recons_method]

        # TODO: anneal losses
        if self.sequence_decoder_weight > 0.:
            seq_loss = self.sequence_loss(latent, sequence, cur_weight=None)
            log_dict['seq_loss'] = seq_loss
        else:
            seq_loss = 0.
        
        if self.structure_decoder_weight > 0.:
            assert not gt_structures is None, "If using structure as an auxiliary loss, ground truth structures must be provided"
            struct_loss = self.structure_loss(latent, gt_structures, cur_weight=None)
            log_dict['struct_loss'] = struct_loss
        else:
            struct_loss = 0.
        
        loss = recons_loss + seq_loss + struct_loss
        log_dict['loss'] = loss
        return loss, log_dict
    
    def sequence_loss(self, latent, sequence, cur_weight=None):
        if self.need_to_setup_sequence_decoder:
            self.setup_sequence_decoder()
        # sequence should be the one generated when saving the latents,
        # i.e. lengths are already trimmed to self.max_seq_len
        # if cur_weight is None, no annealing is done except for the weighting
        # specified when specifying the class
        return self.sequence_loss_fn(latent, sequence, cur_weight)
    
    def structure_loss(self, latent, gt_structures, cur_weight=None):
        if self.need_to_setup_structure_decoder:
            self.setup_structure_decoder()
        return self.structure_loss_fn(latent, gt_structures, cur_weight)

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
    
    def compute_loss(self, batch):
        # loss logic is in the forward function, make wrapper for pytorch lightning
        return self(batch)

    def training_step(self, batch):
        loss, log_dict = self.compute_loss(batch)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=False,
        )
        self.log_dict({f"train/{k}": v for k, v in log_dict.items()})
        return loss

    def validation_step(self, batch):
        # Extract the starting images from data batch
        loss, log_dict = self.compute_loss(batch)
        self.log(
            "val/loss", loss, on_step=True, on_epoch=False
        )
        self.log_dict({f"val/{k}": v for k, v in log_dict.items()})
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


if __name__ == "__main__":
    from plaid.denoisers import UTriSelfAttnDenoiser
    from plaid.datasets import CATHStructureDataModule
    # from plaid.denoisers import PreinitializedTriSelfAttnDenoiser

    shard_dir = "/homefs/home/lux70/storage/data/cath/shards/"
    pdb_dir = "/data/bucket/lux70/data/cath/dompdb"
    dm = CATHStructureDataModule(
        shard_dir,
        pdb_dir,
        seq_len=64,
        batch_size=32,num_workers=0
    )
    dm.setup()
    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))

    device = torch.device("cuda:0")
    model = UTriSelfAttnDenoiser(
        num_blocks=7,
        hid_dim=1024,
        conditioning_strategy="hidden_concat",
        use_self_conditioning=True
    )
    model.to(device)

    diffusion = GaussianDiffusion(model)
    diffusion.to(device=device)
    import IPython; IPython.embed()