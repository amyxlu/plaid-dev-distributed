import typing as T
import time
from pathlib import Path

from omegaconf import OmegaConf

from tqdm import trange
import torch
import numpy as np

from plaid.diffusion import FunctionOrganismDiffusion
from plaid.denoisers import FunctionOrganismDiT, FunctionOrganismUDiT, DenoiserKwargs
from plaid.constants import COMPRESSION_INPUT_DIMENSIONS
from plaid.datasets import NUM_ORGANISM_CLASSES, NUM_FUNCTION_CLASSES
from plaid.utils import timestamp


device = torch.device("cuda")


def default(x, val):
    return x if x is not None else val


class SampleLatent:
    def __init__(
        self,
        # model setup
        model_id: str = "5j007z42",
        model_ckpt_dir: str = "/data/lux70/plaid/checkpoints/plaid-compositional",

        # sampling setup
        organism_idx: int = NUM_ORGANISM_CLASSES,
        function_idx: int = NUM_FUNCTION_CLASSES,
        cond_scale: float = 7,
        num_samples: int = -1,
        beta_scheduler_name: T.Optional[str] = "sigmoid",
        sampling_timesteps: int = 1000 ,
        batch_size: int = -1,
        length: int = 32,  # the final length, after decoding back to structure/sequence, is twice this value
        return_all_timesteps: bool = False,
        sample_scheduler: str = "ddim",  # ["dpm", "ddpm"]

        # output setup
        output_root_dir: str = "/data/lux70/plaid/artifacts/samples",
    ):
        self.model_id = model_id
        self.model_ckpt_dir = Path(model_ckpt_dir)
        self.organism_idx = organism_idx
        self.function_idx = function_idx
        self.cond_scale = cond_scale
        self.num_samples = num_samples
        self.length = length
        self.return_all_timesteps = return_all_timesteps
        self.output_root_dir = output_root_dir

        # default to cuda
        self.device = torch.device("cuda")
        
        # if no batch size is provided, sample all at once
        self.batch_size = batch_size if batch_size > 0 else num_samples

        # set up paths
        self.output_dir = Path(self.output_root_dir) / model_id / f"f{self.function_idx}_o{self.organism_idx}"
        model_path = self.model_ckpt_dir / model_id / "last.ckpt"
        config_path = self.model_ckpt_dir / model_id / "config.yaml"

        # load config
        self.cfg = OmegaConf.load(config_path)

        # if specified a specific beta scheduler or number of sampling steps, override
        # otherwise, use what was used during training
        self.beta_scheduler_name = default(beta_scheduler_name, self.cfg.beta_scheduler_name)
        self.sampling_timesteps = default(sampling_timesteps, self.cfg.timesteps)

        # create the denoiser
        self.denoiser = self.create_denoiser(model_path)

        # Create the sampler solver
        if sample_scheduler == "dpm":
            self.create_dpm_solver()
        else:
            self.diffusion = self.create_diffusion(self.denoiser, model_path)

    def create_dpm_solver(self):
        from ..diffusion.dpm_solver import NoiseScheduleVP, DPM_Solver
        from ..diffusion.beta_schedulers import make_beta_scheduler

        """
        Make noise schedule
        """

        # set up betas from beta scheduler
        beta_scheduler = make_beta_scheduler(
            self.cfg.beta_scheduler_name,
            self.cfg.beta_scheduler_start,
            self.cfg.beta_scheduler_end,
            self.cfg.beta_scheduler_tau
        )
        betas = beta_scheduler(self.sampling_timesteps)
        betas = torch.tensor(betas, dtype=torch.float64)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        noise_schedule = NoiseScheduleVP(
            schedule="discrete",
            betas=betas, 
            alphas_cumprod=alphas_cumprod
        ) 

        """
        Make model wrapper
        """
        from ..diffusion.dpm_solver import model_wrapper
        pred_type_conversion = {
            "pred_v": "v",
            "pred_noise": "noise",
            "pred_x0": "x_start"
        }

        # TODO: how does double self cond work"?
        wrapper = model_wrapper(
            model=self.denoiser,
            noise_schedule=noise_schedule,
            model_type=pred_type_conversion[self.cfg.diffusion.objective],
            model_kwargs={},
            guidance_type="classifier-free",
            guidance_scale=self.cond_scale
        )


    def create_denoiser(
        self,
        model_path,
    ):
        cfg = self.cfg
        compression_model_id = cfg['compression_model_id']
        # shorten_factor = COMPRESSION_SHORTEN_FACTORS[compression_model_id]
        input_dim = COMPRESSION_INPUT_DIMENSIONS[compression_model_id]

        # instantiate the correct denoiser class
        # UDiT supports skip connections and memory-efficient attention, while DiT does not
        denoiser_kwargs = cfg.denoiser
        denoiser_class = denoiser_kwargs.pop("_target_")

        if denoiser_class == "plaid.denoisers.FunctionOrganismUDiT":
            denoiser = FunctionOrganismUDiT(**denoiser_kwargs, input_dim=input_dim)
        elif denoiser_class == "plaid.denoisers.FunctionOrganismDiT":
            denoiser = FunctionOrganismDiT(**denoiser_kwargs, input_dim=input_dim)
        else:
            raise ValueError(f"Unknown denoiser class: {denoiser_class}")

        # lask.ckpt automatically links to the EMA
        ckpt = torch.load(model_path)

        # remove the prefix from the state dict if torch.compile was used during training
        mod_state_dict = {}
        for k, v in ckpt['state_dict'].items():
            if k[:16] == "model._orig_mod.":
                mod_state_dict[k[16:]] = v

        # load weights and create diffusion object
        denoiser.load_state_dict(mod_state_dict)
        denoiser = denoiser.to(self.device)
        return denoiser

    def create_diffusion(self, denoiser):
        diffusion_kwargs = self.cfg.diffusion
        diffusion_kwargs.pop("_target_")
        diffusion_kwargs["sampling_timesteps"] = self.sampling_timesteps 
        diffusion_kwargs["beta_scheduler_name"] = self.beta_scheduler_name
        diffusion = FunctionOrganismDiffusion(model=denoiser,**diffusion_kwargs)
        diffusion = diffusion.to(self.device)
        return diffusion
    
    def dpm_sample(self):
        import IPython;IPython.embed()
    
    def sample(self, use_ddim=True):
        N, L, C = self.batch_size, self.length, self.diffusion.model.input_dim 
        shape = (N, L, C)

        if use_ddim:
            sample_loop_fn = self.diffusion.ddim_sample_loop
        else:
            sample_loop_fn = self.diffusion.p_sample_loop

        # assuming no gradient-guided diffusion:
        with torch.no_grad():
            sampled_latent = sample_loop_fn(
                shape=shape,
                organism_idx=self.organism_idx,
                function_idx=self.function_idx,
                return_all_timesteps=self.return_all_timesteps, 
                cond_scale=self.cond_scale
            )
        return sampled_latent
    
    def run(self):
        num_samples = max(self.num_samples, self.batch_size)
        all_sampled = []
        cur_n_sampled = 0

        start = time.time()
        for _ in trange(0, num_samples, self.batch_size, desc="Sampling batches:"):
            sampled_latent = self.sample()
            all_sampled.append(sampled_latent)
            cur_n_sampled += self.batch_size
        end = time.time()
        print(f"Sampling took {end-start:.2f} seconds.")
        all_sampled = torch.cat(all_sampled, dim=0)
        all_sampled = all_sampled.cpu().numpy()
        all_sampled = all_sampled.astype(np.float16)  # this is ok because values are [-1,1]

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        outpath = self.output_dir / str(timestamp()) / "latent.npz"
        if not outpath.parent.exists():
            outpath.parent.mkdir(parents=True)

        np.savez(outpath, samples=all_sampled)
        print(f"Saved .npz file to {outpath} [shape={all_sampled.shape}].")

        with open(outpath.parent / "sample.txt", "w") as f:
            f.write("Sampling time: {:.2f} seconds.\n".format(end-start))

        return outpath
