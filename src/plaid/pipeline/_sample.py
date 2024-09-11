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


class SampleLatent:
    def __init__(
        self,
        # model setup
        model_id: str = "4hdab8dn",
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

        # output setup
        output_root_dir: str = "/data/lux70/plaid/artifacts/samples",
    ):
        self.model_id = model_id
        self.model_ckpt_dir = Path(model_ckpt_dir)
        self.organism_idx = organism_idx
        self.function_idx = function_idx
        self.cond_scale = cond_scale
        self.num_samples = num_samples
        self.beta_scheduler_name = beta_scheduler_name
        self.sampling_timesteps = sampling_timesteps
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

        # set up diffusion model
        self.diffusion = self.create_diffusion(config_path, model_path)
        self.diffusion = self.diffusion.to(self.device)

    def create_diffusion(
        self,
        config_path,
        model_path,
    ):
        cfg = OmegaConf.load(config_path)
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
        diffusion_kwargs = cfg.diffusion
        diffusion_kwargs.pop("_target_")

        diffusion_kwargs["sampling_timesteps"] = self.sampling_timesteps 
        if self.beta_scheduler_name is not None:
            diffusion_kwargs["beta_scheduler_name"] = self.beta_scheduler_name

        diffusion = FunctionOrganismDiffusion(model=denoiser,**diffusion_kwargs)
        return diffusion

    def sample(self):
        N, L, C = self.batch_size, self.length, self.diffusion.model.input_dim 
        shape = (N, L, C)

        sampled_latent = self.diffusion.ddim_sample_loop(
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

        outpath = self.output_dir / f"{str(timestamp())}.npz"
        np.savez(outpath, samples=all_sampled)
        print(f"Saved .npz file to {outpath} [shape={all_sampled.shape}].")

        with open(self.output_dir / "sample.log", "w") as f:
            f.write("Sampling time: {:.2f} seconds.\n".format(end-start))

        return outpath
