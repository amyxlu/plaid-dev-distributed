from pathlib import Path
from omegaconf import OmegaConf
from dataclasses import dataclass, asdict

from tqdm import trange
import torch
import numpy as np

from plaid.diffusion import FunctionOrganismDiffusion
from plaid.denoisers import FunctionOrganismDiT, FunctionOrganismUDiT, DenoiserKwargs
from plaid.constants import COMPRESSION_INPUT_DIMENSIONS
from plaid.datasets import NUM_ORGANISM_CLASSES, NUM_FUNCTION_CLASSES
from plaid.utils import timestamp


device = torch.device("cuda")


@dataclass
class SampleConfig:
    # model setup
    model_id: str = "4hdab8dn"
    model_ckpt_dir: str = "/data/lux70/plaid/checkpoints/plaid-compositional"

    # sampling setup
    organism_idx: int = NUM_ORGANISM_CLASSES
    function_idx: int = NUM_FUNCTION_CLASSES
    cond_scale: float = 7
    num_samples: int = -1
    beta_scheduler_name: str = "sigmoid"
    sampling_timesteps: int = 1000 
    batch_size: int = 2048 
    length: int = 32  # the final length, after decoding back to structure/sequence, is twice this value
    return_all_timesteps: bool = False

    # output setup
    output_root_dir: str = "/data/lux70/plaid/artifacts/samples"


def create_diffusion(
    config_path,
    model_path,
    override_beta_scheduler_name=None,
    override_sampling_timesteps=None
):
    cfg = OmegaConf.load(config_path)
    compression_model_id = cfg['compression_model_id']
    # shorten_factor = COMPRESSION_SHORTEN_FACTORS[compression_model_id]
    input_dim = COMPRESSION_INPUT_DIMENSIONS[compression_model_id]

    denoiser_kwargs = cfg.denoiser
    denoiser_kwargs.pop("_target_")
    denoiser = FunctionOrganismDiT(**denoiser_kwargs, input_dim=input_dim)

    # lask.ckpt automatically links to the EMA
    ckpt = torch.load(model_path)

    mod_state_dict = {}
    for k, v in ckpt['state_dict'].items():
        if k[:16] == "model._orig_mod.":
            mod_state_dict[k[16:]] = v

    denoiser.load_state_dict(mod_state_dict)
    diffusion_kwargs = cfg.diffusion
    diffusion_kwargs.pop("_target_")

    if override_beta_scheduler_name is not None:
        diffusion_kwargs["beta_scheduler_name"] = override_beta_scheduler_name
    
    if override_sampling_timesteps is not None:
        diffusion_kwargs["sampling_timesteps"] = override_sampling_timesteps

    diffusion = FunctionOrganismDiffusion(model=denoiser,**diffusion_kwargs)

    return diffusion


def sample(cfg: SampleConfig, diffusion: FunctionOrganismDiffusion):
    N, L, C = cfg.batch_size, cfg.length, diffusion.model.input_dim 
    shape = (N, L, C)

    sampled_latent = diffusion.ddim_sample_loop(
        shape=shape,
        organism_idx=cfg.organism_idx,
        function_idx=cfg.function_idx,
        return_all_timesteps=cfg.return_all_timesteps, 
        cond_scale=cfg.cond_scale
    )
    return sampled_latent


def main(cfg):
    model_id = cfg.model_id
    ckpt_dir = Path(cfg.model_ckpt_dir)
    model_path = ckpt_dir / model_id / "last.ckpt"
    config_path = ckpt_dir / model_id / "config.yaml"

    output_dir = Path(cfg.output_root_dir) / model_id / f"f{cfg.function_idx}_o{cfg.organism_idx}"
    device = torch.device("cuda")

    diffusion = create_diffusion(
        config_path,
        model_path,
        override_sampling_timesteps=cfg.sampling_timesteps,
        override_beta_scheduler_name=cfg.beta_scheduler_name,
    )
    diffusion = diffusion.to(device)

    num_samples = max(cfg.num_samples, cfg.batch_size)
    all_sampled = []
    cur_n_sampled = 0

    for _ in trange(0, num_samples, cfg.batch_size, desc="Sampling batches:"):
        sampled_latent = sample(cfg, diffusion)
        all_sampled.append(sampled_latent)
        cur_n_sampled += cfg.batch_size

    all_sampled = torch.cat(all_sampled, dim=0)
    all_sampled = all_sampled.cpu().numpy()
    all_sampled = all_sampled.astype(np.float16)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    timestamp_str = timestamp()
    outpath = output_dir / f"{timestamp_str}.npz"
    np.savez(outpath, samples=all_sampled)
    print(f"Saved .npz file to {outpath} [shape={all_sampled.shape}].")
    return outpath

# def post_rm(outpath, newpath):
#     import shutil
#     shutil.copy(outpath, newpath)
#     shutil.rmtree(Path(outpath).parent, ignore_errors=True)

if __name__ == "__main__":
    import tyro
    cfg = tyro.cli(SampleConfig) 
    outpath = main(cfg)
    # post_rm(outpath, "/mp/lux70/plaid/artifacts/samples"