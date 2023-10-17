#!/usr/bin/env python3

"""Samples from k-diffusion models."""

import argparse
from pathlib import Path

import accelerate

# import safetensors.torch as safetorch
import torch
from tqdm import trange, tqdm

import k_diffusion as K
from k_diffusion.models.esmfold import ESMFOLD_S_DIM
from k_diffusion.proteins import LatentToSequence, LatentToStructure

from dataclasses import dataclass
from pathlib import Path


# @dataclass
# class SampleProteinArgs:
#     batch_size: int = 64
#     checkpoint: Path = Path("/home/amyxlu/kdiffusion/prot_karras_2_00000010.pth")
#     config: Path = Path(
#         "/home/amyxlu/kdiffusion/configs/config_protein_transformer_v1.json"
#     )
#     n: int = 64
#     prefix: str = "out"
#     steps: int = 50


# args = SampleProteinArgs()


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--batch-size", type=int, default=64, help="the batch size")
    p.add_argument(
        "--checkpoint", type=Path, required=True, help="the checkpoint to use"
    )
    p.add_argument("--config", type=Path, help="the model config")
    p.add_argument("-n", type=int, default=64, help="the number of images to sample")
    p.add_argument("--prefix", type=str, default="out", help="the output prefix")
    p.add_argument(
        "--steps", type=int, default=50, help="the number of denoising steps"
    )
    p.add_argument(
        "--num_recycles",
        type=int,
        default=4,
        help="the number of recycles to use when applying the frozen structure head",
    )
    args = p.parse_args()

    config = K.config.load_config(args.config if args.config else args.checkpoint)
    model_config = config["model"]
    size = model_config["input_size"]

    accelerator = accelerate.Accelerator()
    unwrap = accelerator.unwrap_model
    device = accelerator.device
    print("Using device:", device, flush=True)

    inner_model = K.config.make_model(config).eval().requires_grad_(False).to(device)
    # inner_model.load_state_dict(safetorch.load_file(args.checkpoint))
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    unwrap(inner_model).load_state_dict(ckpt["model"])
    accelerator.print("Parameters:", K.utils.n_params(inner_model))
    model = K.Denoiser(inner_model, sigma_data=model_config["sigma_data"])

    sigma_min = model_config["sigma_min"]
    sigma_max = model_config["sigma_max"]

    @torch.no_grad()
    @K.utils.eval_mode(model)
    def run(construct_sequence: bool = False, construct_structure: bool = False):
        if accelerator.is_local_main_process:
            tqdm.write("Sampling...")
        sigmas = K.sampling.get_sigmas_karras(
            args.steps, sigma_min, sigma_max, rho=7.0, device=device
        )

        device = accelerator.device


        def sample_fn(n):
            x = torch.randn([n, size, ESMFOLD_S_DIM], device=device) * sigma_max
            mask = torch.ones([n, size], device=device, dtype=torch.int)
            x_0 = K.sampling.sample_lms(
                model,
                x,
                sigmas,
                extra_args={"mask": mask},
                disable=not accelerator.is_local_main_process,
            )
            return x_0

        x_0 = sample_fn(args.n)

        if construct_sequence:
            sequence_constructor = LatentToSequence(device=device)
            sequence_probs, sequence_idx, sequence_str = sequence_constructor.to_sequence(
                x_0
            )
        
        if construct_structure:
            structure_constructor = LatentToStructure(device=device)
            pdb_strs, metrics = structure_constructor.to_structure(
                x_0,
                sequences=sequence_str,
                num_recycles=args.num_recycles,
                batch_size=args.batch_size,
            )


        # TODO: log? save? calculate metrics?
        # x_0 = K.evaluation.compute_features(
        #     accelerator, sample_fn, lambda x: x, args.n, args.batch_size
        # )
        # return x_0
        # if accelerator.is_main_process:
        #     filename = f"{args.prefix}.pkl"

        #     for i, out in enumerate(x_0):
        # filename = f'{args.prefix}_{i:05}.png'
        # K.utils.to_pil_image(out).save(filename)
        return x_0

    try:
        run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
