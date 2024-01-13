#!/usr/bin/env python3

"""Trains Karras et al. (2022) diffusion models."""

import os
import argparse
import math
from copy import deepcopy
import json
from pathlib import Path

import einops
import accelerate
from accelerate import DistributedDataParallelKwargs
import safetensors.torch as safetorch
import torch
import torch._dynamo
from torch import distributed as dist
from torch import multiprocessing as mp
from torch import optim
from torch.utils import data
from tqdm.auto import tqdm, trange
import pandas as pd
import numpy as np
import wandb

import k_diffusion as K


def ensure_distributed():
    if not dist.is_initialized():
        dist.init_process_group(world_size=1, rank=0, store=dist.HashStore())


def main(args: K.config.TrainArgs):
    mp.set_start_method(args.start_method)
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch._dynamo.config.automatic_dynamic_shapes = False
    except AttributeError:
        pass
    torch.hub.set_dir(Path(args.artifacts_dir) / "torch_hub")

    # ==============================================================================
    # maybe set up configs resume
    # ==============================================================================
    debug_mode = args.debug_mode
    if args.name == "":
        args.name = wandb.util.generate_id()

    checkpoints_dir = Path(args.artifacts_dir) / "checkpoints" / args.name
    ckpt = None
    print("Saving to", checkpoints_dir)
    if not checkpoints_dir.exists():
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
    state_path = checkpoints_dir / "state.json"
    RESUME = state_path.exists() or args.resume

    if RESUME:
        if args.resume:
            ckpt_path = args.resume
        if not args.resume:
            state = json.load(open(state_path))
            ckpt_path = state["latest_checkpoint"]

        print(f"Resuming from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # load the previous config for consistency.
        # if a JSON config is saved, use it -- allows us to hack configs but use past checkpoints
        if (checkpoints_dir / "config.json").exists():
            argsdict = json.load(open(checkpoints_dir / "config.json"))
            args = K.config.dataclass_from_dict(K.config.TrainArgs, argsdict)
        args = K.config.dataclass_from_dict(K.config.TrainArgs, ckpt["config"])

    # save the config if it isn't already done
    config_path = checkpoints_dir / "config.json"
    if not config_path.exists():
        json.dump(K.config.dataclass_to_dict(args), open(config_path, "w"), indent=4)

    model_config = args.model_config
    dataset_config = args.dataset_config
    opt_config = args.opt_config
    sched_config = args.sched_config
    ema_sched_config = args.ema_sched_config

    # override some things in sample_config
    sample_config = args.sample_config
    sample_config.sigma_max = model_config.sigma_sample_density.max_value
    sample_config.sigma_min = model_config.sigma_sample_density.min_value
    sample_config.batch_size = args.batch_size
    sample_config.model_id = args.name

    # ==============================================================================
    # set distributed training
    # ==============================================================================
    ddp_kwargs = accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[ddp_kwargs],
    )
    ensure_distributed()
    device = accelerator.device
    unwrap = accelerator.unwrap_model
    print(f"Process {accelerator.process_index} using device: {device}", flush=True)

    if args.seed is not None:
        seeds = torch.randint(
            -(2**63),
            2**63 - 1,
            [accelerator.num_processes],
            generator=torch.Generator().manual_seed(args.seed),
        )
        torch.manual_seed(seeds[accelerator.process_index])
    demo_gen = torch.Generator().manual_seed(
        torch.randint(-(2**63), 2**63 - 1, ()).item()
    )

    # ==============================================================================
    # make models 
    # ==============================================================================
    inner_model = K.config.make_model(model_config, max_seq_len=args.max_seq_len)
    inner_model_ema = deepcopy(inner_model)
    num_parameters = K.utils.count_parameters(inner_model, require_grad_only=True)

    if args.compile:
        inner_model.compile()
        inner_model_ema.compile()

    if accelerator.is_main_process:
        accelerator.print(f"Number of trainable parameters: {num_parameters:,}")

    # models must be prepared before optimizer creation if using FSDP
    inner_model, inner_model_ema = accelerator.prepare(inner_model, inner_model_ema)

    # ==============================================================================
    # wandb 
    # ==============================================================================
    use_wandb = accelerator.is_main_process and not debug_mode
    if use_wandb:
        log_config = K.config.dataclass_to_dict(args)
        log_config["parameters"] = num_parameters
        wandb.init(
            resume="allow",
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            config=log_config,
            id=args.name,
            save_code=True,
        )

    # ==============================================================================
    # optimizer and EMA 
    # ==============================================================================
    lr = opt_config.lr
    groups = unwrap(inner_model).param_groups(lr)
    if opt_config.type == "adamw":
        opt = optim.AdamW(
            groups,
            lr=lr,
            betas=tuple(opt_config.betas),
            eps=opt_config.eps,
            weight_decay=opt_config.weight_decay,
        )
    elif opt_config.type == "sgd":
        opt = optim.SGD(
            groups,
            lr=lr,
            momentum=opt_config.momentum,
            nesterov=opt_config.nesterov,
            weight_decay=opt_config.weight_decay,
        )
    else:
        raise ValueError("Invalid optimizer type")

    assert ema_sched_config.type == "inverse"
    ema_sched = K.utils.EMAWarmup(
        power=ema_sched_config.power, max_value=ema_sched_config.max_value
    )
    ema_stats = {}

    # ==============================================================================
    # dataset and latent scaler 
    # ==============================================================================
    train_dl, _ = K.config.make_dataset(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_seq_len=args.max_seq_len,
        toy=debug_mode,
    )
    scaler = K.normalization.LatentScaler(
        mode=model_config.normalize_latent_by, origin_dataset=dataset_config.dataset, lm_embedder_type=model_config.lm_embedder_type
    )

    if dataset_config.dataset == "cath":
        num_samples = 35840  # can't do len of an iterable dataset, so write this here
        num_train_batches = math.ceil(num_samples / args.batch_size)
    else:
        num_train_batches = len(train_dl)

    sched = K.config.make_lr_sched(
        sched_config, opt, num_train_batches // args.grad_accum_steps
    )

    train_dl, opt = accelerator.prepare(train_dl, opt)

    # ==============================================================================
    # Post distribution configurations
    # ==============================================================================

    if use_wandb:
        wandb.watch(inner_model)
    if accelerator.num_processes == 1:
        args.gns = False
    if args.gns:
        gns_stats_hook = K.gns.DDPGradientStatsHook(inner_model)
        gns_stats = K.gns.GradientNoiseScale()
    else:
        gns_stats = None
    sample_density = K.config.make_sample_density(model_config, args.max_seq_len)

    model = K.config.make_denoiser_wrapper(model_config)(inner_model)
    model_ema = K.config.make_denoiser_wrapper(model_config)(inner_model_ema)

    if RESUME:
        assert not ckpt is None
        unwrap(model.inner_model).load_state_dict(ckpt["model"])
        unwrap(model_ema.inner_model).load_state_dict(ckpt["model_ema"])
        if opt_config.resume_from_saved_state:
            opt.load_state_dict(ckpt["opt"])
            sched.load_state_dict(ckpt["sched"])
            ema_sched.load_state_dict(ckpt["ema_sched"])

        ema_stats = ckpt.get("ema_stats", ema_stats)
        epoch = ckpt["epoch"] + 1
        step = ckpt["step"] + 1
        if args.gns and ckpt.get("gns_stats", None) is not None:
            gns_stats.load_state_dict(ckpt["gns_stats"])
        demo_gen.set_state(ckpt["demo_gen"])

        del ckpt
    else:
        epoch = 0
        step = 0

    if args.reset_ema:
        unwrap(model.inner_model).load_state_dict(
            unwrap(model_ema.inner_model).state_dict()
        )
        ema_sched = K.utils.EMAWarmup(
            power=ema_sched_config.power, max_value=ema_sched_config.max_value
        )
        ema_stats = {}

    if args.resume_inference:
        if accelerator.is_main_process:
            print(f"Loading {args.resume_inference}...")
        ckpt = safetorch.load_file(args.resume_inference)
        unwrap(model.inner_model).load_state_dict(ckpt)
        unwrap(model_ema.inner_model).load_state_dict(ckpt)
        del ckpt
    

    # ==============================================================================
    # sample / eval function 
    # ==============================================================================
    def sample(step):
        print("Instantiating sampler callback object...")
        should_we_log = sample_config.log_to_wandb and (not debug_mode)
        sample_config.model_step = step

        sampler = K.callback.SampleCallback(
            model=model,
            config=sample_config,
            model_config=model_config,
            is_wandb_setup=True, # skip setting up
            device=device,
        )

        # Load checkpoint from model_id and step
        # sample latent and calculate KID/FID to the saved known distribution
        print("Sampling latent...")
        sampled_latent, stats_dict = sampler.sample_latent(
            clip_range=sample_config.clip_range, save=True, log_wandb_stats=should_we_log, return_raw=False,
        )
        print(stats_dict)

        if not sample_config.n_to_construct == -1:
            sampled_latent = sampled_latent[torch.randperm(sampled_latent.shape[0])][:sample_config.n_to_construct]

        print("Constructing sequences...")
        _, _, strs, _ = sampler.construct_sequence(
            sampled_latent,
            calc_perplexity=sample_config.calc_perplexity,
            save_to_disk=sample_config.save_to_disk,
            log_to_wandb=should_we_log,
        )

        print("Calculating FID/KID...")
        fid, kid, _ = sampler.calculate_fid(
            sampled_latent, log_to_wandb=should_we_log
        )
        print("FID:", fid, "KID:", kid)
        

    # ==============================================================================
    # save function
    # ==============================================================================
    def save():
        accelerator.wait_for_everyone()
        filename = str(checkpoints_dir / f"{step:08}.pth")
        if accelerator.is_main_process:
            tqdm.write(f"Saving to {filename}...")
        inner_model = unwrap(model.inner_model)
        inner_model_ema = unwrap(model_ema.inner_model)
        obj = {
            "config": K.config.dataclass_to_dict(args),
            "model": inner_model.state_dict(),
            "model_ema": inner_model_ema.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "ema_sched": ema_sched.state_dict(),
            "epoch": epoch,
            "step": step,
            "gns_stats": gns_stats.state_dict() if gns_stats is not None else None,
            "ema_stats": ema_stats,
            "demo_gen": demo_gen.get_state(),
        }
        accelerator.save(obj, filename)
        if accelerator.is_main_process:
            state_obj = {"latest_checkpoint": filename}
            json.dump(state_obj, open(state_path, "w"))
        if args.wandb_save_model and use_wandb:
            wandb.save(filename)

    # ==============================================================================
    # main train loop
    # ==============================================================================
    if step > 0:
        print("Skipping batches...")
        for _ in trange(step):
            iter(train_dl)

    print("Starting training at step", step)
    losses_since_last_print = []
    pbar = tqdm(
        total=num_train_batches,
        initial=step,
        smoothing=0.1,
        disable=not accelerator.is_main_process,
        desc="Training",
    )

    def prepare_latent_from_fasta(batch):
        sequences = batch[1]
        x, mask = unwrap(model.inner_model).embed_from_sequences(
            sequences
        )  # (N, L, input_dim=1024)
        return x, mask

    def prepare_latent_from_shards(batch):
        x, seqlen = batch
        mask = (
            einops.repeat(
                torch.arange(x.shape[1], device=device)[None, :],
                "1 L -> N L",
                N=x.shape[0],
            )
            < seqlen[:, None]
        )
        return x, mask
    
    def run_batch(batch):
        pass

    try:
        while True:
            for batch in train_dl:
                with accelerator.accumulate(model):
                    if (
                        K.config.DATASET_TO_PATH[dataset_config.dataset]["loader"]
                        == "FastaDataset"
                    ):
                        x, mask = prepare_latent_from_fasta(batch)
                    elif (
                        K.config.DATASET_TO_PATH[dataset_config.dataset]["loader"]
                        == "ShardedTensorDataset"
                    ):
                        x, mask = prepare_latent_from_shards(batch)
                    else:
                        raise ValueError(f"Invalid dataset {dataset_config.dataset}")

                    # Normalize, maybe
                    x = scaler.scale(x)
                    extra_args, N = {"mask": mask}, x.shape[0]
                    noise = torch.randn_like(x)  # (N, L, d_model)

                    with K.utils.enable_stratified_accelerate(
                        accelerator, disable=args.gns
                    ):
                        sigma = sample_density([N], device=device)

                    with K.models.checkpointing(args.checkpointing):
                        losses, model_output = model.loss(
                            x, noise, sigma, mask, return_model_output=True, **extra_args
                        )
                    loss = accelerator.gather(losses).mean().item()
                    model_output = accelerator.gather(model_output)
                    mean, std = model_output.mean(), model_output.std()
                    losses_since_last_print.append(loss)
                    accelerator.backward(losses.mean())
                    if args.gns:
                        (
                            sq_norm_small_batch,
                            sq_norm_large_batch,
                        ) = gns_stats_hook.get_stats()
                        gns_stats.update(
                            sq_norm_small_batch,
                            sq_norm_large_batch,
                            N,
                            N * accelerator.num_processes,
                        )
                    if args.clip_norm:
                        accelerator.clip_grad_norm_(model.parameters(), args.clip_norm)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)

                    opt.step()
                    sched.step()
                    opt.zero_grad()

                    ema_decay = ema_sched.get_value()
                    K.utils.ema_update_dict(
                        ema_stats,
                        {"loss": loss, "step": step, "epoch": epoch},
                        ema_decay ** (1 / args.grad_accum_steps),
                    )
                    if accelerator.sync_gradients:
                        K.utils.ema_update(model, model_ema, ema_decay)
                        ema_sched.step()

                del batch, x, mask, noise, sigma, model_output

                if step % 25 == 0:
                    loss_disp = sum(losses_since_last_print) / len(
                        losses_since_last_print
                    )
                    losses_since_last_print.clear()
                    avg_loss = ema_stats["loss"]
                    if accelerator.is_main_process:
                        if args.gns:
                            tqdm.write(
                                f"Epoch: {epoch}, step: {step}, loss: {loss_disp:g}, avg loss: {avg_loss:g}, gns: {gns_stats.get_gns():g}"
                            )
                        else:
                            tqdm.write(
                                f"Epoch: {epoch}, step: {step}, loss: {loss_disp:g}, avg loss: {avg_loss:g}"
                            )
                if step % args.log_every == 0:
                    if use_wandb:
                        log_dict = {
                            "epoch": epoch,
                            "step": step,
                            "loss": loss,
                            "lr": sched.get_last_lr()[0],
                            "ema_decay": ema_decay,
                            "pred_mean": mean,
                            "pred_std": std,
                        }
                        if args.gns:
                            log_dict["gradient_noise_scale"] = gns_stats.get_gns()
                        wandb.log(log_dict)

                if step == args.end_step or (step > 0 and step % args.save_every == 0):
                    save()
                
                if step > 0 and step % args.sample_every == 0:
                    sample(step)

                if step == args.end_step:
                    if accelerator.is_main_process:
                        tqdm.write("Done!")
                    return
                step += 1
                pbar.update(1)
            epoch += 1

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    import tyro
    from k_diffusion.config import TrainArgs

    args = tyro.cli(TrainArgs)

    if args.debug_mode:
        try:
            main(args)
        except:
            import pdb, sys, traceback

            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
    else:
        main(args)
