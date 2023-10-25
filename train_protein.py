#!/usr/bin/env python3

"""Trains Karras et al. (2022) diffusion models."""

import os
import argparse
import math
from copy import deepcopy
import json
from pathlib import Path

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

import k_diffusion as K


def ensure_distributed():
    if not dist.is_initialized():
        dist.init_process_group(world_size=1, rank=0, store=dist.HashStore())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--artifacts-dir', type=str, default='artifacts',
                   help='where artifacts will be saved. should have subdirectories `checkpoints` and `sampled`.')
    p.add_argument('--batch-size', type=int, default=64,
                   help='the batch size')
    p.add_argument('--checkpointing', action='store_true',
                   help='enable gradient checkpointing')
    p.add_argument('--compile', action='store_true',
                   help='compile the model')
    p.add_argument('--config', type=str, required=True,
                   help='the configuration file')
    p.add_argument('--demo-every', type=int, default=0,
                   help='save a demo grid every this many steps')
    p.add_argument('--end-step', type=int, default=None,
                   help='the step to end training at')
    p.add_argument('--embedding-n', type=int, help="number of embeddings to save")
    p.add_argument('--evaluate-every', type=int, default=1000,
                   help='save a demo grid every this many steps')
    p.add_argument('--evaluate-with', type=str, default='esmfold_embed',
                   choices=['esmfold_embed'],
                   help='the feature extractor to use for evaluation')
    p.add_argument('--gns', action='store_true',
                   help='measure the gradient noise scale (DDP only, disables stratified sampling)')
    p.add_argument('--grad-accum-steps', type=int, default=1,
                   help='the number of gradient accumulation steps')
    p.add_argument('--lr', type=float,
                   help='the learning rate')
    p.add_argument('--mixed-precision', type=str,
                   help='the mixed precision type')
    p.add_argument('--name', type=str, default='model',
                   help='the name of the run')
    p.add_argument('--num-workers', type=int, default=8,
                   help='the number of data loader workers')
    p.add_argument("--recycling-n", type=int, default=4,
                   help="the number of recycles to use when applying the frozen structure head")
    p.add_argument('--reset-ema', action='store_true',
                   help='reset the EMA')
    p.add_argument('--resume', type=str,
                   help='the checkpoint to resume from')
    p.add_argument('--resume-inference', type=str,
                   help='the inference checkpoint to resume from')
    p.add_argument('--sample-n', type=int, default=64,
                   help='the number of images to sample for demo grids')
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--save-every', type=int, default=1000,
                   help='save checkpoint every this many steps')
    p.add_argument('--seed', type=int,
                   help='the random seed')
    p.add_argument('--start-method', type=str, default='spawn',
                   choices=['fork', 'forkserver', 'spawn'],
                   help='the multiprocessing start method')
    p.add_argument('--toy', action='store_true',
                   help='use a toy dataset')
    p.add_argument('--wandb-entity', type=str, default="amyxlu",
                   help='the wandb entity name')
    p.add_argument('--wandb-group', type=str,
                   help='the wandb group name')
    p.add_argument('--wandb-project', type=str,
                   help='the wandb project name (specify this to enable wandb)')
    p.add_argument('--wandb-save-model', action='store_true',
                   help='save model to wandb')
    args = p.parse_args()

    mp.set_start_method(args.start_method)
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch._dynamo.config.automatic_dynamic_shapes = False
    except AttributeError:
        pass

    config = K.config.load_config(args.config)
    model_config = config['model']
    dataset_config = config['dataset']
    opt_config = config['optimizer']
    sched_config = config['lr_sched']
    ema_sched_config = config['ema_sched']
    seq_len = model_config['input_size']

    # ==============================================================================
    # initialize objects and set up distributed training
    # ==============================================================================
    ddp_kwargs = accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.grad_accum_steps, mixed_precision=args.mixed_precision, kwargs_handlers=[ddp_kwargs])
    ensure_distributed()
    device = accelerator.device
    unwrap = accelerator.unwrap_model
    print(f'Process {accelerator.process_index} using device: {device}', flush=True)

    if args.seed is not None:
        seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes], generator=torch.Generator().manual_seed(args.seed))
        torch.manual_seed(seeds[accelerator.process_index])
    demo_gen = torch.Generator().manual_seed(torch.randint(-2 ** 63, 2 ** 63 - 1, ()).item())

    inner_model = K.config.make_model(config)
    inner_model_ema = deepcopy(inner_model)
    num_parameters = K.utils.count_parameters(inner_model, require_grad_only=True)

    if args.compile:
        inner_model.compile()
        inner_model_ema.compile()

    if accelerator.is_main_process:
        accelerator.print(f'Number of trainable parameters: {num_parameters:,}')

    # models must be prepared before optimizer creation if using FSDP
    inner_model, inner_model_ema = accelerator.prepare(inner_model, inner_model_ema)

    # If logging to wandb, initialize the run
    use_wandb = accelerator.is_main_process and args.wandb_project
    if use_wandb:
        import wandb
        log_config = vars(args)
        log_config['config'] = config
        log_config['parameters'] = num_parameters 
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group, config=log_config, save_code=True)
        if args.name == 'model':
            args.name = wandb.run.id
    artifacts_dir = Path(args.artifacts_dir) / args.name
    checkpoints_dir = artifacts_dir / "checkpoints"
    sampled_dir = artifacts_dir / "sampled"
    project_home_dir = Path(os.environ["KD_PROJECT_HOME"])

    for p in (artifacts_dir, checkpoints_dir, sampled_dir):
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)

    lr = opt_config['lr'] if args.lr is None else args.lr
    # groups = inner_model.param_groups(lr)
    groups = unwrap(inner_model).param_groups(lr)
    if opt_config['type'] == 'adamw':
        opt = optim.AdamW(groups,
                          lr=lr,
                          betas=tuple(opt_config['betas']),
                          eps=opt_config['eps'],
                          weight_decay=opt_config['weight_decay'])
    elif opt_config['type'] == 'sgd':
        opt = optim.SGD(groups,
                        lr=lr,
                        momentum=opt_config.get('momentum', 0.),
                        nesterov=opt_config.get('nesterov', False),
                        weight_decay=opt_config.get('weight_decay', 0.))
    else:
        raise ValueError('Invalid optimizer type')

    if sched_config['type'] == 'inverse':
        sched = K.utils.InverseLR(opt,
                                  inv_gamma=sched_config['inv_gamma'],
                                  power=sched_config['power'],
                                  warmup=sched_config['warmup'])
    elif sched_config['type'] == 'exponential':
        sched = K.utils.ExponentialLR(opt,
                                      num_steps=sched_config['num_steps'],
                                      decay=sched_config['decay'],
                                      warmup=sched_config['warmup'])
    elif sched_config['type'] == 'constant':
        sched = K.utils.ConstantLRWithWarmup(opt, warmup=sched_config['warmup'])
    else:
        raise ValueError('Invalid schedule type')

    assert ema_sched_config['type'] == 'inverse'
    ema_sched = K.utils.EMAWarmup(power=ema_sched_config['power'],
                                  max_value=ema_sched_config['max_value'])
    ema_stats = {}
    
    # dataset specification
    if dataset_config['type'] == "uniref":
        from torch.utils.data import random_split
        from evo.dataset import FastaDataset
        fasta_file = dataset_config['path']
        if args.toy:
            fasta_file = dataset_config['toy_data_path']
        
        ds = FastaDataset(fasta_file, cache_indices=True)
        n_val = int(dataset_config['num_holdout'])
        n_train = len(ds) - n_val  # 153,726,820
        train_set, val_set = random_split(
            ds, [n_train, n_val], generator=torch.Generator().manual_seed(int(dataset_config['random_split_seed']))
        )
    else:
        raise ValueError('Invalid dataset type')

    if accelerator.is_main_process:
        try:
            print(f'Number of items in dataset: {len(train_set):,}')
        except TypeError:
            pass

    num_classes = dataset_config.get('num_classes', 0)
    cond_dropout_rate = dataset_config.get('cond_dropout_rate', 0.1)
    class_key = dataset_config.get('class_key', 1)

    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True, drop_last=True,
                               num_workers=args.num_workers, persistent_workers=True)

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
    sigma_min = model_config['sigma_min']
    sigma_max = model_config['sigma_max']
    sample_density = K.config.make_sample_density(model_config)

    model = K.config.make_denoiser_wrapper(config)(inner_model)
    model_ema = K.config.make_denoiser_wrapper(config)(inner_model_ema)

    state_path = artifacts_dir / 'state.json'

    if state_path.exists() or args.resume:
        if args.resume:
            ckpt_path = args.resume
        if not args.resume:
            state = json.load(open(state_path))
            ckpt_path = state['latest_checkpoint']
        if accelerator.is_main_process:
            print(f'Resuming from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        unwrap(model.inner_model).load_state_dict(ckpt['model'])
        unwrap(model_ema.inner_model).load_state_dict(ckpt['model_ema'])
        opt.load_state_dict(ckpt['opt'])
        sched.load_state_dict(ckpt['sched'])
        ema_sched.load_state_dict(ckpt['ema_sched'])
        ema_stats = ckpt.get('ema_stats', ema_stats)
        epoch = ckpt['epoch'] + 1
        step = ckpt['step'] + 1
        if args.gns and ckpt.get('gns_stats', None) is not None:
            gns_stats.load_state_dict(ckpt['gns_stats'])
        demo_gen.set_state(ckpt['demo_gen'])

        del ckpt
    else:
        epoch = 0
        step = 0

    if args.reset_ema:
        unwrap(model.inner_model).load_state_dict(unwrap(model_ema.inner_model).state_dict())
        ema_sched = K.utils.EMAWarmup(power=ema_sched_config['power'],
                                      max_value=ema_sched_config['max_value'])
        ema_stats = {}

    if args.resume_inference:
        if accelerator.is_main_process:
            print(f'Loading {args.resume_inference}...')
        ckpt = safetorch.load_file(args.resume_inference)
        unwrap(model.inner_model).load_state_dict(ckpt)
        unwrap(model_ema.inner_model).load_state_dict(ckpt)
        del ckpt

    def save():
        accelerator.wait_for_everyone()
        filename = str(checkpoints_dir / f'{step:08}.pth')
        if accelerator.is_main_process:
            tqdm.write(f'Saving to {filename}...')
        inner_model = unwrap(model.inner_model)
        inner_model_ema = unwrap(model_ema.inner_model)
        obj = {
            'config': config,
            'model': inner_model.state_dict(),
            'model_ema': inner_model_ema.state_dict(),
            'opt': opt.state_dict(),
            'sched': sched.state_dict(),
            'ema_sched': ema_sched.state_dict(),
            'epoch': epoch,
            'step': step,
            'gns_stats': gns_stats.state_dict() if gns_stats is not None else None,
            'ema_stats': ema_stats,
            'demo_gen': demo_gen.get_state(),
        }
        accelerator.save(obj, filename)
        if accelerator.is_main_process:
            state_obj = {'latest_checkpoint': filename}
            json.dump(state_obj, open(state_path, 'w'))
        if args.wandb_save_model and use_wandb:
            wandb.save(filename)

    # ==============================================================================
    # set up "frechet esmfold distnace" evaluation 
    # ==============================================================================
    evaluate_enabled = False
    demo_enabled = False
    if args.evaluate_with == "esmfold_embed":
        extractor = K.evaluation.ESMFoldLatentFeatureExtractor(unwrap(inner_model).esmfold_embedder, device=device)
    else:
        raise ValueError('Invalid evaluation feature extractor')
    if accelerator.is_main_process:
        accelerator.print('Loading cached ESMFold features for 50,000 real holdout proteins...')
        cache_location = project_home_dir / "cached_tensors" / "holdout_esmfold_feats.st"
        reals_features = extractor.load_saved_features(cache_location, device="cpu") #, device=device)

    @torch.no_grad()
    @K.utils.eval_mode(model_ema)
    def evaluate():
        if accelerator.is_main_process:
            tqdm.write('Evaluating...')
        sigmas = K.sampling.get_sigmas_karras(args.evaluate_n, sigma_min, sigma_max, rho=7., device=device)
        
        def sample_fn(n):
            x = torch.randn([n, seq_len, model_config['d_model']], device=device) * sigma_max
            model_fn, extra_args = model_ema, {"mask": torch.ones(n, seq_len, device=device).long()}
            x_0 = K.sampling.sample_dpmpp_2m_sde(model_fn, x, sigmas, extra_args=extra_args, eta=0.0, solver_type='heun', disable=True)
            del x

            # 2) Downproject latent space, maybe
            if model_config['d_model'] != model_config['input_dim']:
                x_0 = unwrap(model_ema.inner_model).project_to_input_dim(x_0)

            # 1) Normalize, maybe
            x_0 = K.normalization.undo_scale_embedding(x_0, model_config['normalize_latent_by'])
            return x_0
        
        # fakes_features = K.evaluation.compute_features(accelerator, sample_fn, lambda x: x.mean(dim=1), args.evaluate_n, args.batch_size)
        sampled = K.evaluation.compute_features(accelerator, sample_fn, lambda x: x, args.evaluate_n, args.batch_size)
        np.save(artifacts_dir / "sampled" / f"step{step}_sampled.pkl.npy", sampled.cpu().numpy(), allow_pickle=True)
        fakes_features = sampled.mean(dim=1)

        if accelerator.is_main_process:
            fid = K.evaluation.fid(fakes_features, reals_features.to(device))
            kid = K.evaluation.kid(fakes_features, reals_features.to(device))
            print(f'FID: {fid.item():g}, KID: {kid.item():g}')
            # if accelerator.is_main_process:
            #     metrics_log.write(step, ema_stats['loss'], fid.item(), kid.item())
            if use_wandb:
                wandb.log({'FID': fid.item(), 'KID': kid.item()}, step=step)
                # wandb.log({f"generated_embedding_step{step}": wandb.Table(dataframe=pd.DataFrame(fakes_features.cpu().numpy()).sample(args.embedding_n))})

    # ==============================================================================
    # main train loop 
    # ==============================================================================
    losses_since_last_print = []

    try:
        while True:
            for batch in tqdm(train_dl, smoothing=0.1, disable=not accelerator.is_main_process, desc="Training"):
                with accelerator.accumulate(model):
                    sequences = batch[1]
                    x, mask = unwrap(model.inner_model).embed_from_sequences(sequences)  # (N, L, input_dim=1024)

                    # 1) Normalize, maybe
                    K.normalization.scale_embedding(x, model_config['normalize_latent_by'])

                    # 2) Downproject latent space, maybe
                    if model_config['d_model'] != model_config['input_dim']:
                        x = unwrap(model.inner_model).project_to_d_model(x)  # (N, L, d_model)
                    
                    class_cond, extra_args, N = None, {'mask': mask}, x.shape[0]
                    # if num_classes:
                        # class_cond = batch[class_key]
                        # drop = torch.rand(class_cond.shape, device=class_cond.device)
                        # class_cond.masked_fill_(drop < cond_dropout_rate, num_classes)
                        # extra_args['class_cond'] = class_cond
                    noise = torch.randn_like(x)  # (N, L, d_model)

                    with K.utils.enable_stratified_accelerate(accelerator, disable=args.gns):
                        sigma = sample_density([N], device=device)
                    with K.models.checkpointing(args.checkpointing):
                        losses, model_output = model.loss(x, noise, sigma, return_model_output=True, **extra_args)
                    loss = accelerator.gather(losses).mean().item()
                    model_output = accelerator.gather(model_output)
                    mean, std = model_output.mean(), model_output.std()
                    losses_since_last_print.append(loss)
                    accelerator.backward(losses.mean())
                    if args.gns:
                        sq_norm_small_batch, sq_norm_large_batch = gns_stats_hook.get_stats()
                        gns_stats.update(sq_norm_small_batch, sq_norm_large_batch, N, N * accelerator.num_processes)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.)
                    opt.step()
                    sched.step()
                    opt.zero_grad()

                    ema_decay = ema_sched.get_value()
                    K.utils.ema_update_dict(ema_stats, {'loss': loss}, ema_decay ** (1 / args.grad_accum_steps))
                    if accelerator.sync_gradients:
                        K.utils.ema_update(model, model_ema, ema_decay)
                        ema_sched.step()
                
                del batch, x, mask, noise, sigma, model_output

                if step % 25 == 0:
                    loss_disp = sum(losses_since_last_print) / len(losses_since_last_print)
                    losses_since_last_print.clear()
                    avg_loss = ema_stats['loss']
                    if accelerator.is_main_process:
                        if args.gns:
                            tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss_disp:g}, avg loss: {avg_loss:g}, gns: {gns_stats.get_gns():g}')
                        else:
                            tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss_disp:g}, avg loss: {avg_loss:g}')
                if step % args.log_every == 0:
                    if use_wandb:
                        log_dict = {
                            'epoch': epoch,
                            'loss': loss,
                            'lr': sched.get_last_lr()[0],
                            'ema_decay': ema_decay,
                            'pred_mean': mean,
                            'pred_std': std
                        }
                        if args.gns:
                            log_dict['gradient_noise_scale'] = gns_stats.get_gns()
                        wandb.log(log_dict, step=step)

                # if demo_enabled and step % args.demo_every == 0:
                #     demo()

                if step == args.end_step or (step > 0 and step % args.save_every == 0):
                    save()

                if evaluate_enabled and step > 0 and step % args.evaluate_every == 0:
                    evaluate()

                if step == args.end_step:
                    if accelerator.is_main_process:
                        tqdm.write('Done!')
                    return
                step += 1
            epoch += 1

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
