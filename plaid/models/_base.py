import einops
import torch
import torch._dynamo
from torch import optim
from tqdm.auto import tqdm, trange
import pandas as pd
import numpy as np
import wandb

import lightning as L


class BaseDiffusionModel(L.LightningModule):
    def __init__(
        self,
        debug_mode: bool = False,
        name: str = "",
        lr: float = 1e-4,
        *args,
        **kwargs
    ):        
        # persists as specified on CLI even if resuming
        if name == "":
            self.name = wandb.util.generate_id()

        self.setup_seed()
        self.setup_denoiser()
        self.setup_optimizer()
        self.setup_sample_callback()

    def setup_seed(self):
        if self.args.seed is not None:
            seeds = torch.randint(
                -(2**63),
                2**63 - 1,
                [self.accelerator.num_processes],
                generator=torch.Generator().manual_seed(self.args.seed),
            )
            torch.manual_seed(seeds[self.accelerator.process_index])
        self.demo_gen = torch.Generator().manual_seed(
            torch.randint(-(2**63), 2**63 - 1, ()).item()
        )
    
    def setup_sample_callback(self):
        self.sample_callback = K.callback.SampleCallback(
            model=self.model,
            config=self.sample_config,
            model_config=self.model_config,
            origin_dataset=self.dataset_config.dataset,
            is_wandb_setup=self.debug_mode,  # If debug mode, don't trigger wandb setup
        )

    def setup_optimizer(self, parameters):
        # TODO: make sure this works
        lr = self.opt_config.lr
        if self.model_config.type == "bert_hf":
            groups = self.unwrap(self.inner_model).parameters()
        else:
            if self.args.optimize_by_group:
                groups = self.unwrap(self.inner_model).param_groups(lr)
            else:
                groups = self.unwrap(self.inner_model).parameters()

        if self.opt_config.type == "adamw":
            self.opt = optim.AdamW(
                groups,
                lr=lr,
                betas=tuple(self.opt_config.betas),
                eps=self.opt_config.eps,
                weight_decay=self.opt_config.weight_decay,
            )
        elif self.opt_config.type == "sgd":
            self.opt = optim.SGD(
                groups,
                lr=lr,
                momentum=self.opt_config.momentum,
                nesterov=self.opt_config.nesterov,
                weight_decay=self.opt_config.weight_decay,
            )
        else:
            raise ValueError("Invalid optimizer type")

        self.sched = K.config.make_lr_sched(
            self.sched_config,
            self.opt,
            self.num_train_batches // self.args.grad_accum_steps,
        )

    def prepare_latent_from_fasta(self, batch):
        sequences = batch[1]
        x, mask = self.unwrap(self.model.inner_model).embed_from_sequences(
            sequences, self.dataset_config.max_seq_len, self.dataset_config.min_seq_len
        )  # (N, L, input_dim=1024)
        return x, mask

    def prepare_latent_from_shards(self, batch):
        x, seqlen = batch
        mask = torch.arange(x.shape[1], device=self.device)
        mask = (
            einops.repeat(mask[None, :], "1 L -> N L", N=x.shape[0]) < seqlen[:, None]
        )
        return x, mask

    def sample_eval(self):
        sampled_latent = self.sample_callback.sample_latent(
            clip_range=self.sample_config.clip_range,
            save=self.sample_config.save_to_disk,
            log_wandb_stats=self.sample_config.log_to_wandb,
            return_raw=False,
        )

        if self.sample_config.calc_fid:
            print("Calculating FID/KID...")
            fid, kid = self.sample_callback.calculate_fid(
                sampled_latent, log_to_wandb=self.sample_config.log_to_wandb
            )

        # potentially take a smaller subset to decode into structure/sequence and evaluate
        if not self.sample_config.n_to_construct == -1:
            sampled_latent = sampled_latent[torch.randperm(sampled_latent.shape[0])][
                :self.sample_config.n_to_construct
            ]
        
        print("Constructing sequences...")
        _, _, strs, _ = self.sample_callback.construct_sequence(
            sampled_latent,
            calc_perplexity=self.sample_config.calc_perplexity,
            save_to_disk=self.sample_config.save_to_disk,
            log_to_wandb=self.sample_config.log_to_wandb,
        )

        print("Constructing structures...")
        _, metrics, _ = self.sample_config.construct_structure(
            sampled_latent,
            strs,
            save_to_disk=self.sample_config.save_to_disk,
            log_to_wandb=self.sample_config.log_to_wandb,
        )

    def setup_denoiser(self):
        raise NotImplementedError

    def run_batch(self, batch):
        raise NotImplementedError
    