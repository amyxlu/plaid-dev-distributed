from functools import partial
import json
import math
from pathlib import Path

from dataclasses import dataclass, field, is_dataclass, make_dataclass
from typing import Optional, Dict, List

from jsonmerge import merge
from transformers import (
    get_scheduler,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)

from . import augmentation, layers, models, utils


def dataclass_to_dict(obj):
    outdict = {}
    for k, v in obj.__dict__.items(): 
        if is_dataclass(v):
            outdict[k] = dataclass_to_dict(v)
        else:
            outdict[k] = v
    return outdict


@dataclass
class SigmaDensityConfig:
    type: str = "cosine-interpolated"
    mean: Optional[float] = None
    std: Optional[float] = None
    loc: Optional[float] = None
    scale: Optional[float] = None
    min_value: float = 1e-3
    max_value: float = 1e3
    noise_d_low: int = 32


@dataclass
class ModelConfig:
    type: str = "protein_transformer_v1"
    n_layers: int = 12
    d_model: int = 512
    d_ff: int = 256
    d_head: int = 128
    input_size: int = 512
    input_dim: int = 1024
    min_len: int = 32
    num_classes: int = 0
    dropout: float = 0.0
    loss_config: str = "simple"
    loss_weighting: str = ""
    dropout_rate: float = 0.05
    augment_prob: float = 0.0
    sigma_data: float = 1.0
    sigma_min: float = 1e-2
    sigma_max: float = 80
    normalize_latent_by: str = "channel_minmaxnorm"
    sigma_sample_density: SigmaDensityConfig = field(default_factory=SigmaDensityConfig)


@dataclass
class DatasetConfig:
    type: str = "uniref"
    path: str = "/shared/amyxlu/data/uniref90/uniref90.fasta"
    toy_data_path: Optional[str] = "/shared/amyxlu/data/uniref90/partial.fasta"
    random_split_seed: int = 42
    num_holdout: int = 50000


@dataclass
class OptimizerConfig:
    type: str = "adamw"
    lr: float = 8e-5
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    eps: float = 1e-8
    weight_decay: float = 1e-4


@dataclass
class LRSchedulerConfig:
    type: str = "cosine"
    warmup_steps: int = 1000
    num_cycles: float = 0.5


@dataclass
class EMASchedulerConfig:
    type: str = "inverse"
    power: float = 0.6667
    max_value: float = 0.9999


@dataclass
class TrainArgs:
    model_config: ModelConfig = field(default_factory=ModelConfig)
    dataset_config: DatasetConfig = field(default_factory=DatasetConfig)
    opt_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    sched_config: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    ema_sched_config: EMASchedulerConfig = field(default_factory=EMASchedulerConfig)

    artifacts_dir: str = "/shared/amyxlu/kdplaid"
    batch_size: int = 64
    checkpointing: bool = False
    compile: bool = False
    config: str = ""
    debug_mode: bool = False 
    demo_every: int = 0
    end_step: Optional[int] = None
    embedding_n: Optional[int] = None
    evaluate_every: int = 1000
    evaluate_with: str = "esmfold_embed"
    gns: bool = False
    grad_accum_steps: int = 1
    mixed_precision: Optional[str] = "bf16"
    clip_norm: Optional[float] = 10.0
    name: str = ""
    num_workers: int = 8
    recycling_n: int = 4
    reset_ema: bool = False
    resume: Optional[str] = None
    resume_inference: Optional[str] = None
    sample_n: int = 64
    log_every: int = 10
    save_every: int = 10000
    seed: Optional[int] = None
    start_method: str = "spawn"
    toy: bool = False
    wandb_entity: str = "amyxlu"
    wandb_group: Optional[str] = None
    wandb_project: str = "kdplaid"
    wandb_save_model: bool = False


def round_to_power_of_two(x, tol):
    approxs = []
    for i in range(math.ceil(math.log2(x))):
        mult = 2**i
        approxs.append(round(x / mult) * mult)
    for approx in reversed(approxs):
        error = abs((approx - x) / x)
        if error <= tol:
            return approx
    return approxs[0]


def make_model(config: ModelConfig):
    if config.type == "protein_transformer_v1":
        model = models.ProteinTransformerDenoiserModelV1(
            n_layers=config.n_layers,
            d_model=config.d_model,
            d_ff=config.d_ff,
            d_head=config.d_head,
            input_size=config.input_size,
            input_dim=config.input_dim,
            min_len=config.min_len,
            num_classes=0,
            dropout=config.dropout,
            sigma_data=config.sigma_data,
        )
    else:
        raise ValueError(f'unsupported model type {config["type"]}')
    return model


def make_denoiser_wrapper(config: ModelConfig):
    sigma_data = getattr(config, "sigma_data", 1.0)
    has_variance = getattr(config, "has_variance", False)
    loss_config = getattr(config, "loss_config", "karras")
    
    if loss_config == "karras":
        weighting = getattr(config, "loss_weighting", "karras")
        scales = getattr(config, "loss_scales", 1)
        
        if not has_variance:
            return partial(
                layers.Denoiser,
                sigma_data=sigma_data,
                weighting=weighting,
                scales=scales,
            )
        return partial(
            layers.DenoiserWithVariance,
            sigma_data=sigma_data,
            weighting=weighting,
        )
        
    if loss_config == "simple":
        if has_variance:
            raise ValueError("Simple loss config does not support a variance output")
        return partial(layers.SimpleLossDenoiser, sigma_data=sigma_data)
    
    if loss_config == "vanilla":
        if has_variance:
            raise ValueError("Vanilla loss config does not support a variance output")
        return partial(layers.SimpleVanilla, sigma_data=sigma_data)
    
    raise ValueError("Unknown loss config type")


def make_sample_density(config: ModelConfig):
    sd_config = config.sigma_sample_density
    sigma_data = config.sigma_data
    
    if sd_config.type == "lognormal":
        loc = getattr(sd_config, 'mean', getattr(sd_config, 'loc', None))
        scale = getattr(sd_config, 'std', getattr(sd_config, 'scale', None))
        return partial(utils.rand_log_normal, loc=loc, scale=scale)

    if sd_config.type == "loglogistic":
        loc = getattr(sd_config, 'loc', math.log(sigma_data))
        scale = getattr(sd_config, 'scale', 0.5)
        min_value = getattr(sd_config, 'min_value', 0.0)
        max_value = getattr(sd_config, 'max_value', float("inf"))
        return partial(
            utils.rand_log_logistic,
            loc=loc,
            scale=scale,
            min_value=min_value,
            max_value=max_value,
        )

    if sd_config.type == "loguniform":
        min_value = getattr(sd_config, 'min_value', config.sigma_min)
        max_value = getattr(sd_config, 'max_value', config.sigma_max)
        return partial(utils.rand_log_uniform, min_value=min_value, max_value=max_value)

    if sd_config.type in {"v-diffusion", "cosine"}:
        min_value = getattr(sd_config, 'min_value', 1e-3)
        max_value = getattr(sd_config, 'max_value', 1e3)
        return partial(
            utils.rand_v_diffusion,
            sigma_data=sigma_data,
            min_value=min_value,
            max_value=max_value,
        )

    if sd_config.type == "split-lognormal":
        loc = getattr(sd_config, 'mean', getattr(sd_config, 'loc', None))
        scale_1 = getattr(sd_config, 'std_1', getattr(sd_config, 'scale_1', None))
        scale_2 = getattr(sd_config, 'std_2', getattr(sd_config, 'scale_2', None))
        return partial(
            utils.rand_split_log_normal, loc=loc, scale_1=scale_1, scale_2=scale_2
        )

    if sd_config.type == "cosine-interpolated":
        min_value = getattr(sd_config, 'min_value', min(config.sigma_min, 1e-3))
        max_value = getattr(sd_config, 'max_value', max(config.sigma_max, 1e3))
        image_d = getattr(sd_config, 'image_d', config.input_size)
        noise_d_low = getattr(sd_config, 'noise_d_low', 32)
        noise_d_high = getattr(sd_config, 'noise_d_high', config.input_size)
        return partial(
            utils.rand_cosine_interpolated,
            image_d=image_d,
            noise_d_low=noise_d_low,
            noise_d_high=noise_d_high,
            sigma_data=sigma_data,
            min_value=min_value,
            max_value=max_value,
        )

    raise ValueError("Unknown sample density type")


def make_lr_sched(lr_config: LRSchedulerConfig, optimizer, num_training_steps: int):
    if lr_config.type == "cosine_with_restarts":
        return get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=lr_config.warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=lr_config.num_cycles,
        )
    elif lr_config.type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=lr_config.warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=lr_config.num_cycles,
        )
    else:
        return get_scheduler(
            name=lr_config.type,
            optimizer=optimizer,
            num_warmup_steps=lr_config.warmup_steps,
            num_training_steps=num_training_steps,
        )
        

DICT_NAME_TO_DATACLASS_NAME = {
    "model_config": ModelConfig,
    "dataset_config": DatasetConfig,
    "opt_config": OptimizerConfig,
    "sched_config": LRSchedulerConfig,
    "ema_sched_config": EMASchedulerConfig,
}