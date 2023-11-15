from functools import partial
import enum
import math

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Optional, List

from transformers import (
    get_scheduler,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)

from . import layers, models, utils, external
from .sampling import (
    sample_euler,
    sample_euler_ancestral,
    sample_heun,
    sample_dpm_2,
    sample_dpm_2_ancestral,
    sample_lms,
    sample_dpm_fast,
    # sample_dpm_adaptive,
    sample_dpmpp_2s_ancestral,
    sample_dpmpp_sde,
    sample_dpmpp_2m,
    sample_dpmpp_2m_sde,
    sample_dpmpp_3m_sde,
)


def dataclass_to_dict(obj):
    outdict = {}
    for k, v in obj.__dict__.items():
        if is_dataclass(v):
            outdict[k] = dataclass_to_dict(v)
        else:
            outdict[k] = v
    return outdict


def dataclass_from_dict(dataclass_, d):
    try:
        fieldtypes = {f.name: f.type for f in fields(dataclass_)}
        return dataclass_(**{f: dataclass_from_dict(fieldtypes[f], d[f]) for f in d})
    except:
        return d  # Assume that this is a terminal field (e.g., int, str, etc.)


class SampleSolverType(str, enum.Enum):
    EULER = "euler" 
    EULER_ANCESTRAL = "euler_ancestral" 
    HEUN = "heun" 
    DPM_2 = "dpm_2"
    DPM_2_ANCESTRAL = "dpm_2_ancestral" 
    LMS = "lms" 
    DPM_FAST = "dpm_fast"
    DPM_ADAPTIVE = "dpm_adaptive" 
    DPMPP_2S_ANCESTRAL = "dpmpp_2s_ancestral" 
    DPMPP_SDE = "dpmpp_sde" 
    DPMPP_2M = "dpmpp_2m" 
    DPMPP_2M_SDE = "dpmpp_2m_sde" 
    DPMPP_3M_SDE = "dpmpp_3m_sde" 


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
    n_layers: int = 11
    d_model: int = 512
    d_ff: int = 256
    d_head: int = 128
    input_size: int = 512
    input_dim: int = 1024
    skip_connect: bool = True
    min_len: int = 32
    num_classes: int = 0
    dropout: float = 0.0
    loss_config: str = "karras"
    loss_distance: str = "huber"
    loss_weighting: str = "soft-min-snr"
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
    lr: float = 1e-4 
    betas: List[float] = field(default_factory=lambda: [0.9, 0.99])
    eps: float = 1e-6
    weight_decay: float = 1e-2
    resume_from_saved_state: bool = True 


@dataclass
class LRSchedulerConfig:
    type: str = "inverse_sqrt"
    warmup_steps: int = 10000
    num_cycles: float = 0.5


@dataclass
class EMASchedulerConfig:
    type: str = "inverse"
    power: float = 0.6667
    max_value: float = 0.9999


@dataclass
class SampleCallbackConfig:
    solver_type: SampleSolverType = SampleSolverType.DPMPP_2M_SDE
    seq_len: int = 128
    use_ema: bool = True
    batch_size: int = 32
    n_to_sample: int = 32
    n_to_construct: int = -1
    num_recycles: int = 4
    sigma_max: float = 1e-2
    sigma_min: float = 1e3
    rho: float = 7.0
    n_steps: int = 15
    model_id: Optional[str] = None
    model_step: Optional[int] = None
    device_id: int = 0
    save_to_disk: bool = True
    log_to_wandb: bool = False
    calc_perplexity: bool = True
    base_artifact_dir: str = "/shared/amyxlu/kdplaid"
    sequence_decode_temperature: float = 1.0
    calc_fid: bool = False


@dataclass
class TrainArgs:
    model_config: ModelConfig = field(default_factory=ModelConfig)
    dataset_config: DatasetConfig = field(default_factory=DatasetConfig)
    opt_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    sched_config: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    ema_sched_config: EMASchedulerConfig = field(default_factory=EMASchedulerConfig)
    sample_config: SampleCallbackConfig = field(default_factory=SampleCallbackConfig)

    artifacts_dir: str = "/shared/amyxlu/kdplaid"
    batch_size: int = 128 
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
    clip_norm: Optional[float] = 0.5 
    clip_value: Optional[float] = None
    name: str = ""
    num_workers: int = 8
    recycling_n: int = 4
    reset_ema: bool = False
    resume: Optional[str] = None
    resume_inference: Optional[str] = None
    sample_n: int = 64
    log_every: int = 25
    save_every: int = 5000
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
            skip_connect=getattr(config, "skip_connect", False),
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
    loss_distance = getattr(config, "loss_distance", "mse")

    if loss_config == "karras":
        weighting = getattr(config, "loss_weighting", "karras")
        scales = getattr(config, "loss_scales", 1)

        if not has_variance:
            return partial(
                layers.Denoiser,
                sigma_data=sigma_data,
                weighting=weighting,
                scales=scales,
                loss_distance=loss_distance,
            )
        return partial(
            layers.DenoiserWithVariance,
            sigma_data=sigma_data,
            weighting=weighting,
        )

    if loss_config == "simple":
        if has_variance:
            raise ValueError("Simple loss config does not support a variance output")
        return partial(layers.SimpleLossDenoiser, sigma_data=sigma_data, loss_distance=loss_distance)

    if loss_config == "vanilla":
        if has_variance:
            raise ValueError("Vanilla loss config does not support a variance output")
        return partial(layers.SimpleVanilla, sigma_data=sigma_data, loss_distance=loss_distance)
    
    raise ValueError("Unknown loss config type")


def make_sample_density(config: ModelConfig):
    sd_config = config.sigma_sample_density
    sigma_data = config.sigma_data

    if sd_config.type == "lognormal":
        loc = getattr(sd_config, "mean", getattr(sd_config, "loc", None))
        scale = getattr(sd_config, "std", getattr(sd_config, "scale", None))
        return partial(utils.rand_log_normal, loc=loc, scale=scale)

    if sd_config.type == "loglogistic":
        loc = getattr(sd_config, "loc", math.log(sigma_data))
        scale = getattr(sd_config, "scale", 0.5)
        min_value = getattr(sd_config, "min_value", 0.0)
        max_value = getattr(sd_config, "max_value", float("inf"))
        return partial(
            utils.rand_log_logistic,
            loc=loc,
            scale=scale,
            min_value=min_value,
            max_value=max_value,
        )

    if sd_config.type == "loguniform":
        min_value = getattr(sd_config, "min_value", config.sigma_min)
        max_value = getattr(sd_config, "max_value", config.sigma_max)
        return partial(utils.rand_log_uniform, min_value=min_value, max_value=max_value)

    if sd_config.type in {"v-diffusion", "cosine"}:
        min_value = getattr(sd_config, "min_value", 1e-3)
        max_value = getattr(sd_config, "max_value", 1e3)
        return partial(
            utils.rand_v_diffusion,
            sigma_data=sigma_data,
            min_value=min_value,
            max_value=max_value,
        )

    if sd_config.type == "split-lognormal":
        loc = getattr(sd_config, "mean", getattr(sd_config, "loc", None))
        scale_1 = getattr(sd_config, "std_1", getattr(sd_config, "scale_1", None))
        scale_2 = getattr(sd_config, "std_2", getattr(sd_config, "scale_2", None))
        return partial(
            utils.rand_split_log_normal, loc=loc, scale_1=scale_1, scale_2=scale_2
        )

    if sd_config.type == "cosine-interpolated":
        min_value = getattr(sd_config, "min_value", min(config.sigma_min, 1e-3))
        max_value = getattr(sd_config, "max_value", max(config.sigma_max, 1e3))
        image_d = getattr(sd_config, "image_d", config.input_size)
        noise_d_low = getattr(sd_config, "noise_d_low", 32)
        noise_d_high = getattr(sd_config, "noise_d_high", config.input_size)
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


def make_sample_solver_fn(solver_type: SampleSolverType):
    if solver_type == SampleSolverType.EULER:
        return sample_euler
    elif solver_type == SampleSolverType.EULER_ANCESTRAL:
        return sample_euler_ancestral
    elif solver_type == SampleSolverType.HEUN:
        return sample_heun
    elif solver_type == SampleSolverType.DPM_2:
        return sample_dpm_2
    elif solver_type == SampleSolverType.DPM_2_ANCESTRAL:
        return sample_dpm_2_ancestral
    elif solver_type == SampleSolverType.LMS:
        return sample_lms
    elif solver_type == SampleSolverType.DPM_FAST:
        return sample_dpm_fast
    elif solver_type == SampleSolverType.DPM_ADAPTIVE:
        raise NotImplementedError("TODO: implement argument passing for adaptive DPM with sigma_min and sigma_max instead of fixed sigmas.")
    elif solver_type == SampleSolverType.DPMPP_2S_ANCESTRAL:
        return sample_dpmpp_2s_ancestral
    elif solver_type == SampleSolverType.DPMPP_SDE:
        return sample_dpmpp_sde
    elif solver_type == SampleSolverType.DPMPP_2M:
        return sample_dpmpp_2m
    elif solver_type == SampleSolverType.DPMPP_2M_SDE:
        return sample_dpmpp_2m_sde
    elif solver_type == SampleSolverType.DPMPP_3M_SDE:
        return sample_dpmpp_3m_sde
    else:
        raise ValueError(f"Unknown solver type")