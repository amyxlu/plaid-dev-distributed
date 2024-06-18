import plaid as K
import torch

config_path = "/home/amyxlu/kdiffusion/configs/config_protein_transformer_v1.json"
config = K.config.load_config(config_path)
model_config = config["model"]
dataset_config = config["dataset"]
opt_config = config["optimizer"]
sched_config = config["lr_sched"]
ema_sched_config = config["ema_sched"]
seq_len = model_config["input_size"]

device = torch.device("cuda:0")

inner_model = K.config.make_model(config)
model = K.config.make_denoiser_wrapper(config)(inner_model)
model.to(device)
sample_density = K.config.make_sample_density(model_config)


N = 64
sigma = sample_density([N], device=device)
x = torch.randn(N, seq_len, model_config["d_model"], device=device)
noise = torch.randn_like(x)

extra_args = {"mask": torch.ones(N, seq_len, device=device)}
model_output = model.loss(x, noise, sigma, return_model_output=True, **extra_args)
