from torchvision import datasets, transforms, utils
import torch

device = torch.device("cuda:1")
batch_size = 10

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "mnist",
        download=True,
        train=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),  # first, convert image to PyTorch tensor
                transforms.Normalize((0.1307,), (0.3081,)),  # normalize inputs
            ]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
)
batch = next(iter(train_loader))
img = batch[0]
img = img.to(device)

import plaid as K
from plaid.denoisers.image_transformer_v1 import ImageTransformerDenoiserModelV1

model_config = {
    "type": "image_transformer_v1",
    "input_channels": 1,
    "input_size": [28, 28],
    "patch_size": [4, 4],
    "d_ff": 256,  # ????? added arbitrarily
    "width": 256,
    "depth": 8,
    "loss_config": "karras",
    "loss_weighting": "soft-min-snr",
    "dropout_rate": 0.05,
    "augment_prob": 0.12,
    "sigma_data": 0.6162,
    "sigma_min": 1e-2,
    "sigma_max": 80,
    "sigma_sample_density": {"type": "cosine-interpolated"},
}


model = ImageTransformerDenoiserModelV1(
    n_layers=model_config["depth"],
    d_model=model_config["width"],
    d_ff=model_config["d_ff"],
    in_features=model_config["input_channels"],
    out_features=model_config["input_channels"],
    patch_size=model_config["patch_size"],
    num_classes=0,
    dropout=model_config["dropout_rate"],
    sigma_data=model_config["sigma_data"],
)
model.to(device)

sigmas = K.karras.get_sigmas_karras(
    # batch_size - 1,
    50,
    model_config["sigma_min"],
    model_config["sigma_max"],
    rho=7.0,
    device=device,
)

sample_density = K.config.make_sample_density(model_config)  # a function
sigma = sample_density([batch_size], device=device)
sigma = sigma.to(device)
out = model(img, sigma)
