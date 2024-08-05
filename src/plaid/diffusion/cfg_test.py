from plaid.diffusion import FunctionOrganismDiffusion
from plaid.denoisers import FunctionOrganismDiT
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N, L, C = 4, 128, 32 

denoiser = FunctionOrganismDiT(input_dim=32).to(device)
diffusion = FunctionOrganismDiffusion(model=denoiser)
diffusion.to(device)

# Test the forward pass
x = torch.rand(N, L, C).to(device)
x_self_cond = x.clone()
t = torch.randint(1, 100, (N,)).to(device)
mask = torch.ones(N, L).to(device).bool()
function_y = torch.randint(1, 100, (N,)).to(device)
organism_y = torch.randint(1, 100, (N,)).to(device)

batch = (x, mask, function_y, organism_y, None, None, None)

loss = diffusion.training_step(batch, 0)

import IPython; IPython.embed(); exit(1)