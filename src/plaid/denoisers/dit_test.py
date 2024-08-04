from plaid.denoisers.dit import FunctionOrganismDiT
import torch

N, L, C = 4, 128, 8

denoiser = FunctionOrganismDiT(
    input_dim=C
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
denoiser.to(device)

# Test the forward pass
x = torch.rand(N, L, C).to(device)
x_self_cond = x.clone()
t = torch.randint(1, 100, (N,)).to(device)
mask = torch.ones(N, L).to(device).bool()
function_y = torch.randint(1, 100, (N,)).to(device)
organism_y = torch.randint(1, 100, (N,)).to(device)


denoised = denoiser(x, t, function_y, organism_y, mask, x_self_cond)
import IPython;IPython.embed();exit(1)

