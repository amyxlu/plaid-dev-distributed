from plaid.denoisers.flash_attn_dit import FunctionOrganismDiT, DenoiserKwargs
import torch

N, L, C = 4, 128, 32 

denoiser = FunctionOrganismDiT(
    input_dim=C,
    use_self_conditioning=True,
    use_xformers=True,
    use_skip_connect=True
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

denoiser_kwargs = DenoiserKwargs(
    x=x,
    t=t,
    function_y=function_y,
    organism_y=organism_y,
    mask=mask,
    x_self_cond=x_self_cond
)
optimizer = torch.optim.Adam(denoiser.parameters(), lr=1e-3)


for _ in range(20):
    denoised1 = denoiser.forward_with_cond_drop(denoiser_kwargs, 0.3, 0.3)
    print(denoised1)
    print()

    optimizer.zero_grad()
    loss = (denoised1 - denoiser_kwargs.x) ** 2
    loss.mean().backward()
    optimizer.step()

denoised2 = denoiser.forward_with_cond_scale(denoiser_kwargs, cond_scale=6, rescaled_phi=0.7) 


import IPython;IPython.embed();exit(1)