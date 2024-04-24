from plaid.denoisers.dit import SimpleDiT
from plaid.diffusion.guided import GaussianDiffusion
from plaid.datasets import CompressedH5DataModule
import torch

dit = SimpleDiT()
device = torch.device("cuda")
dit.to(device)

dm = CompressedH5DataModule(batch_size=16)
dm.setup("fit")
dl = dm.train_dataloader()

optimizer = torch.optim.Adam(dit.parameters(), lr=1e-4)
diffusion = GaussianDiffusion(model=dit)
diffusion.to(device)

for i, batch in enumerate(dl):
    if i > 15:
        break
    x, mask, _ = batch
    x, mask = x.to(device), mask.to(device)
    t = torch.randint(0, 500, (x.shape[0],)).to(device)
    out = dit(x, t, mask)

    print(out)
    print(out.mean())
    print(out.shape)
    
    out = diffusion(x, mask) 
    
    loss = ((x - out) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss)

from plaid.callbacks.sample_callback import SampleCallback

callback = SampleCallback(diffusion, calc_fid=False)
callback.on_train_epoch_end()