from plaid.denoisers.hdit import HDiT 
from plaid.diffusion.guided import GaussianDiffusion
from plaid.datasets import CompressedH5DataModule
import torch

model = HDiT()
device = torch.device("cuda")
model.to(device)

x = torch.randn(4, 128, 8).to(device)
t = torch.randint(0, 100, (4,)).to(device)
y = torch.randint(0, 600, (4,)).to(device)
mask = torch.ones_like(x)

output = model(x, t, mask, y)
print(output)

dm = CompressedH5DataModule(batch_size=16)
dm.setup("fit")
dl = dm.train_dataloader()

# optimizer = torch.optim.Adam(dit.parameters(), lr=1e-4)
# diffusion = GaussianDiffusion(model=dit)
# diffusion.to(device)

# for i, batch in enumerate(dl):
#     if i > 15:
#         break
#     x, mask, _ = batch
#     x, mask = x.to(device), mask.to(device)
#     t = torch.randint(0, 500, (x.shape[0],)).to(device)
#     out = dit(x, t, mask)

#     print(out)
#     print(out.mean())
#     print(out.shape)
    
#     out = diffusion(x, mask) 
    
#     loss = ((x - out) ** 2).mean()
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     print(loss)

# from plaid.callbacks.sample_callback import SampleCallback

# callback = SampleCallback(diffusion, calc_fid=True)
# callback.on_train_batch_end(None, diffusion, None, None, None)