from plaid.denoisers import SimpleDiT, CFGDiT 
from plaid.diffusion.beta_schedulers import VDiffusionSigmas
from plaid.callbacks import SampleCallback
from plaid.diffusion import GaussianDiffusion, ElucidatedDiffusion
from plaid.compression.uncompress import UncompressContinuousLatent
from plaid.diffusion.guided import GaussianDiffusion
from plaid.datasets import CompressedH5DataModule
import torch

model = CFGDiT()
device = torch.device("cuda")
model.to(device)

batch_size = 16
t = torch.randint(0, 100, (batch_size,)).to(device).long()

dm = CompressedH5DataModule(
    compression_model_id="jzlv54wl",
    h5_root_dir="/homefs/home/lux70/storage/data/pfam/compressed/subset_5K_with_clan",
    batch_size=16
)
dm.setup("fit")
dl = dm.train_dataloader()
batch = next(iter(dl))
batch = tuple(map(lambda x: x.to(device), batch))
x, mask, clan = batch
clan = clan.long().squeeze()




# TODO: create uncompressor / unscaler? 
# TODO: add sequence / structure again?
uncompressor = UncompressContinuousLatent("jzlv54wl")
sigma_density_generator = VDiffusionSigmas(
    max_value=1e3, min_value=1e-3, sigma_data=1
)
sigma = sigma_density_generator(x.shape[0])
sigma = sigma.to(device)
diffusion = ElucidatedDiffusion(
    model, sigma_density_generator, sigma_data=1,
)
diffusion = diffusion.to(device)

diffusion_loss = diffusion.loss(
    x=x,
    label=clan,
    sigma=sigma,
    mask=mask,
    x_self_cond=None
)

import IPython;IPython.embed()




# callback = SampleCallback(
#         diffusion,
#         log_to_wandb=False,
#         calc_structure= False,
#         calc_sequence= False,
#         calc_perplexity=False,
#         calc_fid = True,
#         fid_holdout_tensor_fpath= "/homefs/home/lux70/plaid_cached_tensors/uniref_esmfold_feats.st",
#         normalize_real_features = True,
#         save_generated_structures = False,
#         num_recycles= 4,
#         outdir = "sampled",
# )

# x = callback.sample_compressed_latent((4, 128, 8))[0]
# latent = diffusion.process_x_to_latent(x)

# callback.calculate_fid(latent.detach(), device)

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


from plaid.diffusion.beta_schedulers import VDiffusionSigmas


sigma_density = VDiffusionSigmas()

N = 16
sigmas = sigma_density((N,))