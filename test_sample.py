from plaid.denoisers.dit import SimpleDiT
from plaid.callbacks.sample_callback import SampleCallback
from plaid.diffusion.guided import GaussianDiffusion
from plaid.compression.uncompress import UncompressContinuousLatent
from plaid.constants import COMPRESSION_INPUT_DIMENSIONS
from plaid.utils import LatentScaler
import torch

device = torch.device("cuda")
compressed_id = "j1v1wv6w"
input_dim = COMPRESSION_INPUT_DIMENSIONS[compressed_id]

dit = SimpleDiT(input_dim=input_dim).to(device)
uncompressor, unscaler = UncompressContinuousLatent(compressed_id), LatentScaler()

diffusion = GaussianDiffusion(
    model=dit, uncompressor=uncompressor, unscaler=unscaler, sampling_timesteps=4
)
diffusion.to(device)

callback = SampleCallback(
    diffusion, calc_fid=True, n_to_sample=12, batch_size=12, n_to_construct=12, 
    fid_holdout_tensor_fpath="/homefs/home/lux70/plaid_cached_tensors/uniref_esmfold_feats.st",
    calc_sequence_properties=True
)

latent, _ = callback.sample_latent((12, 64, input_dim))
latent = diffusion.process_x_to_latent(latent)
out = callback.construct_sequence(latent, device)
import IPython;IPython.embed()