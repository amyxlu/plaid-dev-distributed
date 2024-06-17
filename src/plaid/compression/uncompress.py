from pathlib import Path
from plaid.compression.hourglass_vq import HourglassVQLightningModule
import torch
from copy import deepcopy


class UncompressionLatent:
    def __init__(
        self,
        compression_model_id,
        compression_ckpt_dir,
        init_compress_mode=False,
    ):
        self.device = torch.device("cpu")

        ckpt_path = Path(compression_ckpt_dir) / str(compression_model_id) / "last.ckpt"
        model = HourglassVQLightningModule.load_from_checkpoint(ckpt_path).cpu()

        # we only keep the decoder weights and some attributes for downstream
        self.quantize_scheme = model.quantize_scheme
        self.decoder = deepcopy(model.dec)
        self.quantizer = deepcopy(model.quantizer)
        self.decoder.to(self.device).eval()

        if model.post_quant_proj is not None:
            self.post_quant_proj = model.post_quant_proj
            self.post_quant_proj.to(self.device).eval()
        else:
            self.post_quant_proj = None

        # if we also want to init the weights necessary to compress, don't delete the encoder
        if init_compress_mode:
            self.hourglass_model = model
        else:
            del model

    def to(self, device):
        self.device = device
        self.decoder.to(device)
        return self
    
    def uncompress(self, *args, **kwargs):
        raise NotImplementedError
    

class UncompressContinuousLatent(UncompressionLatent):
    """ For models without any quantization."""
    def __init__(
            self,
            compression_model_id,
            compression_ckpt_dir="/homefs/home/lux70/storage/plaid/checkpoints/hourglass_vq/",
            init_compress_mode=False,
        ):
        super().__init__(compression_model_id, compression_ckpt_dir, init_compress_mode)

    def uncompress(self, z_q, mask=None, verbose=False):
        """Uncompresses by decoder (does not rescale the input)"""
        # quantize and get z_q
        z_q = z_q.to(self.device)

        if mask is not None:
            mask = mask.to(self.device)
 
        if self.post_quant_proj is not None:
            z_q = self.post_quant_proj(z_q)

        return self.decoder(z_q, mask, verbose)
    
    def compress(self, feats, mask=None):
        """
        scale latent and compress with hourglass transformer
        """
        if mask is not None:
            mask = torch.ones((x_norm.shape[0], x_norm.shape[1]))
        del feats

        x_norm, mask = x_norm.to(self.device), mask.to(self.device)

        # compressed_representation manipulated in the Hourglass compression module forward pass
        # to return the detached and numpy-ified representation based on the quantization mode.
        _, _, _, compressed_representation = self.hourglass_model(x_norm, mask.bool(), log_wandb=False)
        return compressed_representation
        

class UncompressDiscreteLatent:
    def __init__(
        self,
        compression_model_id,
        compression_ckpt_dir):
        super().__init__(compression_model_id, compression_ckpt_dir)

        assert self.quantize_scheme == "fsq", f"Only supports fsq but got {self.quantize_scheme}."
    
    def uncompress(self, indices, mask=None, verbose=False):
        codes = self.quantizer.indexes_to_codes(indices)

        if mask is not None:
            mask = mask.to(self.device)
 
        if self.post_quant_proj is not None:
            z_q = self.post_quant_proj(codes)
        
        return self.decoder(z_q, mask, verbose)



if __name__ == "__main__":
    compression_model_id = "2024-03-21T18-49-51"
    device = torch.device("cuda")

    uncompressor = UncompressContinuousLatent(compression_model_id)
    uncompressor.to(device)
    model = uncompressor.model

    from plaid.datasets import CATHShardedDataModule
    dm = CATHShardedDataModule(
        shard_dir="/homefs/home/lux70/storage/data/cath/shards",
        seq_len=256,
    )
    dm.setup()
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    x = batch[0].to(device)
    mask = None

    from plaid.utils import LatentScaler

    scaler = LatentScaler()
    x_norm = scaler.scale(x)

    model.to(device)
    out = model(x_norm, mask)
    x_recons_norm = out[0]
    z_e = out[-1]
    print(out[1])
    print(((x_recons_norm - x_norm) ** 2).mean())

    uncompressed = uncompressor.uncompress(z_e)
    assert torch.allclose(uncompressed, x_recons_norm)
