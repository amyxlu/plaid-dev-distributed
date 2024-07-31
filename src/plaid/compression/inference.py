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
        init_decompress_mode=True
    ):
        self.device = torch.device("cpu")
        ckpt_path = Path(compression_ckpt_dir) / str(compression_model_id) / "last.ckpt"
        self.model = HourglassVQLightningModule.load_from_checkpoint(ckpt_path).cpu()
        self.shorten_factor = self.model.enc.shorten_factor
        self.downproj_factor = self.model.enc.downproj_factor
        
        assert init_compress_mode or init_decompress_mode
        if init_compress_mode is False:
            del self.model.enc
        if init_decompress_mode is False:
            del self.model.dec

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

    def uncompress(self, *args, **kwargs):
        raise NotImplementedError
    
    def compress(self, *args, **kwargs):
        raise NotImplementedError


class UncompressContinuousLatent(UncompressionLatent):
    """For models without any quantization."""

    def __init__(
        self,
        compression_model_id,
        compression_ckpt_dir="/data/lux70/plaid/checkpoints/hourglass_vq/",
        init_compress_mode=False,
        init_decompress_mode=True,
    ):
        super().__init__(compression_model_id, compression_ckpt_dir, init_compress_mode, init_decompress_mode)

    def uncompress(self, z_q, mask=None, verbose=False):
        """Uncompresses by decoder (does not rescale the input)"""
        # quantize and get z_q
        z_q = z_q.to(self.device)

        if mask is not None:
            mask = mask.to(self.device)

        if self.model.post_quant_proj is not None:
            z_q = self.model.post_quant_proj(z_q)

        return self.model.dec(z_q, mask, verbose)

    def compress(self, x_norm, mask=None):
        """
        scale latent and compress with hourglass transformer
        """
        if mask is None:
            mask = torch.ones((x_norm.shape[0], x_norm.shape[1]))
        
        x_norm, mask = x_norm.to(self.device), mask.to(self.device)

        # compressed_representation manipulated in the Hourglass compression module forward pass
        # to return the detached and numpy-ified representation based on the quantization mode.
        _, _, _, compressed_representation, downsampled_mask = self.model(x_norm, mask.bool(), log_wandb=False)
        return compressed_representation, downsampled_mask


class UncompressDiscreteLatent:
    def __init__(self, compression_model_id, compression_ckpt_dir):
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
    compression_model_id = "jzlv54wl"
    device = torch.device("cuda")

    uncompressor = UncompressContinuousLatent(compression_model_id)
    uncompressor.to(device)
    model = uncompressor.model

    from plaid.datasets import CATHShardedDataModule

    dm = CATHShardedDataModule(
        shard_dir="/data/lux70/data/cath/shards",
        seq_len=128,
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
