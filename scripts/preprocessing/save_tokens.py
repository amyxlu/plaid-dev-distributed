import os
from pathlib import Path

import torch
from safetensors.torch import save_file
from tqdm import tqdm

from plaid.datasets import CATHShardedDataModule
from plaid.compression.hourglass_vq import HourglassVQLightningModule
from plaid.transforms import trim_or_pad_batch_first
from plaid.esmfold.misc import batch_encode_sequences
from plaid.utils import LatentScaler
from plaid.losses.functions import masked_mse_loss


"""Config"""
compress_model_dir = "/data/lux70/plaid/checkpoints/hourglass_vq/"
compress_model_id = "2024-03-05T06-20-52"  # soft-violet
compress_model_path = Path(compress_model_dir) / compress_model_id / "last.ckpt"

# the input dataloader to use
max_seq_len = 128
input_dtype = "fp32"
batch_size = 256
num_workers = 0
lm_embedder_type = "esmfold"
shard_dir = "/data/lux70/data/cath/shards/"

device = torch.device("cuda")


"""
Dataloader
"""

print("making datamodule")
dm = CATHShardedDataModule(
    storage_type="hdf5",
    shard_dir=shard_dir,
    embedder=lm_embedder_type,
    seq_len=max_seq_len,
    batch_size=batch_size,
    dtype=input_dtype,
    num_workers=num_workers,
)
dm.setup()
train_dataloader = dm.train_dataloader()
val_dataloader = dm.val_dataloader()


"""
Model
"""
print("loading model")
model = HourglassVQLightningModule.load_from_checkpoint(compress_model_path)


class TokenizeLatent:
    def __init__(
        self,
        model,
        out_dtype="int8",
    ):
        self.latent_scaler = LatentScaler()
        self.out_dtype = out_dtype
        self.device = torch.device("cuda")
        self.model = model
        self.model.to(self.device)

    def _to_int8(self, x):
        assert x.max() < 128
        return x.to(dtype=torch.int8)

    def _to_int16(self, x):
        return x.to(dtype=torch.int16)

    def save_safetensors(eslf, tokens, outpath):
        outpath = Path(outpath)
        if not outpath.parent.exists():
            outpath.parent.mkdir(parents=True)
        save_file({"tokens": tokens}, outpath)

    def tokenize_batch(self, batch):
        x = batch[0].to(self.device)
        sequences = batch[1]

        # make mask
        _, mask, _, _, _ = batch_encode_sequences(sequences)
        mask = mask.to(self.device)
        mask = trim_or_pad_batch_first(mask, pad_to=max_seq_len, pad_idx=0)

        # scale
        x_norm = self.latent_scaler.scale(x)

        # model forward pass!!
        recons_norm, loss, log_dict, quant_out = self.model(
            x_norm, mask.bool(), log_wandb=False
        )  # , return_vq_output=True)
        recons_loss = masked_mse_loss(recons_norm, x_norm, mask)
        # print(recons_loss)

        """
        Process codebook
        """

        # get indices
        B = x.shape[0]
        L = x.shape[1] // self.model.enc.shorten_factor
        # print(B, L)
        codebook = quant_out["min_encoding_indices"].squeeze()
        # print(codebook.shape)
        codebook = codebook.reshape(B, L, -1)
        # print(codebook.shape)

        if self.out_dtype == "int8":
            return self._to_int8(codebook)
        elif self.out_dtype == "int16":
            return self._to_int16(codebook)
        else:
            raise NotImplementedError

    def dataloader_to_tokens(self, dataloader):
        all_tokens = []
        for i, batch in enumerate(tqdm(dataloader)):
            # if i > 3:
            #     break
            codebook = self.tokenize_batch(batch)
            all_tokens.append(codebook)
        tokens = torch.cat(all_tokens)
        return tokens


"""
Save tokens
"""
token_outdir_base = Path("/data/lux70/data/cath/tokens")
latent_tokenizer = TokenizeLatent(model, "int8")

print("save train tensors")
outpath = token_outdir_base / compress_model_id / "train" / f"seqlen_{max_seq_len}" / "tokens.st"
tokens = latent_tokenizer.dataloader_to_tokens(train_dataloader)
latent_tokenizer.save_safetensors(tokens, outpath)

print("save val tensors")
split = "val"
outpath = token_outdir_base / compress_model_id / "val" / f"seqlen_{max_seq_len}" / "tokens.st"
tokens = latent_tokenizer.dataloader_to_tokens(val_dataloader)
latent_tokenizer.save_safetensors(tokens, outpath)
