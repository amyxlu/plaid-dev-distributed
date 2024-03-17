import typing as T

import lightning as L
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim import AdamW
import einops
import wandb
import torch
import numpy as np
import pandas as pd

from plaid.compression.modules import HourglassDecoder, HourglassEncoder, VectorQuantizer, FiniteScalarQuantizer
from plaid.utils import LatentScaler, get_lr_scheduler
from plaid.transforms import trim_or_pad_batch_first
from plaid.esmfold.misc import batch_encode_sequences
from plaid.proteins import LatentToSequence, LatentToStructure
from plaid.losses.modules import SequenceAuxiliaryLoss, BackboneAuxiliaryLoss
from plaid.losses.functions import masked_mse_loss



class HourglassVQLightningModule(L.LightningModule):
    def __init__(
        self,
        dim,
        *,
        depth=4,  # depth used for both encoder and decoder
        shorten_factor=2,
        downproj_factor=2,
        attn_resampling=True,
        updown_sample_type="naive",
        heads=8,
        dim_head=64,
        causal=False,
        norm_out=False,
        use_quantizer="vq",
        # quantizer
        n_e=16,
        e_dim=64,
        vq_beta=0.25,
        fsq_levels: T.Optional[T.List[int]] = None,
        lr=1e-4,
        lr_adam_betas=(0.9, 0.999),
        lr_sched_type: str = "constant",
        lr_num_warmup_steps: int = 0,
        lr_num_training_steps: int = 10_000_000,
        lr_num_cycles: int = 1,
        # scaler
        latent_scaler=LatentScaler(),
        seq_emb_fn=None,
        # auxiliary losses
        seq_loss_weight: float = 0.0,
        struct_loss_weight: float = 0.0,
        log_sequence_loss=False,
        log_structure_loss=False,
    ):
        super().__init__()
        
        """Make quantizer. Can be either the traditional VQ-VAE, the FSQ, or
        none (i.e. output of encoder goes directly back into the decoder).
        """

        if isinstance(use_quantizer,  bool):
            if use_quantizer:
                print("using quantization: vq")
                self.quantize_scheme = "vq"
            else:
                print("using non-quantization mode")
                self.quantize_scheme = None  # no quantization
        else:
            assert use_quantizer in ['vq', 'fsq']
            self.quantize_scheme = use_quantizer
            print(f"using quantizer {use_quantizer}")
        
        if self.quantize_scheme == "vq":
            self.quantizer = VectorQuantizer(n_e, e_dim, vq_beta)

        elif self.quantize_scheme == "fsq":
            self.fsq_levels = fsq_levels

            assert len(fsq_levels) == (dim / downproj_factor) 
            self.quantizer = FiniteScalarQuantizer(fsq_levels)

        self.quantizer.to(self.device)

        self.enc = HourglassEncoder(
            dim=dim,
            depth=depth,
            shorten_factor=shorten_factor,
            downproj_factor=downproj_factor,
            attn_resampling=attn_resampling,
            updown_sample_type=updown_sample_type,
            heads=heads,
            dim_head=dim_head,
            causal=causal,
            norm_out=norm_out
        )
        self.dec = HourglassDecoder(
            dim=dim // downproj_factor,
            depth=depth,
            elongate_factor=shorten_factor,
            upproj_factor=downproj_factor,
            attn_resampling=True,
            updown_sample_type="linear"
        )

        self.z_q_dim = dim // np.prod(dim) 
        self.n_e = n_e
        self.latent_scaler = latent_scaler

        self.lr = lr
        self.lr_adam_betas = lr_adam_betas
        self.lr_sched_type = lr_sched_type
        self.lr_num_warmup_steps = lr_num_warmup_steps
        self.lr_num_training_steps = lr_num_training_steps
        self.lr_num_cycles = lr_num_cycles

        self.seq_emb_fn = seq_emb_fn
        self.log_sequence_loss = log_sequence_loss or (seq_loss_weight > 0.)
        self.log_structure_loss = log_structure_loss or (struct_loss_weight > 0.)
        self.seq_loss_weight = seq_loss_weight
        self.struct_loss_weight = struct_loss_weight

        if self.log_sequence_loss:
            self.sequence_constructor = LatentToSequence()
            self.sequence_constructor.to(self.device)
            self.seq_loss_fn = SequenceAuxiliaryLoss(self.sequence_constructor)

        if self.log_structure_loss:
            self.structure_constructor = LatentToStructure()
            self.structure_constructor.to(self.device)
            self.structure_loss_fn = BackboneAuxiliaryLoss(self.structure_constructor)
        self.save_hyperparameters()
    
    def forward_no_quantize(self, x, mask, verbose=False, log_wandb=True, *args, **kwargs):
        z_e, downsampled_mask = self.enc(x, mask, verbose)
        z_e_out = z_e.clone()
        x_recons = self.dec(z_e, downsampled_mask, verbose)
        recons_loss = masked_mse_loss(x_recons, x, mask)
        loss = recons_loss
        log_dict = {
            "loss": loss.item(),
            "recons_loss": recons_loss.item()
        }
        return x_recons, loss, log_dict, z_e_out

    def forward(self, x, mask, verbose=False, log_wandb=True, *args, **kwargs):
        # exit and call other function not using quantization
        if self.quantize_scheme is None:
            return self.forward_no_quantize(x, mask, verbose, log_wandb, *args, **kwargs)

        # encode and possibly downsample
        z_e, downsampled_mask = self.enc(x, mask, verbose)
        log_dict = {}

        # quantize and get z_q
        if self.quantize_scheme == "vq": 
            quant_out = self.quantizer(z_e, verbose)
            z_q = quant_out['z_q']
            vq_loss = quant_out['loss']
            log_dict["vq_loss"] = quant_out['loss']
            log_dict["vq_perplexity"] = quant_out['perplexity']
            codebook = quant_out['min_encoding_indices'].detach().cpu().numpy().reshape(-1)
            n_bins = self.n_e

        elif self.quantize_scheme == "fsq":
            z_q = self.quantizer.quantize(z_e)
            n_bins = self.quantizer.codebook_size
            vq_loss = 0
            codebook = self.quantizer.codes_to_indexes(z_q).detach().cpu().numpy().reshape(-1)
            quant_out = {"codebook": codebook}  # for inference use
        else:
            raise NotImplementedError

        x_recons = self.dec(z_q, downsampled_mask, verbose)

        if (self.global_step % 5000 == 0) and log_wandb and (self.quantize_scheme is not None):
            fig, ax = plt.subplots()
            ax.hist(codebook, bins=n_bins)
            wandb.log({"codebook_index_hist": wandb.Image(fig)})

        recons_loss = masked_mse_loss(x_recons, x, mask)
        loss = vq_loss + recons_loss
        log_dict['recons_loss'] = recons_loss.item()
        log_dict['loss'] = loss.item()
        return x_recons, loss, log_dict, quant_out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.enc.parameters()) + list(self.dec.parameters()) + list(self.quantizer.parameters()),
            lr=self.lr,
            betas=self.lr_adam_betas
        )
        scheduler = get_lr_scheduler(
            optimizer=optimizer,
            sched_type=self.lr_sched_type,
            num_warmup_steps=self.lr_num_warmup_steps,
            num_training_steps=self.lr_num_training_steps,
            num_cycles=self.lr_num_cycles,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def run_batch(self, batch, prefix="train"):
        """
        The input batch can be:
        (1) precomputed embeddings along with a dictionary of structures (CATHShardedDataModule)
        (2) precomputed embeddings with a placeholder for the structure dictionary (CATHStructureDataModule)
        (3) raw headers and sequences tuples (FastaDataset)

        to trigger the raw sequence mode, the `seq_emb_fn` should be passed, which should be defined outside 
        the train loop, and should of the desired embedding function from ESMFold/etc., already moved to device.
        """
        if len(batch) == 3:
            # todo: might be easier to just save seqlen when processing for making masks
            # in this form, the sequences *must* have the correct length (trimmed, no pad)
            x, sequences, gt_structures = batch
        elif len(batch) == 2:
            assert not self.seq_emb_fn is None
            headers, sequences = batch
            x = self.seq_emb_fn(sequences, device=self.device)
        else:
            raise

        # get masks and ground truth tokens and move to device
        tokens, mask, _, _, _ = batch_encode_sequences(sequences)
        if mask.shape[1] != x.shape[1]:
            # pad with False
            mask = trim_or_pad_batch_first(mask, x.shape[1], pad_idx=0)
            tokens = trim_or_pad_batch_first(tokens, x.shape[1], pad_idx=0)

        x = x.to(self.device)
        mask = mask.to(self.device) 
        tokens = tokens.to(self.device)

        # scale (maybe) latent to be more organized before VQ-Hourglass
        x = self.latent_scaler.scale(x)

        # forward pass
        if not self.quantize_scheme is None:
            x_recons, loss, log_dict, _ = self(x, mask.bool())
        else:
            x_recons, loss, log_dict, _ = self.forward_no_quantize(x, mask.bool())
        self.log_dict({f"{prefix}/{k}": v for k,v in log_dict.items()}, batch_size=x.shape[0])

        # unscale to decode into sequence and/or structure
        x_recons_unscaled = self.latent_scaler.unscale(x_recons)
        batch_size = x_recons_unscaled.shape[0]

        # sequence loss
        if self.log_sequence_loss:
            seq_loss, seq_loss_dict, recons_strs = self.seq_loss_fn(x_recons_unscaled, tokens, mask, return_reconstructed_sequences=True)
            seq_loss_dict = {f"{prefix}/{k}": v for k, v in seq_loss_dict.items()}
            self.log_dict(seq_loss_dict, on_step=(prefix != "val"), on_epoch=True, batch_size=batch_size)
            # if self.global_step % 500 == 0:
            #     tbl = pd.DataFrame({"reconstructed": recons_strs, "original": sequences})
            #     wandb.log({f"{prefix}/recons_strs_tbl": wandb.Table(dataframe=tbl)})
            loss += seq_loss * self.seq_loss_weight

        # structure loss
        if self.log_structure_loss:
            struct_loss, struct_loss_dict = self.structure_loss_fn(x_recons_unscaled, gt_structures, sequences)
            struct_loss_dict = {f"{prefix}/{k}": v.mean() for k, v in struct_loss_dict.items()}
            self.log_dict(struct_loss_dict, on_step=(prefix != "val"), on_epoch=True, batch_size=batch_size)
            loss += struct_loss * self.struct_loss_weight

        return loss

    def training_step(self, batch, batch_idx):
        return self.run_batch(batch, prefix="train")

    def validation_step(self, batch, batch_idx):
        return self.run_batch(batch, prefix="val")


if __name__ == "__main__":
    device = torch.device("cuda")
    hvq = HourglassVQLightningModule(
        dim=1024,
        fsq_levels=[8] * 16,
        downproj_factor=64
    ).to(device)

    x = torch.randn(8, 128, 1024).to(device)
    mask = torch.ones(8, 128).bool().to(device)

    print(hvq(x, mask, verbose=True))