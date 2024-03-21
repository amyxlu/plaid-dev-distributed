from plaid.datasets import CATHStructureDataModule
import os
from pathlib import Path
import time
from plaid.esmfold.misc import batch_encode_sequences
from plaid.transforms import trim_or_pad_batch_first
from plaid.utils import LatentScaler
from plaid.proteins import LatentToStructure
import torch
from plaid.compression.hourglass_vq import HourglassVQLightningModule
import matplotlib.pyplot as plt


device = torch.device("cuda")


def load_compression_model(model_id):
    dirpath = Path(f"/homefs/home/lux70/storage/plaid/checkpoints/hourglass_vq/{model_id}")
    return HourglassVQLightningModule.load_from_checkpoint(dirpath / "last.ckpt")


class CompressionReconstructionCallback:
    """
    For compression experiments, evaluate the reconstruction quality.
    """
    def __init__(
            self,
            compression_model,
            batch_size,
            esmfold=None,
            shard_dir = "/homefs/home/lux70/storage/data/cath/shards/",
            pdb_dir = "/data/bucket/lux70/data/cath/dompdb",
            num_samples: int = 32,
            max_seq_len: int = 256,
            num_recycles: int = 4
        ):
        self.latent_scaler = LatentScaler()
        self.structure_constructor = LatentToStructure(esmfold=esmfold)

        self.batch_size = batch_size
        self.shard_dir = shard_dir
        self.pdb_dir = pdb_dir
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.compression_model = compression_model
        self.num_recycles = num_recycles
        self.base_pdb_dir = Path("/homefs/home/lux70/cache/")

        self.quantize_scheme = self.model.quantize_scheme
        x, mask, sequences, gt_structures = self._get_validation_data() 

        self.x = x
        self.mask = mask
        self.sequences = sequences
        self.gt_structures = gt_structures

    # TODO: device?
        
    def _get_validation_data(self):
        start = time.time()
        print(f"Creating reference validation data of {self.num_samples} points...")

        # only preprocess num_samples data points and load all in one batch
        dm = CATHStructureDataModule(
            self.shard_dir,
            self.pdb_dir,
            seq_len=self.max_seq_len,
            batch_size=self.num_samples,
            max_num_samples=self.num_samples,
            shuffle_val_dataset=False
        ) 
        dm.setup()
        val_dataloader = dm.val_dataloader()
        batch = next(iter(val_dataloader))

        x = batch[0]
        sequences = batch[1]
        gt_structures = batch[-1]

        # make mask
        _, mask, _, _, _ = batch_encode_sequences(sequences)
        mask = trim_or_pad_batch_first(mask, pad_to=self.max_seq_len, pad_idx=0)
        
        end = time.time()
        print(f"Created reference structure validation dataset in {end - start:.2f} seconds.")
        return x, mask, sequences, gt_structures

    def _compress_and_reconstruct(self):
        print("Running dataset through model bottleneck...")
        x_norm = self.latent_scaler.scale(self.x)
        recons_norm, loss, log_dict, quant_out = self.compression_model(x_norm, self.mask.bool(), log_wandb=False)
        recons = self.latent_scaler.unscale(recons_norm)

        if self.quantize_scheme == "vq":
            N, L, _ = x_norm.shape
            print(quant_out['min_encoding_indices'].shape)
            print(quant_out['min_encoding_indices'].reshape(N, -1).shape)
            print(quant_out['min_encoding_indices'].reshape(N, L, -1).shape)
            compressed_representation = quant_out['min_encoding_indices'].reshape(N, L, -1)

        elif self.quantize_scheme == "fsq":
            codebook = quant_out['codebook']
            print(codebook.shape)
            print(codebook.max())
            compressed_representation = codebook.reshape(-1, self.quantizer.num_dimensions)

        else:
            # no quantization, quant_out is the output of the encoder
            compressed_representation = quant_out
        
        # TODO: analysis with the latents
        return recons, loss, log_dict, compressed_representation 
    
    def _save_pdbs(self, struct_features, prefix=""):
        assert prefix in ["", "recons", "orig"]
        for i, pdbstr in enumerate(struct_features):
            with open(self.base_pdb_dir / f"/{prefix}_{i}.pdb", "w") as f: 
                f.write(pdbstr)
    
    def _structure_features_from_latent(self, latent_recons):
        shared_args = {
            "return_raw_features": True,
            "batch_size": self.batch_size,
            "num_recycles": self.num_recycles
        }
        recons_struct = self.structure_constructor.to_structure(latent_recons, self.sequences, **shared_args)
        orig_pred_struct = self.structure_constructor.to_structure(self.x, self.sequences, **shared_args)
        # only the first of the tuple is the structure feature 
        return recons_struct[0], orig_pred_struct[0]
    
    def __call__(self):
        # compress latent and reconstruct
        recons, loss, log_dict, compressed_representation = self._compress_and_reconstruct() 

        # coerce latent back into structure features for both reconstruction and the original prediction 
        recons_struct, orig_pred_struct = self._reconstruct_protein_from_latent(recons)
        self._save_pdbs(recons_struct, "recons")
        self._save_pdbs(orig_pred_struct, "orig")

        # TODO: also compare to the ground truth structure? 
    

if __name__ == "__main__":
    model = load_compression_model("2024-03-17T23-21-19")
    callback = CompressionReconstructionCallback(model, batch_size=4)
    import IPython;IPython.embed()