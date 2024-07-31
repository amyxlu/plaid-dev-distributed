from pathlib import Path
import argparse
from tqdm import tqdm
import h5py
import numpy as np
import torch
from plaid.datasets import CATHShardedDataModule
from plaid.esmfold.misc import batch_encode_sequences
from plaid.compression.hourglass_vq import HourglassVQLightningModule


#####################
# Config
#####################


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--compression_model_id", type=str, default="kyytc8i9")
    args = parser.parse_args()
    return args


device = torch.device("cuda")
shard_dir = "/data/lux70/data/cath/shards/"
ckpt_dir = Path("/data/lux70/cheap/checkpoints")
output_dir = f"/data/lux70/data/cath/compressed/"


#####################
# Helpers
#####################


def _maybe_make_dir(outpath):
    if not Path(outpath).parent.exists():
        Path(outpath).parent.mkdir(parents=True)


def save_compressed(fh, compressed, seqlen, cur_idx):
    for i in range(compressed.shape[0]):
        data = compressed[i, : seqlen[i], :].astype(np.float32)
        # ds = fh.create_dataset(str(cur_idx), data=data, dtype="f", compression="gzip")
        # ds = fh.create_dataset(str(cur_idx), data=data, dtype="f", compression="lzf")
        ds = fh.create_dataset(str(cur_idx), data=data, dtype="f")
        ds.attrs["sequence"] = ""
        cur_idx += 1
    return cur_idx


def run_batch(batch, fh, cur_idx):
    x = batch[0]
    _, mask, _, _, _ = batch_encode_sequences(batch[1])
    seqlens = mask.sum(dim=1)
    x = x.to(device)
    mask = mask.to(device).bool()

    compressed = model(x, mask, infer_only=True)
    cur_idx = save_compressed(fh, compressed, seqlens, cur_idx)
    return cur_idx


def make_h5_database(h5_path, dataloader, seq_len):
    # set global index per database
    print("Making h5 database at", h5_path)

    # open handle
    fh = h5py.File(h5_path, "w")
    cur_idx = 0

    # store metadata
    fh.attrs["max_seq_len"] = seq_len
    fh.attrs["shorten_factor"] = shorten_factor
    fh.attrs["compressed_hid_dim"] = hid_dim
    fh.attrs["dataset_size"] = len(dataloader.dataset)

    # writes each data point as its own dataset within the run batch method
    for batch in tqdm(
        dataloader, desc=f"Running through batches for for {len(dataloader.dataset):,} samples"
    ):
        cur_idx = run_batch(batch, fh, cur_idx)

    # close handle
    fh.close()


if __name__ == "__main__":
    args = parse_args()
    model = HourglassVQLightningModule.load_from_checkpoint(
        ckpt_dir / args.compression_model_id / "last.ckpt"
    )
    model = model.to(device)

    dm = CATHShardedDataModule(shard_dir=shard_dir, seq_len=args.seq_len, batch_size=args.batch_size)
    dm.setup()
    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()

    dirname = f"hourglass_{args.compression_model_id}"
    outdir = Path(output_dir) / dirname / f"seqlen_{args.seq_len}"
    train_h5_path = outdir / "train.h5"
    val_h5_path = outdir / "val.h5"
    _maybe_make_dir(train_h5_path)
    _maybe_make_dir(val_h5_path)

    # metadata to be saved
    hid_dim = 1024 // model.enc.downproj_factor
    shorten_factor = model.enc.shorten_factor

    make_h5_database(train_h5_path, train_dataloader, args.seq_len)
    make_h5_database(val_h5_path, val_dataloader, args.seq_len)
