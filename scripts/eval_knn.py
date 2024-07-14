import tqdm
import argparse

import sklearn
from pathlib import Path
import sklearn
import einops
import pandas as pd
import numpy as np
import torch

from plaid.compression.hourglass_vq import HourglassVQLightningModule
from plaid.esmfold.misc import batch_encode_sequences
from plaid.datasets import CATHShardedDataModule


device = torch.device("cuda")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_compress", action="store_true")
    parser.add_argument("--shard_dir", type=str, default="/homefs/home/lux70/storage/data/cath/shards")
    parser.add_argument("--cath_metadata_fpath", type=str, default="/homefs/home/lux70/storage/data/cath/description/cath-domain-list-S35.txt")
    parser.add_argument("--ckpt_dir", type=str, default="/homefs/home/lux70/storage/plaid/checkpoints/hourglass_vq/")
    parser.add_argument("--compression_id", type=str, default="identity")
    parser.add_argument("--seq_len", type=int, default=512)
    args = parser.parse_args()
    return args


def load_hourglass(ckpt_dir, compression_id):
    ckpt_dir = Path(ckpt_dir)
    ckpt_fpath = ckpt_dir / compression_id / "last.ckpt"
    hourglass = HourglassVQLightningModule.load_from_checkpoint(ckpt_fpath)

    hourglass.eval()
    for param in hourglass.parameters():
        param.requires_grad_(False)

    hourglass.to(device)
    return hourglass


def load_cath_metadata(fpath):
    df = pd.read_csv(fpath, sep="\s+", header=None)
    
    # from the README file
    columns = [
        "cath_id", # original name: "CATH domain name (seven characters)",
        "Class number",
        "Architecture number",
        "Topology number",
        "Homologous superfamily number",
        "S35 sequence cluster number",
        "S60 sequence cluster number",
        "S95 sequence cluster number",
        "S100 sequence cluster number",
        "S100 sequence count number",
        "Domain length",
        "Structure resolution (Angstroms)"
    ]
    
    df.columns = columns
    return df


def load_cath_cache_dataloaders(shard_dir, seq_len):
    dm = CATHShardedDataModule(
        shard_dir=shard_dir,
        seq_len=seq_len,
    )
    dm.setup()

    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()
    print(len(train_dataloader.dataset))
    print(len(val_dataloader.dataset))
    return train_dataloader, val_dataloader


def collect_batches(dataloader, hourglass, compress=True):
    sequences = []
    cath_ids = []
    
    all_x_c = []
    all_m_d = []

    print("dataset length:", len(dataloader.dataset))

    for batch in tqdm(dataloader):
        x, sequence, header = batch
        sequences.extend(sequence)
        cath_ids.extend(header)
        
        aatype, mask, _, _, _ = batch_encode_sequences(sequence)
    
        if not compress:
            assert hourglass is None
            x_c = x
            m_d = mask
        else:
            x = x.to(device)
            mask = mask.to(device)
            with torch.no_grad():
                x_c, m_d = hourglass(x, mask.bool(), infer_only=True)
        
        all_x_c.append(x_c.cpu().numpy())
        all_m_d.append(m_d.cpu().numpy())
    
    all_x_c = np.concatenate(all_x_c, axis=0)
    all_m_d = np.concatenate(all_m_d, axis=0)

    md_broadcast = einops.repeat(all_m_d, "n l -> n l c", c = all_x_c.shape[-1])
    xc_pooled = (all_x_c * md_broadcast).sum(axis=1) / md_broadcast.sum(axis=1)
    
    return {
        "x": all_x_c,
        "mask": all_m_d,
        "sequences": sequences,
        "cath_ids": cath_ids,
        "x_pooled": xc_pooled
    }


def run_knn(n_neighbors, target, train_data, train_df, val_data, val_df):
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X=train_data['x_pooled_ordered'], y=train_df[target].values)
    pred_classes = knn.predict(val_data['x_pooled_ordered'])
    correct = (pred_classes == val_df[target].values).sum() / len(val_df)
    return correct 


def main(args):
    df = load_cath_metadata(args.cath_metadata_fpath)
    if not args.no_compress:
        hourglass = load_hourglass(args.ckpt_dir, args.compression_id)
        shorten_factor = hourglass.enc.shorten_factor
        downproj_factor = hourglass.enc.downproj_factor
    else:
        hourglass = None
        shorten_factor = 1
        downproj_factor = 1

    train_dataloader, val_dataloader = load_cath_cache_dataloaders(args.shard_dir, args.seq_len)

    compress = not args.no_compress
    val_data = collect_batches(val_dataloader, hourglass, compress)
    train_data = collect_batches(train_dataloader, hourglass, compress)

    # create an dataframe to make it easier to manipulate
    train_embed_df = pd.DataFrame({"embedding_idx": np.arange(len(train_data['cath_ids'])), "cath_id": train_data['cath_ids']})
    val_embed_df = pd.DataFrame({"embedding_idx": np.arange(len(val_data['cath_ids'])), "cath_id": val_data['cath_ids']})

    # filter such that we only keep those with both metadata and cath_ids
    train_df = df[df.cath_id.isin(train_data['cath_ids'])]
    val_df = df[df.cath_id.isin(val_data['cath_ids'])]

    # join the dataframes
    train_df = train_df.set_index("cath_id").join(train_embed_df.set_index("cath_id"), how='left', rsuffix="embed_")
    val_df = val_df.set_index("cath_id").join(val_embed_df.set_index("cath_id"), how='left', rsuffix="embed_")

    train_df = train_df[~train_df.embedding_idx.isna()]
    val_df = val_df[~val_df.embedding_idx.isna()]

    # reorder the pooled embedding
    train_data['x_pooled_ordered'] = train_data['x_pooled'][train_df.embedding_idx.values]
    val_data['x_pooled_ordered'] = val_data['x_pooled'][val_df.embedding_idx.values]

    # Run knn experiments:
    results = pd.DataFrame(
        {"compression_model_id": [],
        "shorten_factor:": [],
        "downprojection_factor": [],
        "n_neighbors": [],
        "pred_target": [],
        "acc": []}
    )

    for n_neighbors in [1, 5]:
        for target in ["Class number", "Architecture number", "Topology number", "Homologous superfamily number"]:
            print("n_neighbors:", n_neighbors, "target:", target)
            correct = run_knn(n_neighbors, target, train_data, train_df, val_data, val_df)

            row = pd.DataFrame(
                {
                    "compression_model_id": [args.compression_id],
                    "shorten_factor:": [shorten_factor],
                    "downprojection_factor": [downproj_factor],
                    "n_neighbors": [n_neighbors],
                    "pred_target": [target],
                    "acc": [correct]
                }
            )
            results = pd.concat([results, row])
    
    results.to_csv(f"/homefs/home/lux70/storage/plaid/artifacts/eval/cath_knn/{args.compression_id}.csv", index=False)


if __name__ == "__main__":
    args = get_args()
    main(args)