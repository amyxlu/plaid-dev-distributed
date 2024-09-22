"""
bespoke script for looping through conditional generation by organism.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf, DictConfig
import hydra

from plaid.utils import round_to_multiple
from plaid.datasets import NUM_ORGANISM_CLASSES, NUM_FUNCTION_CLASSES

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--loop_type", type=str, choices=["organism", "function"], required=True)
    parser.add_argument("--start_idx", type=int, default=-1)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--GO_term_contains", type=str, default=None, help="substring to filter GO terms. Can only provide one, unlike organism_indices.")
    parser.add_argument("--organism_indices", type=str, default=None, help="comma separated list of organism indices")
    parser.add_argument("--sample_cfg_path", type=str, default="/homefs/home/lux70/code/plaid/configs/pipeline/sample/sample_latent.yaml")
    parser.add_argument("--decode_cfg_path", type=str, default="/homefs/home/lux70/code/plaid/configs/pipeline/decode/default.yaml")
    parser.add_argument("--cond_scale", type=float, default=7.5)
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--no_log_wandb", action='store_true', help="Don't log results to Weights and Biases")
    return parser.parse_args()

args = get_args()
sample_cfg = OmegaConf.load(args.sample_cfg_path)
decode_cfg = OmegaConf.load(args.decode_cfg_path)

if sample_cfg.function_idx is None:
    sample_cfg.function_idx = NUM_FUNCTION_CLASSES

if sample_cfg.organism_idx is None:
    sample_cfg.organism_idx = NUM_ORGANISM_CLASSES

sample_cfg.cond_scale = float(args.cond_scale)
sample_cfg.sampling_timesteps = int(args.timesteps)

def main():
    df = pd.read_parquet("/data/lux70/data/pfam/val.parquet")

    if args.GO_term_contains is not None:
        df = df[df.GO_term.str.contains(args.GO_term_contains)]

    if args.organism_indices is not None:
        organism_idxs = [int(x) for x in args.organism_indices.split(",")]
        df = df[df.organism_index.isin(organism_idxs)]

    print("Unique GO terms in val dataset:", df.GO_idx.unique())
    print("Unique organisms in val dataset:", df.organism.unique())

    unique_go_idxs = df.GO_idx.unique()
    unique_organism_idxs = df.organism_index.unique()

    start_idx = args.start_idx if args.start_idx != -1 else 0
    end_idx = args.end_idx if args.end_idx != -1 else len(unique_go_idxs)

    """
    Loop through organisms or functions based on the loop_type argument
    """

    # if looping through GO indices:
    if args.loop_type == "function":
        for i, idx in enumerate(unique_go_idxs):
            tmp_df = df[df.GO_idx == idx]

            if (start_idx < i < end_idx): # and len(tmp_df > 512):
                go_term = tmp_df.iloc[0].GO_term
                print("Current ID:", idx, go_term)

                # run, maybe:
                median_len = int(tmp_df.seq_len.median())
                sample_len = round_to_multiple(median_len / 2, multiple=4)

                # for each unique GO term, keep organism as the "unconditional" index and use median length
                sample_cfg.function_idx = int(idx)
                sample_cfg.length = int(sample_len)

                sample(sample_cfg, decode_cfg, go_term_to_log=go_term)
    
    # if looping through organisms:
    elif args.loop_type == "organism":
        for i, organism_idx in enumerate(unique_organism_idxs):
            tmp_df = df[df.organism == organism_idx]

            if (start_idx < i < end_idx): # and (len(tmp_df) > 512):
                organism_name = tmp_df.iloc[0].organism
                print("Current Organism:", organism_idx, organism_name)

                median_len = int(tmp_df.seq_len.median())
                sample_len = round_to_multiple(median_len / 2, multiple=4)

                # for each unique organism, keep function as the "unconditional" index and use median length
                sample_cfg.organism_idx = organism_idx
                sample_cfg.length = int(sample_len)

                sample(sample_cfg, decode_cfg, organism_to_log=organism_idx)

    else:
        raise ValueError("Invalid loop_type. Choose either 'organism' or 'function'.")


def sample(sample_cfg: DictConfig, decode_cfg: DictConfig, go_term_to_log=None, organism_to_log=None):
    # Sample the latent, if it doesn't exist
    sample_latent = hydra.utils.instantiate(sample_cfg)
    print(sample_latent.outdir)

    if not sample_latent.outdir.parent.exists():
        sample_latent = sample_latent.run()
    else:
        return
    
    # Decode the latent
    npz_path = sample_latent.outdir / "latent.npz"
    output_root_dir = sample_latent.outdir / "generated"
    decode_latent = hydra.utils.instantiate(
        decode_cfg,
        npz_path=npz_path,
        output_root_dir=output_root_dir,
    )
    seq_strs, pdb_paths = decode_latent.run()

    if not args.no_log_wandb:
        from plaid.evaluation import RITAPerplexity
        import wandb
        from plaid.utils import extract_avg_b_factor_per_residue

        wandb.init(
            project="plaid-sampling",
            config=OmegaConf.to_container(sample_cfg, throw_on_missing=True, resolve=True),
            id=sample_latent.uid,
            resume="allow",
        )

        if go_term_to_log is not None:
            wandb.log({"GO_term": go_term_to_log})
        if organism_to_log is not None:
            wandb.log({"organism": organism_to_log})

        perplexity_calc = RITAPerplexity()

        samples_d = {
            "structure": [],
            "sequence": [],
            "plddt": [],
            "perplexity": [],
        }

        for i in range(len(pdb_paths)):
            seqstr, pdbpath = seq_strs[i], pdb_paths[i]
            samples_d["structure"].append(wandb.Molecule(str(pdbpath)))
            samples_d["sequence"].append(seqstr)

            # extract pLDDT scores
            plddts = extract_avg_b_factor_per_residue(pdbpath)
            samples_d["plddt"].append(np.mean(plddts))  # average across the pLDDT scores

            # calculate perplexity
            perplexity = perplexity_calc.calc_perplexity(seqstr)
            samples_d["perplexity"].append(perplexity)

        wandb.log(
            {
                "generations": wandb.Table(dataframe=pd.DataFrame(samples_d)),
            }
        )

        wandb.log(
            {
                "plddt_mean": np.mean(samples_d["plddt"]),
                "plddt_std": np.std(samples_d["plddt"]),
                "plddt_median": np.median(samples_d["plddt"]),
                "plddt_hist": wandb.Histogram(samples_d["plddt"]),
            }
        )

        wandb.log(
            {
                "perplexity_mean": np.mean(samples_d["perplexity"]),
                "perplexity_std": np.std(samples_d["perplexity"]),
                "perplexity_median": np.median(samples_d["perplexity"]),
                "perplexity_hist": wandb.Histogram(samples_d["perplexity"]),
            }
        )




if __name__ == "__main__":
    main()
    # #1326,818,2436,300,1357
    # sample_cfg.organism_idx = 1326
    # organism_to_log = "HUMAN"
    # sample(sample_cfg, decode_cfg, organism_to_log=organism_to_log)  