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

"""
Loop through a set of GO terms, for the specified organism.
"""

@hydra.main(config_path="configs/pipeline", config_name="conditional")
def main(cfg: DictConfig):
    df = pd.read_parquet("/data/lux70/data/pfam/val.parquet")

    # possibly mutate these configs later
    sample_cfg = cfg.sample
    decode_cfg = cfg.decode

    if cfg.GO_term_contains is not None:
        df = df[df.GO_term.str.contains(cfg.GO_term_contains)]

    print("Unique GO terms in val dataset:", df.GO_idx.unique())

    unique_go_idxs = df.GO_idx.unique()
    start_idx = cfg.start_idx if cfg.start_idx != -1 else 0
    end_idx = cfg.end_idx if cfg.end_idx != -1 else len(unique_go_idxs)

    if not cfg.no_log_wandb:
        from plaid.evaluation import RITAPerplexity
        perplexity_calc = RITAPerplexity()

    # if looping through GO indices:
    for i, idx in enumerate(unique_go_idxs):
        tmp_df = df[df.GO_idx == idx]

        if (start_idx < i < end_idx): # and len(tmp_df > 512):
            go_term = tmp_df.iloc[0].GO_term
            print("Current ID:", idx, go_term)

            # run, maybe:
            median_len = int(tmp_df.seq_len.median())
            sample_len = round_to_multiple(median_len / 2, multiple=4)

            sample_cfg.function_idx = int(idx)
            sample_cfg.length = int(sample_len)

            sample(sample_cfg, decode_cfg, go_term_to_log=go_term, no_log_wandb=cfg.no_log_wandb, perplexity_calc=perplexity_calc)
    
    # # if looping through organisms:
    # elif cfg.loop_type == "organism":
    #     for i, organism_idx in enumerate(unique_organism_idxs):
    #         tmp_df = df[df.organism == organism_idx]

    #         if (start_idx < i < end_idx): # and (len(tmp_df) > 512):
    #             organism_name = tmp_df.iloc[0].organism
    #             print("Current Organism:", organism_idx, organism_name)

    #             median_len = int(tmp_df.seq_len.median())
    #             sample_len = round_to_multiple(median_len / 2, multiple=4)

    #             # for each unique organism, keep function as the "unconditional" index and use median length
    #             sample_cfg.organism_idx = organism_idx
    #             sample_cfg.length = int(sample_len)

    #             sample(sample_cfg, decode_cfg, organism_to_log=organism_idx)

    # else:
    #     raise ValueError("Invalid loop_type. Choose either 'organism' or 'function'.")


def sample(sample_cfg: DictConfig, decode_cfg: DictConfig, go_term_to_log=None, organism_to_log=None, no_log_wandb=False, perplexity_calc=None):
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

    if not no_log_wandb:
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
            assert not perplexity_calc is None, "Perplexity calculator not provided."
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