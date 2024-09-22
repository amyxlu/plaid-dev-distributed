from typing import Optional
from pathlib import Path
import glob
import os
import time

import wandb
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from safetensors import safe_open
import hydra

from plaid.esmfold import esmfold_v1
from plaid.evaluation import (
    parmar_fid,
    RITAPerplexity,
    CrossConsistencyEvaluation,
    SelfConsistencyEvaluation,
)
from plaid.typed import PathLike
from plaid.utils import extract_avg_b_factor_per_residue, parse_sequence_from_structure


def default(val, default_val):
    return val if val is not None else default_val


def inverse_generate_structure_run(
    inverse_generate_structure_cfg: DictConfig,
    fasta_file,
    outdir,
    esmfold: torch.nn.Module = None,
):
    fold_pipeline = hydra.utils.instantiate(
        inverse_generate_structure_cfg,
        fasta_file=fasta_file,
        outdir=outdir,
        esmfold=esmfold,
    )
    fold_pipeline.run()
    with open(outdir / "fold_config.yaml", "w") as f:
        OmegaConf.save(inverse_generate_structure_cfg, f)


def inverse_generate_sequence_run(
    inverse_generate_sequence_cfg: DictConfig, pdb_dir=None, output_fasta_path=None
):
    """Hydra configurable instantiation, for imports in full pipeline."""
    inverse_fold = hydra.utils.instantiate(
        inverse_generate_sequence_cfg,
        pdb_dir=pdb_dir,
        output_fasta_path=output_fasta_path,
    )
    inverse_fold.run()
    with open(Path(output_fasta_path).parent / "inverse_fold_config.yaml", "w") as f:
        OmegaConf.save(inverse_generate_sequence_cfg, f)


def phantom_generate_structure_run(sample_dir):
    # generated structure to inverse-generated sequence
    cmd = f"""omegafold {str(sample_dir / "inverse_generated/sequences.fasta")} {str(sample_dir / "phantom_generated/structures")} --subbatch_size 64"""
    os.system(cmd)


def phantom_generate_sequence_run(
    phantom_generate_sequence_cfg, pdb_dir, output_fasta_path
):
    inverse_fold = hydra.utils.instantiate(
        phantom_generate_sequence_cfg,
        pdb_dir=pdb_dir,
        output_fasta_path=output_fasta_path,
    )
    inverse_fold.run()
    with open(
        Path(output_fasta_path).parent / "phantom_generate_sequence_config.yaml", "w"
    ) as f:
        OmegaConf.save(phantom_generate_sequence_cfg, f)
    return inverse_fold


@hydra.main(config_path="configs/pipeline", config_name="consistency")
def main(cfg: DictConfig):
    inverse_generate_sequence_cfg = cfg.inverse_generate_sequence
    inverse_generate_structure_cfg = cfg.inverse_generate_structure
    phantom_generate_sequence_cfg = cfg.phantom_generate_sequence

    outdir = Path(cfg.samples_dir)

    import IPython;IPython.embed()

    wandb.init(
        project="plaid-sampling",
        name=cfg.wandb_job_name
    )

    perplexity_calc = RITAPerplexity()

    samples_d = {
        "structure": [],
        "sequence": [],
        "perplexity": [],
    }

    pdb_paths = glob.glob(str(outdir / "generated/structures/*.pdb"))
    seq_strs = [parse_sequence_from_structure(pdb_path=pdb_path) for pdb_path in pdb_paths]
    num_samples = len(pdb_paths)

    for i in range(num_samples):
        seqstr, pdbpath = seq_strs[i], pdb_paths[i]
        samples_d["structure"].append(wandb.Molecule(str(pdbpath)))
        samples_d["sequence"].append(seqstr)

        # calculate perplexity
        perplexity = perplexity_calc.calc_perplexity(seqstr)
        samples_d["perplexity"].append(perplexity)

    esmfold = esmfold_v1()
    esmfold.eval().requires_grad_(False)
    esmfold.cuda()

    try:
        # run ProteinMPNN for generated structures
        input_pdb_dir = outdir / "generated/structures"
        output_fasta_path = outdir / "inverse_generated/sequences.fasta"
        inverse_generate_sequence_run(
            inverse_generate_sequence_cfg,
            pdb_dir=input_pdb_dir,
            output_fasta_path=output_fasta_path,
        )

        # run ESMFold for generated sequences
        input_fasta_file = outdir / "generated" / "sequences.fasta"
        structure_outdir = outdir / "inverse_generated" / "structures"
        inverse_generate_structure_run(
            inverse_generate_structure_cfg,
            fasta_file=input_fasta_file,
            outdir=structure_outdir,
            esmfold=esmfold,
        )

        cross_consistency_calc = CrossConsistencyEvaluation(outdir)

        # should be already sorted in the same order
        ccrmsd = cross_consistency_calc.cross_consistency_rmsd()
        # ccrmspd = cross_consistency_calc.cross_consistency_rmspd()
        cctm = cross_consistency_calc.cross_consistency_tm()
        ccsr = cross_consistency_calc.cross_consistency_sr()

        wandb.log(
            {
                # detailed stats for ccRMSD and ccSR as representative metrics
                "ccrmsd_mean": np.mean(ccrmsd),
                "ccrmsd_std": np.std(ccrmsd),
                "ccrmsd_median": np.std(ccrmsd),
                "ccsr_mean": np.mean(ccsr),
                "ccsr_std": np.std(ccsr),
                "ccsr_median": np.std(ccsr),
                # also log histogram
                "cctm_hist": wandb.Histogram(cctm),
                "ccrmsd_hist": wandb.Histogram(ccrmsd),
                # "ccrmspd_hist": wandb.Histogram(ccrmspd),
                "ccsr_hist": wandb.Histogram(ccsr),
            }
        )

        # add to the big table

        samples_d["ccrmsd"] = ccrmsd
        # samples_d["ccrmspd"] = ccrmspd
        samples_d["cctm"] = cctm
        samples_d["ccsr"] = ccsr

    except:
        # log our table with whatever we got to:
        wandb.log({"generations": wandb.Table(dataframe=pd.DataFrame(samples_d))})
        pass

    # ===========================
    # Phantom generations for self-consistency
    # ===========================


    try:
        # run ProteinMPNN on the structure predictions of our generated sequences to look at self-consistency sequence recovery
        if not (outdir / "phantom_generated").exists():
            Path(outdir / "phantom_generated").mkdir(parents=True)

        input_pdb_dir = outdir / "inverse_generated/structures"
        output_fasta_path = outdir / "phantom_generated/sequences.fasta"
        phantom_generate_sequence_run(
            phantom_generate_sequence_cfg,
            pdb_dir=input_pdb_dir,
            output_fasta_path=output_fasta_path,
        )

        # uses OmegaFold to fold the inverse-fold sequence predictions of generated structures to look at scTM and scRMSD
        phantom_generate_structure_run(outdir)

        self_consistency_calc = SelfConsistencyEvaluation(outdir)

        self_consistency_rmsd = self_consistency_calc.self_consistency_rmsd()
        # self_consistency_rmspd = self_consistency_calc.cross_consistency_rmspd()
        self_consistency_tm = self_consistency_calc.self_consistency_tm()
        self_consistency_sr = self_consistency_calc.self_consistency_sr()

        wandb.log(
            {
                # detailed stats for self-consistency RMSD and SR as representative metrics
                "self_consistency_rmsd_mean": np.mean(self_consistency_rmsd),
                "self_consistency_rmsd_std": np.std(self_consistency_rmsd),
                "self_consistency_rmsd_median": np.median(self_consistency_rmsd),
                "self_consistency_sr_mean": np.mean(self_consistency_sr),
                "self_consistency_sr_std": np.std(self_consistency_sr),
                "self_consistency_sr_median": np.median(self_consistency_sr),
                # also log histogram
                "self_consistency_tm_hist": wandb.Histogram(self_consistency_tm),
                "self_consistency_rmsd_hist": wandb.Histogram(self_consistency_rmsd),
                # "self_consistency_rmspd_hist": wandb.Histogram(self_consistency_rmspd),
                "self_consistency_sr_hist": wandb.Histogram(self_consistency_sr),
            }
        )

        # add to the big table
        samples_d["self_consistency_rmsd"] = self_consistency_rmsd
        # samples_d["self_consistency_rmspd"] = self_consistency_rmspd
        samples_d["self_consistency_tm"] = self_consistency_tm
        samples_d["self_consistency_sr"] = self_consistency_sr

    except:
        pass

    # log our big table
    wandb.log({"generations": wandb.Table(dataframe=pd.DataFrame(samples_d))})


if __name__ == "__main__":
    main()
