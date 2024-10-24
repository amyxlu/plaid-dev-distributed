from pathlib import Path
import os

import wandb
from omegaconf import DictConfig, OmegaConf
import torch
import hydra

from plaid.esmfold import esmfold_v1
from plaid.evaluation import (
    RITAPerplexity,
)
from plaid.typed import PathLike



def default(val, default_val):
    return val if val is not None else default_val


def sample_run(sample_cfg: DictConfig):
    sample_latent = hydra.utils.instantiate(sample_cfg)
    sample_latent = sample_latent.run()  # returns npz path
    with open(sample_latent.outdir / "sample.yaml", "w") as f:
        OmegaConf.save(sample_cfg, f)
    return sample_latent


def decode_run(
    decode_cfg: DictConfig, npz_path: PathLike, output_root_dir: PathLike, esmfold=None, max_seq_len=None
):
    decode_latent = hydra.utils.instantiate(
        decode_cfg, npz_path=npz_path, output_root_dir=output_root_dir, esmfold=esmfold
    )
    seq_strs, pdb_paths = decode_latent.run()
    with open(output_root_dir / "decode_config.yaml", "w") as f:
        OmegaConf.save(decode_cfg, f)
    return seq_strs, pdb_paths


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


@hydra.main(config_path="../configs/pipeline", config_name="full")
def main(cfg: DictConfig):
    
    print(OmegaConf.to_yaml(cfg))

    sample_cfg = cfg.sample
    decode_cfg = cfg.decode
    inverse_generate_sequence_cfg = cfg.inverse_generate_sequence
    inverse_generate_structure_cfg = cfg.inverse_generate_structure
    phantom_generate_sequence_cfg = cfg.phantom_generate_sequence
    # NOTE: no phantom generate structure config, using all OmegaFold defaults.

    # ===========================
    # Sample
    # ===========================

    sample_latent = sample_run(sample_cfg)

    outdir = sample_latent.outdir
    uid = sample_latent.uid
    x = sample_latent.x

    # ===========================
    # Decode
    # ===========================

    if not cfg.run_decode:
        exit(0)

    esmfold = esmfold_v1()
    esmfold.eval().requires_grad_(False)
    esmfold.cuda()

    with torch.no_grad():
        seq_strs, pdb_paths = decode_run(
            decode_cfg,
            npz_path=outdir / "latent.npz",
            output_root_dir=outdir / "generated",
            esmfold=esmfold,
        )

    assert len(seq_strs) == len(pdb_paths) == x.shape[0]


    # ===========================
    # Inverse generations 
    # ===========================

    if cfg.run_cross_consistency:
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
    else:
        pass

    # ===========================
    # Phantom generations
    # ===========================

    if cfg.run_self_consistency:
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
    else:
        pass

    # ===========================
    # Analysis 
    # ===========================

    # TODO: make analysis step also optional
    from plaid.pipeline import run_analysis

    # run and save result CSV
    rita_perplexity = RITAPerplexity()
    df = run_analysis(outdir, rita_perplexity=rita_perplexity)

    # add wandb molecule object:
    wandb.init(
        project="plaid-sampling2",
        config=OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True),
        id=uid,
        resume="allow",
    )
    df['structure'] = [wandb.Molecule(str(pdbpath)) for pdbpath in pdb_paths]
    df['pdbpath'] = [str(pdbpath) for pdbpath in pdb_paths]
    wandb.log({"generations": wandb.Table(dataframe=df)})


if __name__ == "__main__":
    main()
