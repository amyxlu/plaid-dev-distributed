from typing import Optional
from pathlib import Path
from plaid.pipeline import SampleLatent, DecodeLatent, FoldPipeline, InverseFoldPipeline
import os
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from plaid.esmfold import esmfold_v1
from safetensors import safe_open
from plaid.evaluation import parmar_fid
from plaid.typed import PathLike

"""
DDIM T=500.
"""

cond_code = "f2219_o3617"

sample_cfg = OmegaConf.load(
    "/homefs/home/lux70/code/plaid/configs/pipeline/sample_latent_config/ddim_unconditional.yaml"
)

decode_cfg = OmegaConf.load(
    "/homefs/home/lux70/code/plaid/configs/pipeline/decode_config/decode_latent.yaml"
)

inverse_generate_sequence_cfg = OmegaConf.load(
    "/homefs/home/lux70/code/plaid/configs/pipeline/inverse_generate_sequence_config/inverse_fold.yaml"
)

inverse_generate_structure_cfg = OmegaConf.load(
    "/homefs/home/lux70/code/plaid/configs/pipeline/inverse_generate_structure_config/esmfold.yaml"
)

phantom_generate_sequence_cfg = OmegaConf.load(
    "/homefs/home/lux70/code/plaid/configs/pipeline/phantom_generate_sequence_config/default.yaml"
)


def default(val, default_val):
    return val if val is not None else default_val


def sample_run(cfg: DictConfig):
    sample_latent = SampleLatent(
        model_id=cfg.model_id,
        model_ckpt_dir=cfg.model_ckpt_dir,
        organism_idx=cfg.organism_idx,
        function_idx=cfg.function_idx,
        cond_scale=cfg.cond_scale,
        num_samples=cfg.num_samples,
        beta_scheduler_name=cfg.beta_scheduler_name,
        sampling_timesteps=cfg.sampling_timesteps,
        batch_size=cfg.batch_size,
        length=cfg.length,
        return_all_timesteps=cfg.return_all_timesteps,
        output_root_dir=cfg.output_root_dir,
    )
    npz_path = sample_latent.run()  # returns npz path
    with open(npz_path.parent / "sample.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    return npz_path


def decode_run(cfg: DictConfig, npz_path: Optional[PathLike] = None, esmfold=None):
    npz_path = default(npz_path, cfg.npz_path)
    output_root_dir = npz_path.parent / "generated"

    """Hydra configurable instantiation, for imports in full pipeline."""
    decode_latent = DecodeLatent(
        npz_path=npz_path,
        output_root_dir=output_root_dir,
        num_recycles=cfg.num_recycles,
        batch_size=cfg.batch_size,
        device=cfg.device,
        esmfold=esmfold,
    )
    decode_latent.run()
    with open(output_root_dir / "decode_config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    
    return decode_latent


def inverse_generate_structure_run(cfg: DictConfig, fasta_file=None, outdir=None, esmfold: torch.nn.Module = None):
    fasta_file = default(fasta_file, cfg.fasta_file)
    outdir = default(outdir, cfg.outdir)
    fold_pipeline = FoldPipeline(
        fasta_file=fasta_file,
        outdir=outdir,
        esmfold=esmfold,
        max_seq_len=cfg.max_seq_len,
        batch_size=cfg.batch_size,
        max_num_batches=cfg.max_num_batches,
        shuffle=cfg.shuffle,
    )
    fold_pipeline.run()
    with open(outdir / "fold_config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    return fold_pipeline


def inverse_generate_sequence_run(cfg: DictConfig, pdb_dir=None, output_fasta_path=None):
    """Hydra configurable instantiation, for imports in full pipeline."""
    pdb_dir = default(pdb_dir, cfg.pdb_dir)
    output_fasta_path = default(output_fasta_path, cfg.output_fasta_path)
    inverse_fold = InverseFoldPipeline(
        pdb_dir=pdb_dir,
        output_fasta_path=output_fasta_path,
        model_name=cfg.model_name,
        ca_only=cfg.ca_only,
        num_seq_per_target=cfg.num_seq_per_target,
        sampling_temp=cfg.sampling_temp,
        save_score=cfg.save_score,
        save_probs=cfg.save_probs,
        score_only=cfg.score_only,
        conditional_probs_only=cfg.conditional_probs_only,
        conditional_probs_only_backbone=cfg.conditional_probs_only_backbone,
    )
    inverse_fold.run()
    with open(Path(output_fasta_path).parent / "inverse_fold_config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    return inverse_fold


def phantom_generate_structure_run(sample_dir):
    # generated structure to inverse-generated sequence
    cmd = f"""omegafold {str(sample_dir / "inverse_generated/sequences.fasta")} {str(sample_dir / "phantom_generated/structures")} --subbatch_size 64""" 
    os.system(cmd)


def phantom_generate_sequence_run(cfg, pdb_dir=None, output_fasta_path=None):
    pdb_dir = default(pdb_dir, cfg.pdb_dir)
    output_fasta_path = default(output_fasta_path, cfg.output_fasta_path)
    inverse_fold = InverseFoldPipeline(
        pdb_dir=pdb_dir,
        output_fasta_path=output_fasta_path,
        model_name=cfg.model_name,
        ca_only=cfg.ca_only,
        num_seq_per_target=cfg.num_seq_per_target,
        sampling_temp=cfg.sampling_temp,
        save_score=cfg.save_score,
        save_probs=cfg.save_probs,
        score_only=cfg.score_only,
        conditional_probs_only=cfg.conditional_probs_only,
        conditional_probs_only_backbone=cfg.conditional_probs_only_backbone,
    )
    inverse_fold.run()
    with open(Path(output_fasta_path).parent / "inverse_fold_config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    return inverse_fold


if __name__ == "__main__":
    # # ===========================
    # # Sample configuration
    # # ===========================
    # npz_path = sample_run(sample_cfg)

    # x = np.load(npz_path)["samples"]

    # with open(npz_path.parent / "sample_config.yaml", "w") as f:
    #     OmegaConf.save(sample_cfg, f)

    # # ===========================
    # # FID calculation
    # # ===========================
    # gt_path = "/data/lux70/data/pfam/features/all.pt"

    # with safe_open(gt_path, "pt") as f:
    #     gt = f.get_tensor("features").numpy()

    # # randomly sample x
    # idx = np.random.choice(gt.shape[0], size=sample_cfg.num_samples, replace=False)
    # gt = gt[idx]

    # feat = x[:, -1, :, :].mean(axis=1)
    # fid = parmar_fid(feat, gt)
    # with open(npz_path.parent / "fid.txt", "w") as f:
    #     f.write(str(fid))

    # print(fid)

    # # ===========================
    # # Decode
    # # ===========================
    # esmfold = esmfold_v1()
    # esmfold.eval().requires_grad_(False)
    # esmfold.cuda()

    # decode_run(decode_cfg, npz_path=npz_path, esmfold=esmfold)

    # # ===========================
    # # Inverse generations for cross-consistency
    # # ===========================

    # # run ProteinMPNN for generated structures
    # input_pdb_dir = npz_path.parent / "generated/structures"
    # output_fasta_path = npz_path.parent / "inverse_generated/sequences.fasta"
    # inverse_generate_sequence_run(inverse_generate_sequence_cfg, pdb_dir=input_pdb_dir, output_fasta_path=output_fasta_path)

    # # run ESMFold for generated sequences
    # input_fasta_file = npz_path.parent / "generated" / "sequences.fasta"
    # structure_outdir = npz_path.parent / "inverse_generated" / "structures"
    # inverse_generate_structure_run(inverse_generate_structure_cfg, fasta_file=input_fasta_file, outdir=structure_outdir, esmfold=esmfold)

    # ===========================
    # Phantom generations for self-consistency
    # ===========================

    npz_path = Path("/data/lux70/plaid/artifacts/samples/5j007z42/ddim/5j007z42/f2219_o3617/240917_0619/latent.npz")

    # run ProteinMPNN on the structure predictions of our generated sequences to look at self-consistency sequence recovery
    # if not (npz_path.parent / "phantom_generated").exists():
    #     Path(npz_path.parent / "phantom_generated").mkdir(parents=True)

    input_pdb_dir = npz_path.parent / "inverse_generated/structures"
    output_fasta_path = npz_path.parent / "phantom_generated/sequences.fasta"
    phantom_generate_sequence_run(inverse_generate_sequence_cfg, pdb_dir=input_pdb_dir, output_fasta_path=output_fasta_path) 

    # uses OmegaFold to fold the inverse-fold sequence predictions of generated structures to look at scTM and scRMSD
    phantom_generate_structure_run(npz_path.parent)
