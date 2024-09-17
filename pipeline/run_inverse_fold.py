from plaid.pipeline._inverse_fold import InverseFoldPipeline
import hydra
from omegaconf import DictConfig


def run(cfg: DictConfig):
    """Hydra configurable instantiation, for imports in full pipeline."""
    inverse_fold = InverseFoldPipeline(
        pdb_dir=cfg.pdb_dir,
        output_fasta_path=cfg.output_fasta_path,
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


@hydra.main(config_path="../configs/pipeline", config_name="inverse_fold", version_base=None)
def hydra_run(cfg: DictConfig):
    """Hydra configurable instantiation for running as standalone script."""
    run(cfg)


if __name__ == "__main__":
    hydra_run()