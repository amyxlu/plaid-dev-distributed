from plaid.pipeline._fold import FoldPipeline
import hydra
from omegaconf import DictConfig


def run(cfg: DictConfig, esmfold=None):
    """Hydra configurable instantiation, for imports in full pipeline.
    
    Optional: if ESMFold was already loaded elsewhere, pass it as an argument to save on GPU memory.
    """

    fold_pipeline = FoldPipeline(
        fasta_file=cfg.fasta_file,
        outdir=cfg.outdir,
        esmfold=esmfold,
        max_seq_len=cfg.max_seq_len,
        batch_size=cfg.batch_size,
        max_num_batches=cfg.max_num_batches,
        shuffle=cfg.shuffle
    )
    fold_pipeline.run()


@hydra.main(config_path="../configs/pipeline", config_name="esmfold", version_base=None)
def hydra_run(cfg: DictConfig):
    """Hydra configurable instantiation for running as standalone script."""
    run(cfg)


if __name__ == "__main__":
    hydra_run()