from pathlib import Path
from plaid.pipeline._fold import FoldPipeline

import argparse

parser = argparse.ArgumentParser(description="Run ESMFold on a FASTA file.")
parser.add_argument("--fasta_file", type=str, help="Path to the input FASTA file.")
parser.add_argument("--outdir", type=str, default=None, help="Output directory for PDB files.")
parser.add_argument("--max_seq_len", type=int, default=None, help="Maximum sequence length.")
parser.add_argument("--batch_size", type=int, default=-1, help="Batch size for processing.")
parser.add_argument("--max_num_batches", type=int, default=None, help="Maximum number of batches to process.")
parser.add_argument("--shuffle", action="store_true", help="Shuffle the input sequences.")
args = parser.parse_args()


def run(esmfold=None):
    """Hydra configurable instantiation, for imports in full pipeline.
    
    Optional: if ESMFold was already loaded elsewhere, pass it as an argument to save on GPU memory.
    """

    fold_pipeline = FoldPipeline(
        fasta_file=Path(args.fasta_file),
        outdir=Path(args.outdir),
        esmfold=esmfold,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        max_num_batches=args.max_num_batches,
        shuffle=args.shuffle
    )
    fold_pipeline.run()


if __name__ == "__main__":
    run()

