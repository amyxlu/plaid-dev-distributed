from plaid.esmfold import esmfold_v1
esmfold = esmfold_v1()

from plaid.pipeline._fold import FoldPipeline

fasta_file = "/data/lux70/data/pfam/val.fasta"
# fasta_file = "/homefs/home/lux70/generated.fasta"
outdir = "/data/lux70/data/pfam/val_fold"
batch_size = 8 
max_seq_len = 768
max_num_batches = 200

fold_pipeline = FoldPipeline(
    fasta_file,
    outdir,
    esmfold=esmfold,
    max_seq_len=max_seq_len,
    batch_size=batch_size,
    max_num_batches=max_num_batches,
    shuffle=True
)