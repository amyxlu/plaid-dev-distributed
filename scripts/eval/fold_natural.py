from plaid.esmfold import esmfold_v1
esmfold = esmfold_v1()
esmfold.set_chunk_size(64)

from plaid.pipeline._fold import FoldPipeline

fasta_file = "/data/lux70/data/pfam/val.fasta"
outdir = "/data/lux70/data/pfam/val_stats/folded"
batch_size = 32 
max_seq_len = 256 
max_num_batches = 200
num_recycles = 4 


fold_pipeline = FoldPipeline(
    fasta_file,
    outdir,
    esmfold=esmfold,
    max_seq_len=max_seq_len,
    batch_size=batch_size,
    max_num_batches=max_num_batches,
    num_recycles=num_recycles,
    shuffle=True
)

fold_pipeline.run()