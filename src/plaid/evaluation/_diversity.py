"""
https://github.com/jasonkyuyim/multiflow/blob/main/multiflow/experiments/utils.py#L1
"""
import os
import subprocess
from biotite.sequence.io import fasta


def run_easy_cluster(designable_dir, output_dir):
    # designable_dir should be a directory with individual PDB files in it that we want to cluster
    # output_dir is where we are going to save the easy cluster output files

    # Returns the number of clusters

    easy_cluster_args = [
        'foldseek',
        'easy-cluster',
        designable_dir,
        os.path.join(output_dir, 'res'),
        output_dir,
        '--alignment-type',
        '1',
        '--cov-mode',
        '0',
        '--min-seq-id',
        '0',
        '--tmscore-threshold',
        '0.5',
    ]
    process = subprocess.Popen(
        easy_cluster_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, _ = process.communicate()
    del stdout # We don't actually need the stdout, we will read the number of clusters from the output files
    rep_seq_fasta = fasta.FastaFile.read(os.path.join(output_dir, 'res_rep_seq.fasta'))
    return len(rep_seq_fasta)
 
