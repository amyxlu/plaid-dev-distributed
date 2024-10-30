from pathlib import Path    
import shutil
import subprocess

from ..constants import UNIREF_DATABASE_PATH


EASY_SEARCH_OUTPUT_COLS = [
    "query",
    "target",
    "evalue",
    "alnlen",
    "qseq",
    "tseq",
    "qheader",
    "theader",
    "taxid",
    "taxname",
    "taxlineage",
]

# query,target,evalue,gapopen,pident,fident,nident,qstart,qend,qlen
# tstart,tend,tlen,alnlen,raw,bits,cigar,qseq,tseq,qheader,theader,qaln,taln,qframe,tframe,mismatch,qcov,tcov
# qset,qsetid,tset,tsetid,taxid,taxname,taxlineage,qorfstart,qorfend,torfstart,torfend


def mmseqs_easysearch(sample_dir, fasta_file_name="generated/sequences.fasta"):
    sample_dir = Path(sample_dir)
    fasta_file = sample_dir / fasta_file_name
    output_file = sample_dir / "mmseqs_easysearch.m8"
    tmp_dir = sample_dir / "tmp" 

    cmd = [
        "mmseqs",
        "easy-search",
        str(fasta_file),
        UNIREF_DATABASE_PATH,
        str(output_file),
        str(tmp_dir),
        "--format-output",
        ",".join(EASY_SEARCH_OUTPUT_COLS)
        "--split-memory-limit",
        "10G"
    ]

    subprocess.run(cmd)
    shutil.rmtree(tmp_dir)

    return output_file


def mmseqs_easycluster(sample_dir, fasta_file_name="generated/sequences.fasta"):
    sample_dir = Path(sample_dir)
    fasta_file = sample_dir / fasta_file_name
    output_file = sample_dir / "mmseqs_easycluster.m8"
    tmp_dir = sample_dir / "tmp"

    cmd = [
        "mmseqs",
        "easy-cluster",
        str(fasta_file),
        str(output_file),
        str(tmp_dir),
    ]

    subprocess.run(cmd)
    # shutil.rmtree(tmp_dir)

    return output_file
