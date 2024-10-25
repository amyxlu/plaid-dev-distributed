from pathlib import Path
from plaid.constants import PDB_DATABASE_PATH
import subprocess
import shutil


# query,target,evalue,gapopen,pident,fident,nident,qstart,qend,qlen
# tstart,tend,tlen,alnlen,raw,bits,cigar,qseq,tseq,qheader,theader,qaln,taln,mismatch,qcov,tcov
# qset,qsetid,tset,tsetid,taxid,taxname,taxlineage,
# lddt,lddtfull,qca,tca,t,u,qtmscore,ttmscore,alntmscore,rmsd,prob
# complexqtmscore,complexttmscore,complexu,complext,complexassignid

EASY_SEARCH_OUTPUT_COLS = [
    "query",
    "target",
    "theader",
    "evalue",
    "alntmscore",
    "rmsd",
    "lddt",
    "prob",
    "taxid",
    "taxname",
    "taxlineage"
]


def foldseek_easysearch(sample_dir: str, structure_subdir_name="designable"):
    sample_dir = Path(sample_dir)
    structures_dir = sample_dir / structure_subdir_name
    output_file = sample_dir / "foldseek_easysearch.m8"
    tmp_dir = sample_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    cmd = [
        "foldseek",
        "easy-search",
        str(structures_dir),
        PDB_DATABASE_PATH,
        output_file,
        str(tmp_dir),
        "--alignment-type",
        "1",  # TM mode align
        "--format-output",
        ",".join(EASY_SEARCH_OUTPUT_COLS)
    ]

    subprocess.run(cmd)
    shutil.rmtree(tmp_dir)
    return output_file


def foldseek_easycluster(sample_dir: str, structure_subdir_name="designable"):
    structures_dir = sample_dir / structure_subdir_name
    output_file = sample_dir / "foldseek_easycluster.m8"
    tmp_dir = sample_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    cmd = [
        "foldseek",
        "easy-cluster",
        str(structures_dir),
        output_file,
        "tmp",
        "--alignment-type",
        "2",
        "--tmscore-threshold",
        "0.5",
    ]

    subprocess.run(cmd)
    # shutil.rmtree(tmp_dir)

    return output_file
