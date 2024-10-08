import subprocess

PDB_DATABASE = "/data/bucket/robins21/pdb"

def foldseek_easy_search(i, o, filtered=None):
    if filtered != None:
        i = "/homefs/home/robins21/fold_seek_results_/temp_foldseek/"
    cmd = [
        "foldseek",
        "easy-search",
        i,
        PDB_DATABASE,
        o,
        "--alignment-type",
        "1",
        "--format-output",
        "query,target,theader,evalue,alntmscore,qtmscore,ttmscore,rmsd,lddt,prob",
        "tmp",
    ]
    subprocess.run(cmd)


def mmseqs_easy_search(inputfile):
    cmd = ["mmseqs", "easy-search", inputfile, "/shared/amyxlu/data/uniref50db", "aln", "tmp"]
    subprocess.run(cmd)

