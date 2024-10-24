import argparse
from pathlib import Path
import subprocess
import shutil

from plaid.constants import PDB_DATABASE_PATH
from plaid.evaluation._foldseek import foldseek_easysearch, foldseek_easycluster

parser = argparse.ArgumentParser()
parser.add_argument('--sample_dir', default="/data/lux70/plaid/artifacts/samples/5j007z42/val_dist//data/lux70/plaid/artifacts/samples/5j007z42/val_dist/f989_o1326_l144_s3")
args = parser.parse_args()

# TODO: run this only on designable structures
# "generated/structures/designable"
foldseek_easycluster(args.sample_dir, "generated/structures")
foldseek_easysearch(args.sample_dir, "generated/structures")

# sample_dir = Path(args.sample_dir)
# structures_dir = sample_dir / "generated/structures"
# easy_search_output_file = sample_dir / "foldseek_easysearch.m8"
# easy_cluster_output_file = sample_dir / "foldseek_easycluster.m8"
# tmp_dir = sample_dir / "tmp"
# tmp_dir.mkdir(exist_ok=True)

# query,target,evalue,gapopen,pident,fident,nident,qstart,qend,qlen
#                                 tstart,tend,tlen,alnlen,raw,bits,cigar,qseq,tseq,qheader,theader,qaln,taln,mismatch,qcov,tcov
#                                 qset,qsetid,tset,tsetid,taxid,taxname,taxlineage,
#                                 lddt,lddtfull,qca,tca,t,u,qtmscore,ttmscore,alntmscore,rmsd,prob
#                                 complexqtmscore,complexttmscore,complexu,complext,complexassignid
#                                  [query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits]

# EASY_SEARCH_OUTPUT_COLS = [
#     "query",
#     "target",
#     "theader",
#     "evalue",
#     "alntmscore",
#     "rmsd",
#     "lddt",
#     "prob",
#     "taxid",
#     "taxname",
#     "taxlineage"
# ]


# cmd = [
#     "foldseek",
#     "easy-search",
#     str(structures_dir),
#     PDB_DATABASE_PATH,
#     easy_search_output_file,
#     str(tmp_dir),
#     "--alignment-type",
#     "1",  # TM mode align
#     "--format-output",
#     ",".join(EASY_SEARCH_OUTPUT_COLS)
# ]

# subprocess.run(cmd)

# import shutil
# shutil.rmtree(tmp_dir)



# run_foldseek_and_analysis(
#     input_folder=args.sample_dir,
#     next_folder="generated/structures",  # hack since some folders are organized by length
#     outputfolder=str(Path(args.sample_dir) / "foldseek_outputs"),
#     outputfile=str(Path(args.sample_dir) / "foldseek.csv"),
#     run_diversity=True,#args.no_diversity,
#     run_novelty=True,#args.no_novelty,
#     use_filter=True,#args.no_filter,
# )


# cmd = [
#     "foldseek",
#     "easy-cluster",
#     str(structures_dir),
#     easy_cluster_output_file,
#     "tmp",
#     "--alignment-type",
#     "2",
#     "--tmscore-threshold",
#     "0.5",
# ]
# subprocess.run(cmd)