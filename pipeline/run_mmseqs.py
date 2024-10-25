import argparse

from plaid.constants import PDB_DATABASE_PATH
from plaid.evaluation import mmseqs_easysearch, mmseqs_easycluster

parser = argparse.ArgumentParser()
parser.add_argument('--sample_dir', default="/data/lux70/plaid/artifacts/samples/5j007z42/val_dist/f989_o1326_l144_s3")
parser.add_argument("--fasta_file_name", default="generated/sequences.fasta")
args = parser.parse_args()

# TODO: run this only on designable structures
# "generated/structures/designable"
mmseqs_easycluster(args.sample_dir,  args.fasta_file_name)
mmseqs_easysearch(args.sample_dir, args.fasta_file_name) 