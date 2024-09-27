from plaid.utils import parse_sequence_from_structure, write_to_fasta
from pathlib import Path
import glob
import warnings
from tqdm.auto import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pdbdir", type=str, required=True)
args = parser.parse_args()

# pdbdir = Path("/data/lux70/data/pfam/val_stats/generated/structures")
# pdbdir = Path("/data/lux70/plaid/baselines/protpardelle/samples_large/samples/generated")
pdbdir = Path(args.pdbdir)
pdb_paths = glob.glob(str(pdbdir / "*pdb"))
pdb_paths.sort()

warnings.filterwarnings('ignore') 

seq_dict = {}

for p in tqdm(pdb_paths):
    with open(p, "r") as f:
        pdbstr = f.read()
    sequence = parse_sequence_from_structure(pdbstr)
    seq_dict[p] = sequence

outfasta = pdbdir / "../sequences.fasta"

headers, sequences = [], []
cur_i = 0

for k, v in seq_dict.items():
    # headers.append(k)
    headers.append(f"sample{cur_i}")
    sequences.append(v)
    cur_i += 1

write_to_fasta(
    sequences,
    outfasta,
    headers
)