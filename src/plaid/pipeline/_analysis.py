import glob

import numpy as np
import pandas as pd

from ..evaluation._structure_metrics import calculate_rmsd
from ..evaluation._tmalign import run_tmalign
from ..evaluation._perplexity import RITAPerplexity

from ..utils._misc import extract_avg_b_factor_per_residue, parse_sequence_from_structure
from ..utils._protein_properties import calculate_df_protein_property_mp



def run_analysis(sample_dir, rita_perplexity: RITAPerplexity = None):
    import warnings
    warnings.filterwarnings("ignore")

    # Gather paths; sort should guarantee that samples are in the same order
    generated_pdb_paths = glob.glob(str(sample_dir / "generated/structures/*pdb"))
    inverse_generated_pdb_paths = glob.glob(str(sample_dir / "inverse_generated/structures/*pdb"))
    phantom_generated_pdb_paths = glob.glob(str(sample_dir / "phantom_generated/structures/*pdb"))

    generated_pdb_paths.sort()
    inverse_generated_pdb_paths.sort()
    phantom_generated_pdb_paths.sort()

    assert len(generated_pdb_paths) == len(inverse_generated_pdb_paths) == len(phantom_generated_pdb_paths)

    # Initialize dataframe
    d = {
        "pdb_paths": [],
        "sequences": [],
        "inverse_generated_pdb_paths": [],
        "phantom_generated_pdb_paths": [],
    }

    # parse sequence directly from structure to make sure there are no mismatches
    print("Parsing sequences from structures")
    for i, p in enumerate(generated_pdb_paths):
        d['pdb_paths'].append(p)
        d['inverse_generated_pdb_paths'].append(inverse_generated_pdb_paths[i])
        d['phantom_generated_pdb_paths'].append(inverse_generated_pdb_paths[i])

        with open(p, "r") as f:
            pdbstr = f.read()
            
        sequence = parse_sequence_from_structure(pdbstr)
        d['sequences'].append(sequence)

    # apply metrics to dataframe:

    try:
        df = pd.DataFrame(d)
        df.head()

        print("Calculating average pLDDT")
        df['plddt'] = df.apply(lambda row: np.mean(extract_avg_b_factor_per_residue(row['pdb_paths'])), axis=1)

        print("Calculating ccRMSD")
        df['ccrmsd'] = df.apply(lambda row: calculate_rmsd(row['pdb_paths'], row['inverse_generated_pdb_paths']), axis=1)

        print("Calculating scRMSD")
        df['scrmsd'] = df.apply(lambda row: calculate_rmsd(row['pdb_paths'], row['phantom_generated_pdb_paths']), axis=1)

        print("Calculating cctm")
        df['cctm'] = df.apply(lambda row: run_tmalign(row['pdb_paths'], row['inverse_generated_pdb_paths']), axis=1)

        print("Calculating sctm")
        df['sctm'] = df.apply(lambda row: run_tmalign(row['pdb_paths'], row['phantom_generated_pdb_paths']), axis=1)

        df['designable'] = df.ccrmsd < 2

        print("Calculating perplexity under RITA")
        if rita_perplexity is None:
            rita_perplexity = RITAPerplexity()
        
        df['perplexity'] = df.apply(lambda row: rita_perplexity.calc_perplexity(row['sequences']), axis=1)

        print("Calculating sequence properties")
        df = calculate_df_protein_property_mp(df)

    # if any any time we get an error, save the df as it is in that state 
    except Exception as e:
        print(e)
        print(f"Will save the dataframe as it is to {sample_dir / 'designability.csv'}")
        pass

    df.to_csv(sample_dir / "designability.csv")

    return df

