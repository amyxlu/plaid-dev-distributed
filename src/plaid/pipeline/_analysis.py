import glob
import uuid
import shutil
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..evaluation._structure_metrics import calculate_rmsd
from ..evaluation._tmalign import run_tmalign
from ..evaluation._perplexity import RITAPerplexity

from ..utils._misc import (
    extract_avg_b_factor_per_residue,
    parse_sequence_from_structure,
)
from ..utils._protein_properties import calculate_df_protein_property_mp


def _ensure_parent_exists(path):
    path = Path(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True)


def copy(src, dst, suffix=""):
    _ensure_parent_exists(dst)
    original_suffix = Path(dst).suffix
    dst = str(dst).replace(original_suffix, f"{suffix}{original_suffix}")
    shutil.copy2(src, dst)


def move(src, dst, suffix=""):
    _ensure_parent_exists(dst)
    original_suffix = Path(dst).suffix
    dst = str(dst).replace(original_suffix, f"{suffix}{original_suffix}")
    shutil.move(src, dst)


def move_designable(
    df,
    delete_original=False,
    original_dir_prefix: str = "generated/structures",
    target_dir_prefix="",
    add_hex_suffix=True
):
    """
    Given a dataframe with a column 'designable' that indicates whether a structure is designable or not,
    move the structures to a new directory based on the value of 'designable'.
    
    You can specify the relationship between the original subdirectory and the new subdirectory
    with respect to a root sample path. Default is to replace "generated/structures" with "".
    
    For example, "/data/samples/timestamp/generated/structures" -> "/data/samples/timestamp/designable"
    and "/data/samples/timestamp/generated/structures" -> "/data/samples/timestamp/undesignable"

    The latter option can be useful if you are moving designable proteins to several levels up
    in the directory (e.g. combining across many lengths).
    For example, you can set "100/generated/structures" and "" to loop over all lengths in a subdirectory.
    i.e. "/data/samples/by_length/100/generated/structures" -> "/data/samples/by_length/designable"

    To avoid filename clashes, after moving, a hex suffix is given to the file name.
    """
    generated_pdb_paths = df.pdb_paths

    if delete_original:
        for i, p in enumerate(generated_pdb_paths):
            suffix = f"_{uuid.uuid4().hex[:6]}" if add_hex_suffix else ""
            if df["designable"][i]:
                move(p, p.replace(original_dir_prefix, f"{target_dir_prefix}designable"), suffix)
            else:
                move(p, p.replace(original_dir_prefix, f"{target_dir_prefix}undesignable"), suffix)

    else:
        for i, p in enumerate(generated_pdb_paths):
            suffix = f"_{uuid.uuid4().hex[:6]}" if add_hex_suffix else ""
            if df["designable"][i]:
                copy(p, p.replace(original_dir_prefix, f"{target_dir_prefix}designable"), suffix)
            else:
                copy(p, p.replace(original_dir_prefix, f"{target_dir_prefix}undesignable"), suffix)


def run_analysis(sample_dir, rita_perplexity: RITAPerplexity = None):
    import warnings

    warnings.filterwarnings("ignore")

    # Gather paths; sort should guarantee that samples are in the same order
    generated_pdb_paths = glob.glob(str(sample_dir / "generated/structures/*pdb"))
    inverse_generated_pdb_paths = glob.glob(
        str(sample_dir / "inverse_generated/structures/*pdb")
    )

    generated_pdb_paths.sort()
    inverse_generated_pdb_paths.sort()
    assert (
        len(generated_pdb_paths) == len(inverse_generated_pdb_paths)
    )

    # maybe run self-consistency (if phantom generated structures were generated)
    phantom_generated_pdb_paths = glob.glob(
        str(sample_dir / "phantom_generated/structures/*pdb")
    )
    run_self_consistency = len(phantom_generated_pdb_paths) > 0

    if run_self_consistency:
        phantom_generated_pdb_paths.sort()
        assert (
            len(generated_pdb_paths) == len(phantom_generated_pdb_paths)
        )

    # Initialize dataframe
    d = {
        "pdb_paths": [],
        "sequences": [],
        "inverse_generated_pdb_paths": [],
    }

    if run_self_consistency:
        d["phantom_generated_pdb_paths"] = []

    # parse sequence directly from structure to make sure there are no mismatches
    print("Parsing sequences from structures")
    for i, p in enumerate(generated_pdb_paths):
        d["pdb_paths"].append(p)
        d["inverse_generated_pdb_paths"].append(inverse_generated_pdb_paths[i])

        if run_self_consistency:
            d["phantom_generated_pdb_paths"].append(phantom_generated_pdb_paths[i])

        with open(p, "r") as f:
            pdbstr = f.read()

        sequence = parse_sequence_from_structure(pdbstr)
        d["sequences"].append(sequence)

    # apply metrics to dataframe:
    try:
        df = pd.DataFrame(d)
        df.head()

        print("Calculating average pLDDT")
        df["plddt"] = df.apply(
            lambda row: np.mean(extract_avg_b_factor_per_residue(row["pdb_paths"])),
            axis=1,
        )

        print("Calculating ccRMSD")
        df["ccrmsd"] = df.apply(
            lambda row: calculate_rmsd(
                row["pdb_paths"], row["inverse_generated_pdb_paths"]
            ),
            axis=1,
        )

        print("Calculating ccTM")
        df["ccTM"] = df.apply(
            lambda row: run_tmalign(
                row["pdb_paths"], row["inverse_generated_pdb_paths"]
            ),
            axis=1,
        )

        df["designable"] = df.ccrmsd < 2

        # run self-consistency metrics, if applicable:
        if run_self_consistency:
            print("Calculating scRMSD")
            df["scrmsd"] = df.apply(
                lambda row: calculate_rmsd(
                    row["pdb_paths"], row["phantom_generated_pdb_paths"]
                ),
                axis=1,
            )

            print("Calculating sctm")
            df["sctm"] = df.apply(
                lambda row: run_tmalign(
                    row["pdb_paths"], row["phantom_generated_pdb_paths"]
                ),
                axis=1,
            )

        # calculate perplexity:
        print("Calculating perplexity under RITA")
        if rita_perplexity is None:
            rita_perplexity = RITAPerplexity()

        df["perplexity"] = df.apply(
            lambda row: rita_perplexity.calc_perplexity(row["sequences"]), axis=1
        )

        # calculate protein properties
        print("Calculating sequence properties")
        df = calculate_df_protein_property_mp(df)

    # if any any time we get an error, save the df as it is in that state
    except Exception as e:
        print(e)
        print(f"Will save the dataframe as it is to {sample_dir / 'designability.csv'}")
        pass

    df.to_csv(sample_dir / "designability.csv")

    return df