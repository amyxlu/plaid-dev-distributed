""" Calculate additional protein attributes from sequence for conditioning.
"""
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np
import typing as T


# Calculated using a random subset of 40399 UniRef90 sequences,
# subcropped to 128 amino acids, and uses the bin edges and fitted using
# sklearn.preprocessing.KBinsDiscretizer with n_bins=6, encode='ordinal', strategy='quantile'
# (see training/uniref/plot_uniref_chunks_len128.ipynb)

SS_BOUNDARIES = {
    "uniref": {
        "helix": [0.1647, 0.3294, 0.4941, 0.6589, 0.8236],
        "turn": [0.1628, 0.3255, 0.4883, 0.6510, 0.8138],
        "sheet": [0.1524, 0.3048, 0.4571, 0.6095, 0.7619],
    },
    "cath": {
        "helix": [0.2109375, 0.25, 0.2734375, 0.296875, 0.3203125],
        "turn": [0.171875, 0.2109375, 0.234375, 0.265625, 0.3046875],
        "sheet": [0.1875, 0.2265625, 0.2578125, 0.296875, 0.3359375],
    },
}


def _quantize_frac(arr: np.ndarray, boundaries: np.ndarray):
    """
    0: < arr[0]
    1: arr[0] to arr[1]
    2: ...
    3: ...
    4: ...
    5: > arr[-1]
    """
    return np.digitize(arr, boundaries)


def _quantize_secondary_structure_fractions(fracs, origin_dataset: str = "uniref"):
    helix_fracs, turn_fracs, sheet_fracs = fracs[:, 0], fracs[:, 1], fracs[:, 2]
    helix_quantized = _quantize_frac(
        helix_fracs, SS_BOUNDARIES[origin_dataset]["helix"]
    )
    turn_quantized = _quantize_frac(turn_fracs, SS_BOUNDARIES[origin_dataset]["turn"])
    sheet_quantized = _quantize_frac(
        sheet_fracs, SS_BOUNDARIES[origin_dataset]["sheet"]
    )
    return np.stack([helix_quantized, turn_quantized, sheet_quantized], axis=1)


def _sequence_to_secondary_structure_frac(
    sequence: str,
) -> T.Tuple[float, float, float]:
    sequence = str(sequence).replace(">", "")
    sequence = sequence.replace("<", "")
    X = ProteinAnalysis(str(sequence))
    return X.secondary_structure_fraction()


def sequences_to_secondary_structure_fracs(
    sequences: T.List[str], quantized: bool = True, origin_dataset: str = "uniref"
) -> T.List[T.Tuple[float, float, float]]:
    fracs = [_sequence_to_secondary_structure_frac(s) for s in sequences]
    fracs = np.stack(fracs, axis=0)
    if quantized:
        return _quantize_secondary_structure_fractions(fracs, origin_dataset)
    else:
        return np.array(fracs)


if __name__ == "__main__":
    import IPython

    IPython.embed()