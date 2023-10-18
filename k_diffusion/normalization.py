import numpy as np
import typing as T
import torch
import os
from . import utils


ArrayLike = T.Union[np.ndarray, T.List[float], torch.Tensor]


GLOBAL_SEQEMB_STATS = {
    "uniref": {
        "max": 3038.4783,
        "min": -920.1115,
        "mean": 1.2394488,
        "std": 70.907074,
    },
    "cath": {
        "max": 2853.481,
        "min": -878.217,
        "mean": 1.289,
        "std": 71.788,
    },
}

CHANNELWISE_NPY_PATHS = {
    "uniref": {
        "max": f"{os.environ['KD_PROJECT_HOME']}/cached_tensors/subset_100000/channelwise_max.npy",
        "min": f"{os.environ['KD_PROJECT_HOME']}/cached_tensors/subset_100000/channelwise_min.npy",
        "mean": f"{os.environ['KD_PROJECT_HOME']}/cached_tensors/subset_100000/channelwise_mean.npy",
        "std": f"{os.environ['KD_PROJECT_HOME']}/cached_tensors/subset_100000/channelwise_std.npy",
    }
}


def _array_conversion(
    x: T.Union[float, ArrayLike],
    minv: T.Union[float, ArrayLike],
    maxv: T.Union[float, ArrayLike],
) -> ArrayLike:
    assert type(minv) == type(maxv)

    if isinstance(minv, float):
        return x, minv, maxv

    elif isinstance(x, np.ndarray):
        minv = utils.npy(minv)
        maxv = utils.npy(maxv)
        return x, minv, maxv

    elif isinstance(x, torch.Tensor):
        if isinstance(minv, torch.Tensor):
            minv = minv.to(x.device)
            maxv = maxv.to(x.device)
            return x, minv, maxv
        elif isinstance(minv, np.ndarray):
            # Usually this is the case during training
            minv = torch.from_numpy(minv).to(x.device)
            maxv = torch.from_numpy(maxv).to(x.device)
            return x, minv, maxv

    else:
        raise TypeError("Invalid input type.")


def _minmax_scaling(
    x: ArrayLike,
    data_minv: T.Union[float, ArrayLike],
    data_maxv: T.Union[float, ArrayLike],
    scaled_minv: float = -1.0,
    scaled_maxv: float = 1.0,
) -> ArrayLike:
    """
    Scales all values to between a max and min value, either globally or channel-wise.
    If global, data_minv and data_maxv should be floats. Otherwise,
    they should be ArrayLike denoting the max and min for each channel with shape (1024,)

    The default scaling range is between -1 and 1, following DDPM.
    Follows the sklearn API:
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    """
    x, data_minv, data_maxv = _array_conversion(x, data_minv, data_maxv)
    X_std = (x - data_minv) / (data_maxv - data_minv)
    X_scaled = X_std * (scaled_maxv - scaled_minv) + scaled_minv
    return X_scaled


def _undo_minmax_scaling(
    x_scaled: ArrayLike,
    data_minv: T.Union[float, ArrayLike],
    data_maxv: T.Union[float, ArrayLike],
    scaled_minv: float = -1.0,
    scaled_maxv: float = 1.0,
):
    x_scaled, data_minv, data_maxv = _array_conversion(x_scaled, data_minv, data_maxv)
    x_std = (x_scaled - scaled_minv) / (scaled_maxv - scaled_minv)
    x = x_std * (data_maxv - data_minv) + data_minv
    return x


def _standardize(
    x: ArrayLike, meanv: T.Union[float, ArrayLike], stdv: T.Union[float, ArrayLike]
) -> ArrayLike:
    """
    Standardize to center on zero mean with unit std, either globally or channel-wise.
    If global, data_minv and data_maxv should be floats. Otherwise,
    they should be ArrayLike denoting the max and min for each channel with shape (1024,)
    """
    x, meanv, stdv = _array_conversion(x, meanv, stdv)
    return (x - meanv) / stdv


def _undo_standardize(
    x_scaled: ArrayLike,
    meanv: T.Union[float, ArrayLike],
    stdv: T.Union[float, ArrayLike],
):
    x_scaled, meanv, stdv = _array_conversion(x_scaled, meanv, stdv)
    return x_scaled * stdv + meanv


def _scaled_l2_norm(x, scale_factor=1.0):
    """
    Scale to L2 unit norm along channel for each sample, following the
    treatment of CLIP embeddings in DALLE-2.

    Optionally scale up by sqrt(embed_dim), following suggestion from
    https://github.com/lucidrains/DALLE2-pytorch/issues/60
    """
    x = x / x.norm(dim=-1, p="fro", keepdim=True)
    x *= scale_factor
    return x


def _check_valid_mode(mode: str):
    return mode in [
        "global_minmaxnorm",
        "global_standardize",
        "channel_minmaxnorm",
        "channel_standardize",
    ]


def _check_valid_origin_dataset(origin_dataset: str):
    return origin_dataset in ["uniref", "cath"]


def load_channelwise_stats(origin_dataset: str = "uniref"):
    NPY_PATHS = CHANNELWISE_NPY_PATHS[origin_dataset]
    return {
        "max": np.load(NPY_PATHS["max"]),
        "min": np.load(NPY_PATHS["min"]),
        "mean": np.load(NPY_PATHS["mean"]),
        "std": np.load(NPY_PATHS["std"]),
    }


def scale_embedding(
    x: ArrayLike, mode: str = "channel_standardize", origin_dataset: str = "uniref"
):
    if (mode is None) or (mode == "none"):
        return x

    assert _check_valid_mode(mode), f"Invalid mode {mode}."
    assert _check_valid_origin_dataset(origin_dataset)

    if "channel_" in mode:
        stat_dict = load_channelwise_stats(origin_dataset)
    else:
        stat_dict = GLOBAL_SEQEMB_STATS[origin_dataset]
    maxv, minv, meanv, stdv = (
        stat_dict["max"],
        stat_dict["min"],
        stat_dict["mean"],
        stat_dict["std"],
    )

    with torch.no_grad():
        if mode == "global_minmaxnorm":
            x_scaled = _minmax_scaling(x, minv, maxv)
        elif mode == "global_standardize":
            x_scaled = _standardize(x, meanv, stdv)
        elif mode == "channel_minmaxnorm":
            x_scaled = _minmax_scaling(x, minv, maxv)
        elif mode == "channel_standardize":
            x_scaled = _standardize(x, meanv, stdv)
        else:
            raise NotImplementedError

    return x_scaled


def undo_scale_embedding(
    x_scaled: ArrayLike,
    mode: str = "channel_standardize",
    origin_dataset: str = "uniref",
):
    if (mode is None) or (mode == "none"):
        return x_scaled

    assert _check_valid_mode(mode), f"Invalid mode {mode}."
    assert _check_valid_origin_dataset(origin_dataset)

    if "channel_" in mode:
        stat_dict = load_channelwise_stats(origin_dataset)
    else:
        stat_dict = GLOBAL_SEQEMB_STATS[origin_dataset]
    maxv, minv, meanv, stdv = (
        stat_dict["max"],
        stat_dict["min"],
        stat_dict["mean"],
        stat_dict["std"],
    )

    with torch.no_grad():
        if mode == "global_minmaxnorm":
            x_scaled = _undo_minmax_scaling(x_scaled, minv, maxv)
        elif mode == "global_standardize":
            x_scaled = _undo_standardize(x_scaled, meanv, stdv)
        elif mode == "channel_minmaxnorm":
            x_scaled = _undo_minmax_scaling(x_scaled, minv, maxv)
        elif mode == "channel_standardize":
            x_scaled = _undo_standardize(x_scaled, meanv, stdv)
        else:
            raise NotImplementedError
    return x_scaled


def _clamp(tensor: ArrayLike, min_values: ArrayLike, max_values: ArrayLike):
    """
    Clamp values to min/max values defined by an array.
    """
    tensor, min_values, max_values = _array_conversion(tensor, min_values, max_values)
    return torch.where(
        tensor < min_values,
        min_values,
        torch.where(tensor > max_values, max_values, tensor),
    )


def clamp_embedding(
    x: ArrayLike, mode: str = "channel_standardize", origin_dataset: str = "uniref"
):
    if (mode is None) or (mode == "none"):
        return x

    assert _check_valid_origin_dataset(origin_dataset)
    if "channel_" in mode:
        stat_dict = load_channelwise_stats(origin_dataset)
    else:
        raise NotImplementedError("Use native functions instead.")
    
    maxv, minv, meanv, stdv = (
        stat_dict["max"],
        stat_dict["min"],
        stat_dict["mean"],
        stat_dict["std"],
    )

    with torch.no_grad():
        x = _clamp(x, minv, maxv)
    return x 
