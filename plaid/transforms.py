import torch
import random
import einops


def mask_from_seq_lens(x: torch.Tensor, seqlen: torch.Tensor):
    mask = torch.arange(x.shape[1], device=x.device)
    mask = einops.repeat(mask[None, :], "1 L -> N L", N=x.shape[0]) < seqlen[:, None]
    return mask.long()


def _get_random_sequence_crop(s, length):
    if len(s) > length:
        start = random.randint(0, len(s) - length)
        return s[start : start + length]
    else:
        return s


def get_random_sequence_crop_batch(sequence_batch, max_len, min_len=None):
    if not min_len is None:
        sequence_batch = list(filter(lambda s: len(s) >= min_len, sequence_batch))
    return [_get_random_sequence_crop(seq, max_len) for seq in sequence_batch]

