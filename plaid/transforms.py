import torch
import einops


def mask_from_seq_lens(x: torch.Tensor, seqlen: torch.Tensor):
    mask = torch.arange(x.shape[1], device=x.device)
    mask = (
        einops.repeat(mask[None, :], "1 L -> N L", N=x.shape[0]) < seqlen[:, None]
    )
    return mask.long()