import os
import re
import typing as T
from pathlib import Path
from tqdm import trange

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

import torch
import typing as T
import numpy as np
import re
from openfold.np import residue_constants

from .utils._misc import npy, to_tensor, outputs_to_avg_metric
from .decoder import FullyConnectedNetwork
from .esmfold import ESMFOLD_Z_DIM, esmfold_v1
from .esmfold.misc import output_to_pdb, batch_encode_sequences
from .transforms import trim_or_pad_batch_first


ArrayLike = T.Union[np.ndarray, torch.Tensor, T.List]
PathLike = T.Union[str, Path]


CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"

# https://github.com/dauparas/ProteinMPNN/blob/main/protein_mpnn_utils.py#L61
PROTEINMPNN_AACHAR_TO_AAIDX_ARR = list("ARNDCQEGHILKMFPSTWYV-")
PROTEINMPNN_AAIDX_TO_AACHAR = {
    idx: char for idx, char in enumerate(PROTEINMPNN_AACHAR_TO_AAIDX_ARR)
}
PROTEINMPNN_AACHAR_TO_AAIDX = {
    char: idx for idx, char in enumerate(PROTEINMPNN_AACHAR_TO_AAIDX_ARR)
}


class DecoderTokenizer:
    def __init__(self, vocab="openfold"):
        if vocab == "openfold":
            self.aachar_to_aaidx = residue_constants.restype_order_with_x
            self.aaidx_to_aachar = {v: k for k, v in self.aachar_to_aaidx.items()}
        elif vocab == "proteinmpnn":
            self.aachar_to_aaidx = PROTEINMPNN_AACHAR_TO_AAIDX
            self.aaidx_to_aachar = PROTEINMPNN_AAIDX_TO_AACHAR
        else:
            raise ValueError(f"Unknown vocab {vocab}")

        self.vocab_size = len(self.aaidx_to_aachar)
        self.unk_idx = self.aachar_to_aaidx.get("X", None)
        self.pad_idx = self.aachar_to_aaidx.get("_", None)
        self.eos_idx = self.aachar_to_aaidx.get(">", None)
        self.bos_idx = self.aachar_to_aaidx.get("<", None)

    def __len__(self):
        return len(self.aaidx_to_aachar)

    def _char2idx(self, char: str) -> int:
        return self.aachar_to_aaidx.get(char, self.unk_idx)

    def str_to_aatype_sequence(
        self, seq: T.Union[T.Iterable, str], as_torch_tensor: bool = True
    ):
        if isinstance(seq, str):
            seq = list(seq)

        aatype = [self._char2idx(aa) for aa in seq]
        if as_torch_tensor:
            return torch.tensor(aatype)
        else:
            return aatype

    def aatype_to_str_sequence(self, aatype: T.List[int], strip_mode: str = "none"):
        assert strip_mode in ["none", "strip_pad", "strip_eos", "strip_after_eos"]
        aastr = "".join([self.aaidx_to_aachar[aa] for aa in npy(aatype)])
        if strip_mode == "none":
            return aastr
        elif strip_mode == "strip_pad":
            aastr = aastr.replace("_", "")
        elif strip_mode == "strip_eos":
            # strip ">" and everything after it
            pattern = r"^(.*?)[>]"
        elif strip_mode == "strip_after_eos":
            # keep ">" but strip everything after it
            pattern = r"^(.*?[>])"
            match = re.search(pattern, aastr)
            if match:
                aastr = match.group(1)
        else:
            raise ValueError(f"Unrecognized strip_mode: {strip_mode}")
        return aastr

    def collate_dense_tensors(
        self, samples: T.List[torch.Tensor], pad_v: int
    ) -> torch.Tensor:
        """
        Takes a list of tensors with the following dimensions:
            [(d_11,       ...,           d_1K),
            (d_21,       ...,           d_2K),
            ...,
            (d_N1,       ...,           d_NK)]
        and stack + pads them into a single tensor of:
        (N, max_i=1,N { d_i1 }, ..., max_i=1,N {diK})
        """

        if len(samples) == 0:
            return torch.Tensor()
        if len(set(x.dim() for x in samples)) != 1:
            raise RuntimeError(
                f"Samples has varying dimensions: {[x.dim() for x in samples]}"
            )
        (device,) = tuple(set(x.device for x in samples))  # assumes all on same device
        max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
        result = torch.empty(
            len(samples), *max_shape, dtype=samples[0].dtype, device=device
        )
        result.fill_(pad_v)
        for i in range(len(samples)):
            result_i = result[i]
            t = samples[i]
            result_i[tuple(slice(0, k) for k in t.shape)] = t
        return result

    def batch_encode_sequences(
        self, sequences: T.Sequence[str], pad_v: T.Optional[int] = None
    ) -> T.Tuple[torch.Tensor, torch.Tensor]:
        """Encode sequences using this tokenization scheme, mostly to generate labels during training
        of the autoregressive decoder.

        Args:
            sequences (T.Sequence[str]): List of amino acid sequence strings.
            add_eos_bos (bool): Whether or not to also add the local EOS/BOS token in generating the labels.

        Returns:
            T.Tuple[torch.Tensor, torch.Tensor]: Amino acid indices and mask (0 if padded, 1 otherwise).
        """
        if pad_v is None:
            pad_v = self.pad_idx

        aatype_list = []

        for seq in sequences:
            aatype_seq = self.str_to_aatype_sequence(seq)
            aatype_list.append(aatype_seq)

        aatype = self.collate_dense_tensors(aatype_list, pad_v=pad_v)
        mask = self.collate_dense_tensors(
            [aatype.new_ones(len(aatype_seq)) for aatype_seq in aatype_list],
            pad_v=pad_v,
        )
        return aatype, mask

    def _is_valid_aa(self, aa):
        return aa in CANONICAL_AA

    def remove_invalid_aa(self, string: str):
        return "".join([s for s in string if self._is_valid_aa(s)])



class LatentToSequence:
    def __init__(self, temperature: float = 1.0):
        """On construction, all models are on the CPU."""
        self.temperature = temperature
        self.tokenizer = DecoderTokenizer()
        self.decoder = FullyConnectedNetwork.from_pretrained(device="cpu")
        self.device = torch.device("cpu")

        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False

    def to(self, device):
        """Move onto the device for the usecase before calling to_sequence()."""
        self.decoder = self.decoder.to(device)
        self.device = device
        return self

    def to_sequence(
        self, latent: ArrayLike, mask=None, return_logits=False, drop_mask_idx=True
    ):
        if not mask is None:
            mask = torch.ones_like(latent)
        latent = to_tensor(latent, device=self.device)
        assert (
            latent.device == self.decoder.device
        ), "Make sure to call .to(device) to move decoder to the correct device."

        with torch.no_grad():
            output_logits = self.decoder(latent)

        # adjust by temperature
        output_logits /= self.temperature

        # remove UNK token
        if drop_mask_idx:
            _mask = (
                torch.arange(output_logits.shape[-1], device=self.device)
                != self.tokenizer.unk_idx
            )
            drop_mask_logits = torch.index_select(
                input=output_logits,
                dim=-1,
                index=torch.arange(output_logits.shape[-1], device=self.device)[_mask],
            )
            argmax_idx = drop_mask_logits.argmax(-1)
            dist = torch.distributions.OneHotCategorical(logits=drop_mask_logits)
            sequence_probs = F.softmax(drop_mask_logits, dim=-1)
        else:
            # get the argmax index & compare it to the actual sample, to get a sense as to how temperature affects diversity
            argmax_idx = output_logits.argmax(-1)
            dist = torch.distributions.OneHotCategorical(logits=output_logits)
            sequence_probs = F.softmax(output_logits, dim=-1)
        
        sequence_idx = dist.sample().argmax(-1)
        sequence_probs = torch.gather(
            sequence_probs, dim=-1, index=argmax_idx.unsqueeze(-1)
        ).squeeze(-1)
        stochasticity = (argmax_idx == sequence_idx).sum() / torch.numel(argmax_idx)
        # print(f"percentage similarty to argmax idx: {stochasticity:.3f}")

        sequence_str = [
            self.tokenizer.aatype_to_str_sequence(s)
            for s in sequence_idx.long().cpu().numpy()
        ]

        if return_logits:
            # return the original output logits, e.g. for loss & backprop purposes
            return output_logits, sequence_idx, sequence_str
        else:
            return sequence_probs, sequence_idx, sequence_str


class LatentToStructure:
    def __init__(self, esmfold=None, chunk_size=64):
        self.device = torch.device("cpu")
        if esmfold is None:
            import time
            print("loading esmfold model...")
            start = time.time()
            esmfold = esmfold_v1() 
            end = time.time()
            print(f"ESMFold model created in {end-start:.2f} seconds.")
        
        self.esmfold = esmfold
        # self.esmfold.to("cpu")
        self.esmfold.set_chunk_size(chunk_size)
        del self.esmfold.esm  # save some GPU space
        assert not self.esmfold.trunk is None

        self.esmfold.eval()
        for param in self.esmfold.parameters():
            param.requires_grad = False

    def to(self, device):
        self.esmfold = self.esmfold.to(device)
        self.device = device
        return self

    @torch.no_grad()
    def run_batch(
        self, s_, aa_, mask_, residx_, num_recycles, return_metrics=False, *args, **kwargs
    ):
        # https://github.com/facebookresearch/esm/blob/main/esm/esmfold/v1/esmfold.py#L208
        # utils.print_cuda_memory_usage()
        _, L, _ = s_.shape
        z_ = s_.new_zeros(s_.shape[0], L, L, ESMFOLD_Z_DIM).to(self.device)

        def maybe_pad(tensor, length):
            if tensor.shape[1] != length:
                return trim_or_pad_batch_first(tensor, length, pad_idx=0)
            else:
                return tensor
            
        mask_ = maybe_pad(mask_, L)
        aa_ = maybe_pad(aa_, L)
        residx_ = maybe_pad(residx_, L)

        with torch.no_grad():
            output = self.esmfold.folding_trunk(
                s_s_0=s_,
                s_z_0=z_,
                aa=aa_,
                residx=residx_,
                mask=mask_,
                num_recycles=num_recycles,
            )
        pdb_str = output_to_pdb(output)
        if return_metrics:
            metric = outputs_to_avg_metric(output)
            return pdb_str, output, metric
        else:
            return pdb_str, output

    def to_structure(
        self,
        latent: ArrayLike,
        sequences: T.List[str],
        num_recycles: int = 4,
        batch_size: T.Optional[int] = None,
        return_metrics: bool = False, 
        verbose: bool = False,
        *args,
        **kwargs
    ) -> T.Tuple[T.List[PathLike], T.Union[T.Dict, pd.DataFrame]]:
        """set up devices and tensors"""
        aatype, mask, residx, _, _ = batch_encode_sequences(sequences)
        aatype, mask, residx = tuple(
            map(lambda x: x.to(self.device), (aatype, mask, residx))
        )
        latent = to_tensor(latent, device=self.device)
        assert (
            latent.device == self.esmfold.device
        ), "Make sure to call .to(device) to move trunk to the correct device."

        if batch_size is None:
            if verbose:
                print("Generating structure from latents")
            return self.run_batch(
                latent, aatype, mask, residx, num_recycles, return_metrics 
            )

        else:
            metric_dfs = []
            output_dicts_list = []
            all_pdb_strs = []
            for start in trange(
                0, len(latent), batch_size, desc="(Generating structure)"
            ):
                
                # Process current batch
                s_, aa_, mask_, residx_ = tuple(
                    map(
                        lambda x: x[start : start + batch_size],
                        (latent, aatype, mask, residx),
                    )
                )
                
                # Collect outputs
                outputs = self.run_batch(s_, aa_, mask_, residx_, num_recycles, return_metrics)
                if return_metrics:
                   pdb_str, output_dict, metric_df = outputs 
                   metric_dfs.append(metric_df)
                else:
                   pdb_str, output_dict = outputs 
                all_pdb_strs.extend(pdb_str)
                output_dicts_list.append(output_dict)

            # combine results at the end of the batches
            # outputs = 
            # if return_metrics: 
            #     results = {k: v for D in results for k, v in D.items()}
            # else:
            #     results = [pd.DataFrame.from_dict(d) for d in results]
            #     results = pd.concat(results)
            # return all_pdb_strs, results
            if return_metrics:
                return all_pdb_strs, output_dicts_list, metric_dfs
            else:
                return all_pdb_strs, output_dicts_list


if __name__ == "__main__":
    import torch
    from plaid.proteins import LatentToSequence, LatentToStructure

    device = torch.device("cuda")
    sequence_constructor = LatentToSequence().to(device)
    structure_constructor = LatentToStructure().to(device)
    latent = torch.randn(16, 128, 1024).to(device)
    _, _, strs = sequence_constructor.to_sequence(latent)
    output = structure_constructor.to_structure(latent, strs)
