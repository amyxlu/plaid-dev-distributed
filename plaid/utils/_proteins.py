import os
import re
import typing as T
from pathlib import Path
from tqdm import trange

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

from esm.esmfold.v1.misc import output_to_pdb, batch_encode_sequences

import torch
import typing as T
import numpy as np
import re
from openfold.np.residue_constants import restype_order_with_x

from ._misc import npy, to_tensor
from ..layers import FullyConnectedNetwork
from ..esmfold import ESMFold, ESMFOLD_Z_DIM


ArrayLike = T.Union[np.ndarray, torch.Tensor, T.List]
PathLike = T.Union[str, Path]


restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}

restype_3to1 = {v: k for k, v in restype_1to3.items()}


CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"
OPENFOLD_AAIDX_TO_AACHAR = {idx: char for idx, char in enumerate(restype_order_with_x)}
OPENFOLD_AACHAR_TO_AAINDEX = {char: idx for idx, char in enumerate(restype_order_with_x)}

# https://github.com/dauparas/ProteinMPNN/blob/main/protein_mpnn_utils.py#L61
PROTEINMPNN_AACHAR_TO_AAIDX_ARR = list("ARNDCQEGHILKMFPSTWYV-")
PROTEINMPNN_AAIDX_TO_AACHAR = {idx: char for idx, char in enumerate(PROTEINMPNN_AACHAR_TO_AAIDX_ARR)}
PROTEINMPNN_AACHAR_TO_AAIDX = {char: idx for idx, char in enumerate(PROTEINMPNN_AACHAR_TO_AAIDX_ARR)}

cache_dir = Path(os.path.dirname(__file__)) / "../cached_tensors/subset_5000_oct24"
DECODER_CKPT_PATH = Path(os.path.dirname(__file__)) / "../cached_tensors/decoder_vocab_21.ckpt"


def load_sequence_decoder(ckpt_path=None, device=None, eval_mode=True):
    if ckpt_path is None:
        ckpt_path = DECODER_CKPT_PATH
    tokenizer = DecoderTokenizer()
    n_classes = len(tokenizer)
    decoder = FullyConnectedNetwork(mlp_num_layers=2, n_classes=n_classes)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    decoder.load_state_dict(checkpoint["model_state_dict"])
    if eval_mode:
        for param in decoder.parameters():
            param.requires_grad = False
        decoder.eval()
    if not device is None:
        decoder.to(device)
    return decoder


class DecoderTokenizer:
    def __init__(self, vocab="vocab_21"):
        if vocab == "vocab_21":
            self.aachar_to_aaidx = OPENFOLD_AACHAR_TO_AAINDEX
            self.aaidx_to_aachar = OPENFOLD_AAIDX_TO_AACHAR
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
            [aatype.new_ones(len(aatype_seq)) for aatype_seq in aatype_list], pad_v=pad_v
        )
        return aatype, mask

    def _is_valid_aa(self, aa):
        return aa in CANONICAL_AA

    def remove_invalid_aa(self, string: str):
        return "".join([s for s in string if self._is_valid_aa(s)])


def outputs_to_avg_metric(outputs):
    avg_metrics = {}
    metrics_to_log = [
        "plddt",
        "ptm",
        "aligned_confidence_probs",
        "predicted_aligned_error",
    ]

    for metric in metrics_to_log:
        value = npy(outputs[metric])

        if value.ndim == 1:
            mean = value
        elif value.ndim == 2:
            mean = np.mean(value, axis=1)
        else:
            assert value.ndim > 2
            mean = np.mean(value, axis=tuple(range(1, value.ndim)))

        avg_metrics[metric] = mean

    return avg_metrics


class LatentToSequence:
    def __init__(self, device, temperature: float = 1.0):
        self.temperature = temperature
        self.device = device
        self.tokenizer = DecoderTokenizer("vocab_21")
        self.decoder = load_sequence_decoder(device=device).eval().requires_grad_(False)

    def to_sequence(self, latent: ArrayLike):
        latent = to_tensor(latent, device=self.device)
        with torch.no_grad():
            sequence_logits = self.decoder(latent)

        # remove UNK token
        _mask = torch.arange(sequence_logits.shape[-1], device=self.device) != self.tokenizer.unk_idx
        sequence_logits = torch.index_select(
            input=sequence_logits,
            dim=-1,
            index=torch.arange(sequence_logits.shape[-1], device=self.device)[_mask]
        )

        # adjust by temperature
        sequence_logits /= self.temperature

        # get the argmax index & compare it to the actual sample, to get a sense as to how temperature affects diversity
        argmax_idx = sequence_logits.argmax(-1)
        sequence_idx = argmax_idx
        dist = torch.distributions.OneHotCategorical(logits=sequence_logits)
        sequence_idx = dist.sample().argmax(-1)
        stochasticity = (argmax_idx == sequence_idx).sum() / (argmax_idx.shape[0] * argmax_idx.shape[1])
        print(f"percentage similarty to argmax idx: {stochasticity:.3f}")
            
        sequence_str = [
            self.tokenizer.aatype_to_str_sequence(s)
            for s in sequence_idx.long().cpu().numpy()
        ]
        # softmax to probabilities, and only grab that for the argmax index
        sequence_probs = F.softmax(sequence_logits, dim=-1)
        sequence_probs = torch.gather(sequence_probs, dim=-1, index=argmax_idx.unsqueeze(-1)).squeeze(-1)
        return sequence_probs, sequence_idx, sequence_str


class LatentToStructure:
    def __init__(self, device, esmfold: ESMFold = None):
        self.esmfold = ESMFold(make_lm=False, make_trunk=True) if esmfold is None else esmfold
        self.esmfold.set_chunk_size(64)
        self.esmfold.eval().requires_grad_(False)
        self.esmfold.to(device)
        self.device = device
        assert not self.esmfold.trunk is None

    @torch.no_grad()
    def to_structure(
        self,
        latent: ArrayLike,
        sequences: T.List[str],
        num_recycles: int = 1,
        batch_size: T.Optional[int] = None,
    ) -> T.Tuple[T.List[PathLike], T.List[PathLike]]:
        # TODO: allow for variable dimensions latent
        latent = to_tensor(latent, device=self.device)
        batch_size = latent.shape[0] if batch_size is None else batch_size
        N, L, _ = latent.shape
        aatype, mask, residx, _, _ = batch_encode_sequences(sequences)
        aatype, mask, residx = tuple(
            map(lambda x: x.to(self.device), (aatype, mask, residx))
        )

        metrics = []
        all_pdb_strs = []
        
        for start in trange(0, len(latent), batch_size, desc="(Generating structure from latents..)"):
            torch.cuda.empty_cache()
            # https://github.com/facebookresearch/esm/blob/main/esm/esmfold/v1/esmfold.py#L208
            # utils.print_cuda_memory_usage()
            s_, aa_, mask_, residx_ = tuple(
                map(
                    lambda x: x[start : start + batch_size],
                    (latent, aatype, mask, residx),
                )
            )
            z_ = latent.new_zeros(s_.shape[0], L, L, ESMFOLD_Z_DIM).to(self.device)
            
            with torch.no_grad():
                output = self.esmfold.folding_trunk(
                    s_s_0=s_,
                    s_z_0=z_,
                    aa=aa_,
                    residx=residx_,
                    mask=mask_,
                    num_recycles=num_recycles,
                )
            metrics.append(pd.DataFrame(outputs_to_avg_metric(output)))
            all_pdb_strs.extend(output_to_pdb(output))
        metrics = pd.concat(metrics)
        return all_pdb_strs, metrics

if __name__ == "__main__":
    import torch
    from plaid.utils._proteins import LatentToSequence, LatentToStructure
    device = torch.device("cuda:3")
    sequence_constructor = LatentToSequence(device, strategy="onehot_categorical")
    structure_constructor = LatentToStructure(device)
    latent = torch.randn(16, 128, 1024).to(device)
    _, _, strs = sequence_constructor.to_sequence(latent)    
    # structure_constructor.to_structure(latent, strs)

