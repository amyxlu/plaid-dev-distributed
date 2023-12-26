import math
import typing as T

import numpy as np
import torch
from torch.nn.functional import nll_loss
from transformers import AutoTokenizer, AutoModelForCausalLM

from . import utils


def polynomial_kernel(x, y):
    d = x.shape[-1]
    dot = x @ y.transpose(-2, -1)
    return (dot / d + 1) ** 3


def squared_mmd(x, y, kernel=polynomial_kernel):
    m = x.shape[-2]
    n = y.shape[-2]
    kxx = kernel(x, x)
    kyy = kernel(y, y)
    kxy = kernel(x, y)
    kxx_sum = kxx.sum([-1, -2]) - kxx.diagonal(dim1=-1, dim2=-2).sum(-1)
    kyy_sum = kyy.sum([-1, -2]) - kyy.diagonal(dim1=-1, dim2=-2).sum(-1)
    kxy_sum = kxy.sum([-1, -2])
    term_1 = kxx_sum / m / (m - 1)
    term_2 = kyy_sum / n / (n - 1)
    term_3 = kxy_sum * 2 / m / n
    return term_1 + term_2 - term_3


def calc_kid_fn(x, y, max_size=5000):
    x_size, y_size = x.shape[0], y.shape[0]
    n_partitions = math.ceil(max(x_size / max_size, y_size / max_size))
    total_mmd = x.new_zeros([])
    for i in range(n_partitions):
        cur_x = x[
            round(i * x_size / n_partitions) : round((i + 1) * x_size / n_partitions)
        ]
        cur_y = y[
            round(i * y_size / n_partitions) : round((i + 1) * y_size / n_partitions)
        ]
        total_mmd = total_mmd + squared_mmd(cur_x, cur_y)
    return total_mmd / n_partitions


class _MatrixSquareRootEig(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        vals, vecs = torch.linalg.eigh(a)
        ctx.save_for_backward(vals, vecs)
        return vecs @ vals.abs().sqrt().diag_embed() @ vecs.transpose(-2, -1)

    @staticmethod
    def backward(ctx, grad_output):
        vals, vecs = ctx.saved_tensors
        d = vals.abs().sqrt().unsqueeze(-1).repeat_interleave(vals.shape[-1], -1)
        vecs_t = vecs.transpose(-2, -1)
        return vecs @ (vecs_t @ grad_output @ vecs / (d + d.transpose(-2, -1))) @ vecs_t


def sqrtm_eig(a):
    if a.ndim < 2:
        raise RuntimeError("tensor of matrices must have at least 2 dimensions")
    if a.shape[-2] != a.shape[-1]:
        raise RuntimeError("tensor must be batches of square matrices")
    return _MatrixSquareRootEig.apply(a)


def calc_fid_fn(x, y, eps=1e-8):
    x_mean = x.mean(dim=0)
    y_mean = y.mean(dim=0)
    mean_term = (x_mean - y_mean).pow(2).sum()
    x_cov = torch.cov(x.T)
    y_cov = torch.cov(y.T)
    eps_eye = torch.eye(x_cov.shape[0], device=x_cov.device, dtype=x_cov.dtype) * eps
    x_cov = x_cov + eps_eye
    y_cov = y_cov + eps_eye
    x_cov_sqrt = sqrtm_eig(x_cov)
    cov_term = torch.trace(
        x_cov + y_cov - 2 * sqrtm_eig(x_cov_sqrt @ y_cov @ x_cov_sqrt)
    )
    return mean_term + cov_term



class RITAPerplexity:
    def __init__(self, device):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            "lightonai/RITA_xl", trust_remote_code=True
        )
        self.model.to(self.device)
        self.model.eval().requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained("lightonai/RITA_xl")

    def calc_perplexity(self, sequence):
        """Calculates the perplexity under RITA for a single model"""
        input_ids = torch.tensor(self.tokenizer.encode(sequence)).unsqueeze(0)
        input_ids = input_ids.to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]
        return math.exp(loss)

    def batch_eval(self, all_sequences, batch_size: int = None):
        """ Calculates the average perplexity under RITA for a batch of strings"""
        if not len(set([len(s) for s in all_sequences])) == 1:
            raise NotImplementedError("Batched calculation only supports sequences of the same length at the moment.")
        
        batch_size = len(all_sequences) if not batch_size else batch_size
        all_perplexities = []
        for i in range(0, len(all_sequences), batch_size):
            sequences = all_sequences[i : i + batch_size]
            input_ids = self.tokenizer.batch_encode_plus(sequences)['input_ids']
            input_ids = utils.to_tensor(input_ids, device=self.device)
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
            loss, logits = outputs[:2]
            all_perplexities.append(torch.exp(loss).item())
        
        return np.mean(all_perplexities)


class ESMPseudoPerplexity:
    """Follows the per-sequence definition in the Lin et al., 2022 appendix."""

    def __init__(self, device, esm_model_name: str = "esm2_t48_15B_UR50D"):
        self.device = device
        model, alphabet = torch.hub.load("facebookresearch/esm:main", esm_model_name)
        self.pad_idx = alphabet.padding_idx
        self.nlayers = int(esm_model_name.split("_")[1][1:])
        self.batch_converter = alphabet.get_batch_converter()
        self.model = model.to(device)
        self.model.eval()

    def batch_calc_perplexity(self, sequences: T.List[str]):
        labels, strs, tokens = self.batch_converter(sequences)
        B, L, _ = tokens.shape
        perplexities = []

        # at each position, replace the token with a mask token and calculate the "perplexity"
        for pos in range(len(L)):
            tokens_ = tokens.clone()
            tokens_[:, pos] = torch.where(
                tokens[:, pos] == self.pad_idx, self.pad_idx, self.mask_idx
            )
            with torch.no_grad():
                results = self.model(
                    tokens_.to(self.device),
                    repr_layers=[self.nlayers - 1],
                    return_contacts=False,
                )
                nll = nll_loss(
                    results["logits"], labels.to(self.device), ignore_index=self.pad_idx
                )
                perplexities.append(torch.exp(nll).item())
        return perplexities
    