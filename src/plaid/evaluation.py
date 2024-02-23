import math
import typing as T

import numpy as np
from scipy import linalg
import torch
from torch.nn.functional import nll_loss
from transformers import AutoTokenizer, AutoModelForCausalLM

from .utils import to_tensor


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
        """Calculates the average perplexity under RITA for a batch of strings"""
        if not len(set([len(s) for s in all_sequences])) == 1:
            raise NotImplementedError(
                "Batched calculation only supports sequences of the same length at the moment."
            )

        batch_size = len(all_sequences) if not batch_size else batch_size
        all_perplexities = []
        for i in range(0, len(all_sequences), batch_size):
            sequences = all_sequences[i : i + batch_size]
            input_ids = self.tokenizer.batch_encode_plus(sequences)["input_ids"]
            input_ids = to_tensor(input_ids, device=self.device)
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


"""
Clean FID implementations
https://arxiv.org/abs/2104.11222
"""


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


"""
Compute the KID score given the sets of features
"""


def parmar_kid(feats1, feats2, num_subsets=100, max_subset_size=1000):
    n = feats1.shape[1]
    m = min(min(feats1.shape[0], feats2.shape[0]), max_subset_size)
    t = 0
    for _subset_idx in range(num_subsets):
        x = feats2[np.random.choice(feats2.shape[0], m, replace=False)]
        y = feats1[np.random.choice(feats1.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    kid = t / num_subsets / m
    return float(kid)


def parmar_fid(feats1, feats2):
    mu1, sig1 = np.mean(feats1, axis=0), np.cov(feats1, rowvar=False)
    mu2, sig2 = np.mean(feats2, axis=0), np.cov(feats2, rowvar=False)
    try:
        return frechet_distance(mu1, sig1, mu2, sig2)
    except ValueError as e:
        print(e)
        return np.nan


# https://github.com/RosettaCommons/RFDesign/blob/98f7435944068f0b8a864eef3029a0bad8e530ca/hallucination/util/metrics.py

def lDDT(ca0,ca,s=0.001):
    '''smooth lDDT:
    s=0.35  - good for training (smooth)
    s=0.001 - (or smaller) good for testing
    '''
    L = ca0.shape[0]
    # Computes batched the p-norm distance between each pair of the two collections of row vectors.
    d0 = torch.cdist(ca0,ca0)
    d0 = d0 + 999.9*torch.eye(L,device=ca0.device) # exclude diagonal
    i,j = torch.where(d0<15.0)
    d = torch.cdist(ca,ca)
    dd = torch.abs(d0[i,j]-d[i,j])+1e-3
    def f(x,m,s):
        return 0.5*torch.erf((torch.log(dd)-np.log(m))/(s*2**0.5))+0.5
    lddt = torch.stack([f(dd,m,s) for m in [0.5,1.0,2.0,4.0]],dim=-1).mean()
    return 1.0-lddt


def RMSD(P, Q):
    '''Kabsch algorthm'''
    def rmsd(V, W):
        return torch.sqrt( torch.sum( (V-W)*(V-W) ) / len(V) )
    def centroid(X):
        return X.mean(axis=0)

    cP = centroid(P)
    cQ = centroid(Q)
    P = P - cP
    Q = Q - cQ

    # Computation of the covariance matrix
    C = torch.mm(P.T, Q)

    # Computate optimal rotation matrix using SVD
    V, S, W = torch.svd(C)

    # get sign( det(V)*det(W) ) to ensure right-handedness
    d = torch.ones([3,3],device=P.device)
    d[:,-1] = torch.sign(torch.det(V) * torch.det(W))

    # Rotation matrix U
    U = torch.mm(d*V, W.T)

    # Rotate P
    rP = torch.mm(P, U)

    # get RMS
    rms = rmsd(rP, Q)

    return rms #, rP


def KL(P, Q, eps=1e-6):
    '''KL-divergence between two sets of 6D coords'''
    kl = [(Pi*torch.log((Pi+eps)/(Qi+eps))).sum(0).mean() for Pi,Qi in zip(P,Q)]
    kl = torch.stack(kl).mean()
    return kl



# https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/validation_metrics.py
# 
# Copyright 2021 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def drmsd(structure_1, structure_2, mask=None):
    def prep_d(structure):
        d = structure[..., :, None, :] - structure[..., None, :, :]
        d = d ** 2
        d = torch.sqrt(torch.sum(d, dim=-1))
        return d

    d1 = prep_d(structure_1)
    d2 = prep_d(structure_2)

    drmsd = d1 - d2
    drmsd = drmsd ** 2
    if(mask is not None):
        drmsd = drmsd * (mask[..., None] * mask[..., None, :])
    drmsd = torch.sum(drmsd, dim=(-1, -2))
    n = d1.shape[-1] if mask is None else torch.min(torch.sum(mask, dim=-1))
    drmsd = drmsd * (1 / (n * (n - 1))) if n > 1 else (drmsd * 0.)
    drmsd = torch.sqrt(drmsd)

    return drmsd


def drmsd_np(structure_1, structure_2, mask=None):
    structure_1 = torch.tensor(structure_1)
    structure_2 = torch.tensor(structure_2)
    if(mask is not None):
        mask = torch.tensor(mask)

    return drmsd(structure_1, structure_2, mask)


def gdt(p1, p2, mask, cutoffs):
    n = torch.sum(mask, dim=-1)
    
    p1 = p1.float()
    p2 = p2.float()
    distances = torch.sqrt(torch.sum((p1 - p2)**2, dim=-1))
    scores = []
    for c in cutoffs:
        score = torch.sum((distances <= c) * mask, dim=-1) / n
        score = torch.mean(score)
        scores.append(score)

    return sum(scores) / len(scores)


def gdt_ts(p1, p2, mask):
    return gdt(p1, p2, mask, [1., 2., 4., 8.])


def gdt_ha(p1, p2, mask):
    return gdt(p1, p2, mask, [0.5, 1., 2., 4.])
