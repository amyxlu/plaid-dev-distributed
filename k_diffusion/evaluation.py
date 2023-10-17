import math
import os
from pathlib import Path
import typing as T

from cleanfid.inception_torchscript import InceptionV3W
import clip
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from tqdm.auto import trange

from . import utils
from .models.esmfold import ESMFold
import einops


class ESMFoldLatentFeatureExtractor(nn.Module):
    def __init__(self, esmfold, device="cpu"):
        """Note: even though when sampling, we may use a smaller dimension than 1024, for
        feature evaluation, we use the original dimension (i.e. 1024) for consistency.
        """
        super().__init__()
        self.esmfold = esmfold.eval().requires_grad_(False)
        self.esmfold.to(device)

    def load_saved_features(self, location, device="cpu"):
        import safetensors.torch as st
        return st.load_file(location)["features"].to(device=device)

    def forward(self, sequences: T.List[str], max_len: int = 512, min_len: int = 30):
        sequences = utils.get_random_sequence_crop_batch(
            sequences, max_len=max_len, min_len=min_len
        )
        with torch.no_grad():
            embed_results = self.esmfold.infer_embedding(sequences)
            feats = embed_results["s"].detach()  # (N, L, 1024)
            masks = embed_results["mask"].detach()  # (N, L)

        # mask-aware mean pool over the length dimension
        N, L, C = feats.shape
        masks = einops.repeat(masks, "N L -> N L C", C=C)  # (N, L, 1024)
        feats = feats * masks  # (N, L, 1024)
        feats = feats.sum(dim=1) / masks.sum(dim=1)  # (N, 1024)
        return feats


class InceptionV3FeatureExtractor(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        path = (
            Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
            / "k-diffusion"
        )
        url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt"
        digest = "f58cb9b6ec323ed63459aa4fb441fe750cfe39fafad6da5cb504a16f19e958f4"
        utils.download_file(path / "inception-2015-12-05.pt", url, digest)
        self.model = InceptionV3W(str(path), resize_inside=False).to(device)
        self.size = (299, 299)

    def forward(self, x):
        x = F.interpolate(
            x, self.size, mode="bicubic", align_corners=False, antialias=True
        )
        if x.shape[1] == 1:
            x = torch.cat([x] * 3, dim=1)
        x = (x * 127.5 + 127.5).clamp(0, 255)
        return self.model(x)


class CLIPFeatureExtractor(nn.Module):
    def __init__(self, name="ViT-B/16", device="cpu"):
        super().__init__()
        self.model = clip.load(name, device=device)[0].eval().requires_grad_(False)
        self.normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )
        self.size = (
            self.model.visual.input_resolution,
            self.model.visual.input_resolution,
        )

    @classmethod
    def available_models(cls):
        return clip.available_models()

    def forward(self, x):
        x = (x + 1) / 2
        x = F.interpolate(
            x, self.size, mode="bicubic", align_corners=False, antialias=True
        )
        if x.shape[1] == 1:
            x = torch.cat([x] * 3, dim=1)
        x = self.normalize(x)
        x = self.model.encode_image(x).float()
        x = F.normalize(x) * x.shape[-1] ** 0.5
        return x


class DINOv2FeatureExtractor(nn.Module):
    def __init__(self, name="vitl14", device="cpu"):
        super().__init__()
        self.model = (
            torch.hub.load("facebookresearch/dinov2", "dinov2_" + name)
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )
        self.size = 224, 224

    @classmethod
    def available_models(cls):
        return ["vits14", "vitb14", "vitl14", "vitg14"]

    def forward(self, x):
        x = (x + 1) / 2
        x = F.interpolate(
            x, self.size, mode="bicubic", align_corners=False, antialias=True
        )
        if x.shape[1] == 1:
            x = torch.cat([x] * 3, dim=1)
        x = self.normalize(x)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            x = self.model(x).float()
        x = F.normalize(x) * x.shape[-1] ** 0.5
        return x


def compute_features(accelerator, sample_fn, extractor_fn, n, batch_size):
    n_per_proc = math.ceil(n / accelerator.num_processes)
    feats_all = []
    try:
        for i in trange(
            0, n_per_proc, batch_size, disable=not accelerator.is_main_process, desc="(Evaluation)", leave=False 
        ):
            cur_batch_size = min(n - i, batch_size)
            samples = sample_fn(cur_batch_size)[:cur_batch_size]
            samples = accelerator.gather(extractor_fn(samples))
            feats_all.append(samples)
    except StopIteration:
        pass
    return torch.cat(feats_all)[:n]


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


@utils.tf32_mode(matmul=False)
def kid(x, y, max_size=5000):
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


@utils.tf32_mode(matmul=False)
def fid(x, y, eps=1e-8):
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


import typing as T
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn.functional import nll_loss


class RITAPerplexity:
    def __init__(self, device):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            "lightonai/RITA_xl", trust_remote_code=True
        )
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("lightonai/RITA_xl")

    def calc_perplexity(self, sequence):
        input_ids = torch.tensor(self.tokenizer.encode(sequence)).unsqueeze(0)
        input_ids = input_ids.to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]
        return math.exp(loss)

    def batch_calc_perplexity(self, sequences):
        # todo: https://huggingface.co/transformers/v2.11.0/main_classes/tokenizer.html
        # need to figure out how huggingface deals with batches
        return [self.calc_perplexity(seq) for seq in sequences]


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


# def reconstruct(
#     latent,
#     device,
#     sequence_reconstructor,
#     structure_reconstructor,
#     construct_structure=True,
#     calc_perplexity=True,
#     num_recycles=1,
#     outdir=None,
# ):
#     """Bundle sequence and structure reconstruction together.
#     If outdir is specified, will write to disk, otherwise simply return the results."""

#     sequence_results = {}
#     latent = latent.to(device)
#     probs, _, sequences = sequence_reconstructor.to_sequence(latent)
#     # TODO: truncate and mask if using vocab_23

#     sequence_results = pd.DataFrame(
#         {
#             "sequences": sequences,
#             "mean_residue_confidence": probs.mean(dim=1).cpu().numpy(),
#             # add additional log-to-disk metrics here
#         }
#     )

#     # maybe calculate sequence perplexities -- since RITA takes up memory, this is optional
#     if calc_perplexity:
#         perplexity_calculator = perplexity.RITAPerplexity(device)
#         perplexities = perplexity_calculator.batch_calc_perplexity(sequences)
#         sequence_results["perplexity"] = utils.npy(perplexities)

#     # make structure generation optional
#     if construct_structure:
#         pdbstrs, structure_results = structure_reconstructor.to_structure(
#             latent, sequences, num_recycles=num_recycles
#         )
#         structure_results = pd.DataFrame(structure_results)
#     else:
#         pdbstrs = None
#         structure_results = None

#     # maybe write things to disk
#     if not outdir is None:
#         outdir = Path(outdir)

#         # write resulting sequences as a FASTA for downstream OmegaFold, etc.
#         utils.write_to_fasta(sequences, outdir / "generated_sequences.fasta")

#         # write auxiliary information to disk
#         sequence_results.to_csv(outdir / "generated_sequences.csv", index=False)

#         if not structure_results is None:
#             # write individual PDB strings to disk
#             for i, pdbstr in enumerate(pdbstrs):
#                 utils.write_pdb_to_disk(
#                     pdbstr, outdir / "generated_structures" / f"sample_{i}.pdb"
#                 )

#             # write auxiliary information to disk
#             structure_results.to_csv(
#                 outdir / "generated_structures" / "structure_confidence.csv", index=False
#             )

#     # return for in-program analysis
#     return sequence_results, structure_results, pdbstrs