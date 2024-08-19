import typing as T
import torch
import torch.nn as nn
import abc
import einops
from functools import partial

from . import (
    # timesteps
    TimestepEmbedder,
    FourierTimestepEmbedder,
    # positional embeddings
    RotaryEmbedding,
    SinusoidalPosEmb,
    # auxiliary
    LabelEmbedder,
    Mlp,
)

# named input tuple
from .. import DenoiserKwargs

# label embedding sizes
from ...datasets import NUM_FUNCTION_CLASSES, NUM_ORGANISM_CLASSES


class BaseDenoiser(nn.Module):
    def __init__(
        self,
        input_dim=32,
        hidden_size=768,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        use_self_conditioning=True,
        timestep_embedding_strategy: str = "fourier",
        pos_embedding_strategy: str = "rotary",
        max_seq_len: T.Optional[int] = 256,
        *args,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.use_self_conditioning = use_self_conditioning

        # cast input dimension to hidden dimension
        self.x_proj = self.make_input_projection(*args, **kwargs)

        # cast output dimension to hidden dimension
        self.output_layer = self.make_output_projection(*args, **kwargs)

        # self conditioning
        if use_self_conditioning:
            self.self_conditioning_mlp = self.make_self_conditioning_projection(*args, **kwargs) 
        else:
            self.self_conditioning_mlp = None

        # abstract methods for timesteps and positional encodings
        self.t_embedder = self.make_timestep_embedding(
            timestep_embedding_strategy, hidden_size, *args, **kwargs
        )
        self.pos_embed = self.make_positional_embedding(
            pos_embedding_strategy, hidden_size, *args, **kwargs
        )

        # label embedder
        self.function_y_embedder = LabelEmbedder(
            NUM_FUNCTION_CLASSES, hidden_size, add_cfg_embedding=True
        )
        self.organism_y_embedder = LabelEmbedder(
            NUM_ORGANISM_CLASSES, hidden_size, add_cfg_embedding=True
        )

        ############################################# 
        ############################################# 
        # must be implemented in subclasses  
        self.blocks = self.make_denoising_blocks()
        self.initialize_weights()
        ############################################# 
        ############################################# 

    """
    Denoising block initialization must be implemented in subclasses.
    """

    @abc.abstractmethod
    def make_denoising_blocks(self, *args, **kwargs):
        raise NotImplementedError
    
    @abc.abstractmethod
    def initialize_weights(self):
        raise NotImplementedError

    """
    Default projections and embedders, which can be overridden in subclasses.
    """

    def make_input_projection(self, *args, **kwargs):
        return Mlp(self.input_dim, self.hidden_size)

    def make_output_projection(self, *args, **kwargs):
        return Mlp(self.hidden_size, self.input_dim)

    def make_self_conditioning_projection(self, *args, **kwargs):
        return Mlp(self.hidden_size * 2, self.hidden_size)

    def make_positional_embedding(self, strategy: str, hidden_size: int):
        assert strategy in ["rotary", "learned", "sinusoidal", None]
        if strategy == "rotary":
            return RotaryEmbedding(dim=hidden_size)
        elif strategy == "learned":
            return nn.Parameter(torch.zeros(1, hidden_size))
        else:
            # default: sinusoidal, as is done with the original DiT paper
            assert not self.max_seq_len is None


    def make_timestep_embedding(self, strategy: str, hidden_size: int):
        """ By default, we use a frequency transform and then an MLP projection."""
        assert strategy in ["fourier", "sinusoidal", "default", None]
        if strategy == "fourier":
            return FourierTimestepEmbedder(embed_dim=hidden_size)
        else:
            # default: sinusoidal transform
            return TimestepEmbedder(hidden_size=hidden_size, frequency_embedding_size=256) 

    """
    Default forward and utility functions; subclasses can override as needed.
    """

    def xavier_init_module(self, m):
        for p in m.parameters():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            else:
                if p.dim() > 1:
                    torch.nn.init.xavier_normal_(p)

    def forward_with_cond_drop(
        self,
        denoiser_kwargs: DenoiserKwargs,
        function_y_cond_drop_prob: float,
        organism_y_cond_drop_prob: float,
    ):
        """Forward pass for diffusion training, with label dropout."""

        # unpack named tuple
        x = denoiser_kwargs.x
        t = denoiser_kwargs.t
        function_y = denoiser_kwargs.function_y
        organism_y = denoiser_kwargs.organism_y
        mask = denoiser_kwargs.mask
        x_self_cond = denoiser_kwargs.x_self_cond

        # project along the channel dimension if using self-conditioning
        if x_self_cond is not None:
            x = self.self_conditioning_mlp(torch.cat([x, x_self_cond], dim=-1))

        # project back out to the hidden size to be used by the blocks
        x = self.input_projection(x)

        # add positional embedding
        x = self.pos_embedder(x)

        # add trainable timestep embedding
        t = self.t_embedder(t)  # (N, D)

        # get function and organism label embeddings, potentially dropping out the label for classifier-free guidance training
        function_y = self.function_y_embedder(
            function_y, self.training, function_y_cond_drop_prob
        )
        organism_y = self.organism_y_embedder(
            organism_y, self.training, organism_y_cond_drop_prob
        )

        # combine timestep and label conditioning labels
        c = t + function_y + organism_y

        # if mask is not supplied, assume that nothing needs to be masked
        if mask is None:
            mask = torch.ones(x.shape[:2], device=x.device).bool()

        # pass through blocks and final layer
        for block in self.blocks:
            x = block(x, c, mask)  # (N, L, D)

        return self.final_layer(x, c)  # (N, L, out_channels)

    def forward_with_cond_scale(
        self, denoiser_kwargs: DenoiserKwargs, cond_scale: float, rescaled_phi: float
    ):
        """Forward pass for sampling model predictions, with a conditioning scale.
        Adapted from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/classifier_free_guidance.py#L355
        """

        # force conditioning: no label drop
        logits = self.forward_with_cond_drop(
            denoiser_kwargs,
            function_y_cond_drop_prob=0.0,
            organism_y_cond_drop_prob=0.0,
        )

        if cond_scale == 1:
            return logits

        # force unconditional: always no label drop
        null_logits = self.forward_with_cond_drop(
            denoiser_kwargs,
            function_y_cond_drop_prob=1.0,
            organism_y_cond_drop_prob=1.0,
        )

        # apply cond scaling factor
        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        # use rescaling technique proposed in https://arxiv.org/abs/2305.08891
        if rescaled_phi == 0.0:
            return scaled_logits

        std_fn = partial(
            torch.std, dim=tuple(range(1, scaled_logits.ndim)), keepdim=True
        )
        rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))

        return rescaled_logits * rescaled_phi + scaled_logits * (1.0 - rescaled_phi)

    def forward(
        self,
        denoiser_kwargs: DenoiserKwargs,
        use_cond_dropout: bool = False,
        **kwargs: T.Any
    ):
        if use_cond_dropout:
            return self.forward_with_cond_drop(denoiser_kwargs, **kwargs)
        else:
            return self.forward_with_cond_scale(denoiser_kwargs, **kwargs)
