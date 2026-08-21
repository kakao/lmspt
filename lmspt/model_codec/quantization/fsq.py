"""
Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
Code adapted from Jax version in Appendix A.1
"""

from __future__ import annotations
from functools import wraps, partial
from contextlib import nullcontext
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor, int32
from torch.amp import autocast

from einops import rearrange, pack, unpack
from lmspt.model_codec.dac_layers import WNConv1d


# helper functions

def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None


def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)

    return inner


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


# tensor helpers

def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


# main class

class FSQ(Module):
    def __init__(
            self,
            levels: List[int],
            input_dim: int | None = None,
            num_codebooks=1,
            keep_num_codebooks_dim: bool | None = None,
            scale: float | None = None,
            allowed_dtypes: Tuple[torch.dtype, ...] = (torch.float32, torch.float64),
            channel_first: bool = False,
            projection_has_bias: bool = True,
            return_indices=True,
            force_quantization_f32=True
    ):
        super().__init__()
        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.input_dim = default(input_dim, len(_levels) * num_codebooks)

        self.channel_first = channel_first

        self.in_proj = WNConv1d(self.input_dim, effective_codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(effective_codebook_dim, self.input_dim, kernel_size=1)

        self.return_indices = return_indices
        if return_indices:
            self.codebook_size = self._levels.prod().item()
            implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))
            self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

        self.allowed_dtypes = allowed_dtypes
        self.force_quantization_f32 = force_quantization_f32

    def bound(self, z, eps: float = 1e-3):
        """ Bound `z`, an array of shape (..., d). """
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z):
        """ Quantizes z, returns quantized zhat, same shape as z. """
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized):
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def codes_to_indices(self, zhat):
        """ Converts a `code` to an index in the codebook. """
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_level_indices(self, indices):
        """ Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings """
        if indices.dim() == 1:
            indices = rearrange(indices, '... -> ... 1')
        else:
            indices = indices.squeeze(1)
            indices = rearrange(indices, 'b n -> (b n) 1')

        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def indices_to_codes(self, indices):
        """ Inverse of `codes_to_indices`. """
        assert exists(indices)

        b, _, n = indices.shape
        codes = self._indices_to_codes(indices)
        codes = rearrange(codes, '(b n) c -> b c n', b=b, c=self.codebook_dim, n=n)
        codes = self.out_proj(codes)
        return codes

    def encode(self, x):
        z = self.in_proj(x)
        z = z.transpose(2, 1) # b n c
        z = rearrange(z, 'b n (c d) -> b n c d', c=self.num_codebooks)
        force_f32 = self.force_quantization_f32
        quantization_context = partial(autocast, 'cuda', enabled=False) if force_f32 else nullcontext

        with quantization_context():
            orig_dtype = z.dtype

            if force_f32 and orig_dtype not in self.allowed_dtypes:
                z = z.float()

            codes = self.quantize(z)

            # returning indices could be optional
            codes = self.codes_to_indices(codes)
            codes = rearrange(codes, 'b n c -> b c n')
            codes = rearrange(codes, 'b c n -> c b n')

        return codes

    def decode(self, indices):
        indices = rearrange(indices, 'c b n -> b c n')
        codes = self.indices_to_codes(indices)
        codes = codes.squeeze(-1)
        return codes

    def forward(self, z):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """

        z = self.in_proj(z)
        z = z.transpose(2, 1)  # b x n x d
        z = rearrange(z, 'b n (c d) -> b n c d', c=self.num_codebooks)

        # whether to force quantization step to be full precision or not

        force_f32 = self.force_quantization_f32
        quantization_context = partial(autocast, 'cuda', enabled=False) if force_f32 else nullcontext

        with quantization_context():
            orig_dtype = z.dtype

            if force_f32 and orig_dtype not in self.allowed_dtypes:
                z = z.float()

            codes = self.quantize(z)

            # returning indices could be optional

            indices = None

            if self.return_indices:
                indices = self.codes_to_indices(codes)

            codes = rearrange(codes, 'b n c d -> b n (c d)')

            codes = codes.type(orig_dtype)

        # project out
        codes = codes.transpose(2, 1)
        out = self.out_proj(codes)

        # reconstitute image or video dimensions

        if not self.keep_num_codebooks_dim and self.return_indices:
            indices = maybe(rearrange)(indices, '... 1 -> ...')

        # return quantized output and indices

        return out, indices
