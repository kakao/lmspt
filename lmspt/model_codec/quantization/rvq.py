"""
From DAC: https://github.com/descriptinc/descript-audio-codec/blob/main/dac/model
"""
from typing import Union, Optional

import numpy as np
import torch
import torch.nn as nn

try:
    from torch.nn.utils import weight_norm
except:
    from torch.nn.utils.parameterizations import weight_norm

from torch import distributed
from einops import rearrange
from lmspt.utils.attrdict import AttrDict as edict
from lmspt.model_codec.quantization.vq import VectorQuantize, EMAVectorQuantize


class ResidualVectorQuantize(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.quantizers = nn.ModuleList(
            [
                VectorQuantize(input_dim, codebook_size, codebook_dim[i])
                for i in range(n_codebooks)
            ]
        )
        self.quantizer_dropout = quantizer_dropout

    def forward(self, z, n_quantizers: int = None, possibly_no_quantizer=False):
        """Quantized the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors
        Parameters
        ----------
        z : Tensor[B x D x T]
        n_quantizers : int, optional
            No. of quantizers to use
            (n_quantizers < self.n_codebooks ex: for quantizer dropout)
            Note: if `self.quantizer_dropout` is True, this argument is ignored
                when in training mode, and a random number of quantizers is used.
        Returns
        -------
        dict
            A dictionary with the following keys:

            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
        """
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []
        latents = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            if possibly_no_quantizer:
                dropout = torch.randint(0, self.n_codebooks + 1, (z.shape[0],))
            else:
                dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break

            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(
                residual
            )

            if i == 0:
                z_q_1 = z_q_i.clone()  # latent after first quantization

            # Create mask to apply quantizer dropout
            mask = (
                torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            )
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            # Sum losses
            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()

            codebook_indices.append(indices_i)
            latents.append(z_e_i)
        codes = torch.stack(codebook_indices, dim=1)
        latents = torch.cat(latents, dim=1)

        return z_q, codes, latents, commitment_loss, codebook_loss, z_q_1

    def from_codes(self, codes: torch.Tensor):
        """Given the quantized codes, reconstruct the continuous representation
        Parameters
        ----------
        codes : Tensor[B x N x T]
            Quantized discrete representation of input
        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        """
        z_q = 0.0
        z_p = []
        n_codebooks = codes.shape[1]
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i
        return z_q, torch.cat(z_p, dim=1), codes

    def from_latents(self, latents: torch.Tensor):
        """Given the unquantized latents, reconstruct the
        continuous representation after quantization.

        Parameters
        ----------
        latents : Tensor[B x N x T]
            Continuous representation of input after projection

        Returns
        -------
        Tensor[B x D x T]
            Quantized representation of full-projected space
        Tensor[B x D x T]
            Quantized representation of latent space
        """
        z_q = 0
        z_p = []
        codes = []
        dims = np.cumsum([0] + [q.codebook_dim for q in self.quantizers])

        n_codebooks = np.where(dims <= latents.shape[1])[0].max(axis=0, keepdims=True)[
            0
        ]
        for i in range(n_codebooks):
            j, k = dims[i], dims[i + 1]
            z_p_i, codes_i = self.quantizers[i].decode_latents(latents[:, j:k, :])
            z_p.append(z_p_i)
            codes.append(codes_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i

        return z_q, torch.cat(z_p, dim=1), torch.stack(codes, dim=1)

    def encode(self, z: torch.Tensor, n_q: Optional[int] = None, st: Optional[int]= None) -> torch.Tensor:
        residual = z
        all_indices = []
        n_q = n_q or len(self.quantizers)
        st = st or 0
        for layer in self.quantizers[st:n_q]:
            quantized, indices = layer.encode(residual)
            residual -= quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, codes: torch.Tensor):
        z_q = 0
        for i, indices in enumerate(codes):
            quantized = self.quantizers[i].decode(indices)
            z_q += quantized
        return z_q


class DualResidualVectorQuantize(nn.Module):
    def __init__(
        self,
        n_codebooks: int = 8,
        n_semantic_codebooks: int = 1,
        input_dim: int = 1024,
        acoustic_codebook_dim: int = 1024,
        semantic_codebook_dim: int = 1024,
        acoustic_codebook_size: int = 1024,
        semantic_codebook_size: int = 1024,
        semantic_quantizer: str = None,
        quantizer_dropout: float = 0.0,
        semantic_subtraction: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert n_codebooks > n_semantic_codebooks, (
            f"Number of quantizers {n_codebooks} must be larger "
            f"than the number of semantic quantizers {n_semantic_codebooks}."
        )
        self.max_n_q = n_codebooks
        self.n_q_semantic = n_semantic_codebooks
        self.n_q_acoustic = n_codebooks - n_semantic_codebooks
        self.semantic_subtraction = semantic_subtraction

        self.rvq_first = None

        self.use_rvq = False
        self.use_fsq = False

        self.semantic_codebook_size = semantic_codebook_size
        self.acoustic_codebook_size = acoustic_codebook_size
        self.semantic_codebook_dim = semantic_codebook_dim
        self.acoustic_codebook_dim = acoustic_codebook_dim

        semantic_quantizer_kwargs = kwargs.pop("semantic_quantizer_kwargs", {})
        if semantic_quantizer_kwargs is not None:
            semantic_quantizer_kwargs["input_dim"] = input_dim

        self.build_semantic_quantizer(
            input_dim=input_dim,
            n_codebooks=self.n_q_semantic,
            codebook_size=semantic_codebook_size,
            codebook_dim=semantic_codebook_dim,
            semantic_quantizer=semantic_quantizer,
            semantic_quantizer_kwargs=semantic_quantizer_kwargs,
            **kwargs
        )

        self.rvq_rest = ResidualVectorQuantize(
            input_dim=input_dim,
            codebook_size=acoustic_codebook_size,
            codebook_dim=acoustic_codebook_dim,
            n_codebooks=self.n_q_acoustic,
            quantizer_dropout=quantizer_dropout,
        )

        self.vocab_usage_record_times: int = 0
        self.register_buffer('semantic_vocab_usage', torch.zeros(n_semantic_codebooks, self.semantic_codebook_size))
        self.register_buffer('acoustic_vocab_usage', torch.zeros(n_codebooks - n_semantic_codebooks, self.acoustic_codebook_size))

    def reset_vocab_usage(self):
        self.semantic_vocab_usage.zero_()
        self.acoustic_vocab_usage.zero_()
        self.vocab_usage_record_times = 0

    def build_semantic_quantizer(
            self,
            input_dim: int = 1024,
            n_codebooks: int = 1,
            codebook_size: int = 1024,
            codebook_dim: int = 1024,
            semantic_quantizer: str = None,
            semantic_quantizer_kwargs: dict = {},
            **kwargs,
    ):
        if semantic_quantizer == "fsq":
            from lmspt.model_codec.quantization.fsq import FSQ
            self.rvq_first = FSQ(**semantic_quantizer_kwargs)
            self.use_fsq = True
            self.semantic_codebook_size = np.prod(semantic_quantizer_kwargs.get("levels"))
        else:
            self.rvq_first = ResidualVectorQuantize(
                input_dim=input_dim,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                n_codebooks=1,
                quantizer_dropout=False,
                **kwargs
            )
            self.use_rvq = True

    def get_vocab_usage(self, codes, vocab_size, vocab_usage):
        device = codes.device
        prob_per_class_is_chosen = torch.zeros(len(codes), vocab_size).to(device)

        for q, code in enumerate(codes):
            _cnt = code.reshape(-1).bincount(minlength=vocab_size).float()
            prob_per_class_is_chosen[q] = _cnt

        handler = distributed.all_reduce(prob_per_class_is_chosen, async_op=True)
        if handler is not None:
            handler.wait()
        prob_per_class_is_chosen /= prob_per_class_is_chosen.sum()

        if self.vocab_usage_record_times == 0:
            vocab_usage.copy_(prob_per_class_is_chosen)
        elif self.vocab_usage_record_times < 100:
            vocab_usage.mul_(0.9).add_(prob_per_class_is_chosen, alpha=0.1)
        else:
            vocab_usage.mul_(0.99).add_(prob_per_class_is_chosen, alpha=0.01)

        ret = (vocab_usage > 0 / vocab_size).float().mean()
        return ret

    def forward(
            self,
            x_semantic: torch.Tensor,
            x_acoustic: torch.Tensor,
    ):
        s_z, s_codes, s_latents, s_commit_loss, s_codebook_loss = self._forward_semantic_quantizer(x_semantic)
        if self.semantic_subtraction:
            x_acoustic -= s_z
        a_z, a_codes, a_latents, a_commit_loss, a_codebook_loss, _ = self.rvq_rest(x_acoustic)
        z = s_z + a_z

        codes = torch.cat(
            [s_codes, a_codes], dim=1
        ) # B x n_q x T

        info = {
            "semantic_commit_loss": s_commit_loss,
            "semantic_codebook_loss": s_codebook_loss,
            "acoustic_commit_loss": a_commit_loss,
            "acoustic_codebook_loss": a_codebook_loss,
            "metrics": {}
        }

        if self.training and distributed.is_initialized():
            semantic_vocab_usage = self.get_vocab_usage(s_codes.transpose(1, 0), self.semantic_codebook_size,
                                                        self.semantic_vocab_usage)
            acoustic_vocab_usage = self.get_vocab_usage(a_codes.transpose(1, 0), self.acoustic_codebook_size,
                                                        self.acoustic_vocab_usage)
            self.vocab_usage_record_times += 1

            info["metrics"].update({
                "semantic/vocab_usage": semantic_vocab_usage.item(),
                "acoustic/vocab_usage": acoustic_vocab_usage.item(),
            })

        return z, codes, edict(info), [s_z, a_z]

    def _forward_semantic_quantizer(self, x):
        if self.use_rvq:
            z, codes, latents, commit_loss, codebook_loss, _ = self.rvq_first(x)
        elif self.use_fsq:
            z, codes = self.rvq_first(x)
            codes = codes.unsqueeze(1)
            commit_loss = torch.tensor(0., device=x.device, requires_grad=False)
            codebook_loss = torch.tensor(0., device=x.device, requires_grad=False)
            latents = None
        return z, codes, latents, commit_loss, codebook_loss

    def _encode_semantic(self, x):
        if self.use_rvq:
            codes = self.rvq_first.encode(x)
        elif self.use_fsq:
            codes = self.rvq_first.encode(x)
        return codes

    def decode_semantic(self, codes):
        quantized = self.rvq_first.decode(codes[:self.n_q_semantic])
        return quantized

    def encode(self, x_semantic: torch.Tensor, x_acoustic: torch.Tensor) -> torch.Tensor:
        codes = self._encode_semantic(x_semantic)
        if self.semantic_subtraction:
            s_z = self.rvq_first.decode(codes)
            x_acoustic -= s_z
        acoustic_codes = self.rvq_rest.encode(x_acoustic)
        codes = torch.cat([codes, acoustic_codes], dim=0)
        # codes is [K, B, T], with T frames, K nb of codebooks.
        return codes

    def decode(self, codes: torch.Tensor, acoustic_only=False) -> torch.Tensor:
        # codes is [K, B, T], with T frames, K nb of codebooks.
        if not acoustic_only:
            quantized = self.decode_semantic(codes)
        else:
            quantized = 0
        if codes.shape[0] > self.n_q_semantic:
            acoustic_quantized = self.rvq_rest.decode(codes[self.n_q_semantic:])
            quantized += acoustic_quantized
        return quantized


if __name__ == "__main__":
    # rvq = ResidualVectorQuantize(quantizer_dropout=True)
    # x = torch.randn(16, 512, 80)
    # y = rvq(x)
    # print(y["latents"].shape)

    dual_rvq = DualResidualVectorQuantize(
        input_dim=1024,
        n_codebooks=8,
        n_semantic_codebooks=1,
        semantic_codebook_size=2048,
        semantic_codebook_dim=8,
        acoustic_codebook_size=2048,
        acoustic_codebook_dim=8,
        semantic_quantizer="rvq",
        quantizer_dropout=False,
        semantic_quantizer_kwargs={
            "levels": [8,8,8,6,5]
        },
        semantic_subtraction=True
    )

    frame_rate = 12.5
    segment_size = 4
    T = int(frame_rate * segment_size)

    output_sample_rate = 16000
    rates = [8,8,5,4]

    output_sample_rate = 24000
    rates = [8,8,6,5]

    from lmspt.model_codec.dac_model import Decoder
    decoder = Decoder(
        1024,
        256,
        rates,
    )

    semantic_x, acoustic_x = torch.randn(16, 1024, T), torch.randn(16, 1024, T)
    z_q, codes, vq_info, [z_q_semantic, z_q_acoustic] = dual_rvq(semantic_x, acoustic_x)

    print(z_q.shape, codes.shape)

    expected_output_size = output_sample_rate * segment_size
    recon_x = decoder(z_q)

    print(expected_output_size - recon_x.size(-1))

    codes = dual_rvq.encode(semantic_x, acoustic_x)

    print(codes.shape)

    z_q = dual_rvq.decode(codes)

    print(z_q.shape)


