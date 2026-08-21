from typing import List, Union, Optional
import random
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

from lmspt.model_codec.quantization.rvq import DualResidualVectorQuantize
from lmspt.model_codec.dac_model import Encoder, Decoder, pad_to_length
from lmspt.utils.attrdict import AttrDict as edict


class LMSPT(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 64,
        semantic_encoder_dim: int = None,
        acoustic_encoder_dim: int = None,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        n_codebooks: int = 8,
        n_semantic_codebooks: int = 1,
        codebook_size: int = 2048,
        codebook_dim: Union[int, list] = 8,
        acoustic_codebook_size: int = None,
        semantic_codebook_size: int = None,
        acoustic_codebook_dim: Union[int, list] = None,
        semantic_codebook_dim: int = None,
        quantizer_dropout: float = 1.0,
        semantic_quantizer: str = "rvq",
        semantic_quantizer_kwargs: dict = None,
        sample_rate: int = 44100,
        output_sample_rate: int = None,
        aux_decoder: nn.Module = None,
        use_shared_encoder: bool = False,
        semantic_subtraction: bool = True,
        dilation: list = [1, 3, 9],
        residual_kernel_size: int = 7,
        is_causal: bool = False,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.output_sample_rate = output_sample_rate or sample_rate
        self.is_causal = is_causal

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        semantic_encoder_dim = semantic_encoder_dim or encoder_dim

        self.semantic_encoder = Encoder(
            semantic_encoder_dim,
            encoder_rates,
            latent_dim,
            dilation=dilation,
            residual_kernel_size=residual_kernel_size,
            is_causal=is_causal,
        )

        if use_shared_encoder:
            self.acoustic_encoder = self.semantic_encoder
        else:
            acoustic_encoder_dim = acoustic_encoder_dim or encoder_dim
            self.acoustic_encoder = Encoder(
                acoustic_encoder_dim,
                encoder_rates,
                latent_dim,
                dilation=dilation,
                residual_kernel_size=residual_kernel_size,
                is_causal=is_causal,
            )

        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
            dilation=dilation,
            residual_kernel_size=residual_kernel_size,
            is_causal=is_causal,
        )
        self.aux_decoder = aux_decoder
        if self.aux_decoder is None:
            self.aux_decoder = Decoder(
                latent_dim,
                decoder_dim,
                decoder_rates,
                sample_rate=sample_rate,
                is_causal=is_causal,
            )

        self.encoder_rates = encoder_rates

        if semantic_codebook_size is None:
            semantic_codebook_size = codebook_size

        if semantic_codebook_dim is None:
            semantic_codebook_dim = codebook_dim

        if acoustic_codebook_size is None:
            acoustic_codebook_size = codebook_size

        if acoustic_codebook_dim is None:
            acoustic_codebook_dim = codebook_dim

        self.quantizer = DualResidualVectorQuantize(
            input_dim=latent_dim,
            n_codebooks=n_codebooks,
            n_semantic_codebooks=n_semantic_codebooks,
            semantic_codebook_size=semantic_codebook_size,
            semantic_codebook_dim=semantic_codebook_dim,
            acoustic_codebook_size=acoustic_codebook_size,
            acoustic_codebook_dim=acoustic_codebook_dim,
            semantic_quantizer=semantic_quantizer,
            quantizer_dropout=quantizer_dropout,
            semantic_quantizer_kwargs=semantic_quantizer_kwargs,
            semantic_subtraction=semantic_subtraction
        )

        self.quantizer_setup = {
            "input_dim": latent_dim,
            "n_codebooks": n_codebooks,
            "n_semantic_codebooks": n_semantic_codebooks,
            "semantic": {
                "quantizer": semantic_quantizer,
                "size": self.quantizer.semantic_codebook_size,
                "dim": self.quantizer.semantic_codebook_dim,
                "kwargs": semantic_quantizer_kwargs
            },
            "acoustic": {
                "quantizer": "rvq",
                "size": self.quantizer.acoustic_codebook_size,
                "dim": self.quantizer.acoustic_codebook_dim,
                "qdrop": quantizer_dropout
            }
        }

    def reset_vocab_usage(self):
        self.quantizer.reset_vocab_usage()

    @torch.no_grad()
    def encode(self, audio_data):
        z_semantic = self.semantic_encoder(audio_data)
        z_acoustic = self.acoustic_encoder(audio_data)
        codes = self.quantizer.encode(z_semantic, z_acoustic)
        return codes

    @torch.no_grad()
    def decode(self, codes, acoustic_only=False):
        # codes is [K, B, T], with T frames, K nb of codebooks.
        z_q = self.quantizer.decode(codes, acoustic_only=acoustic_only)
        x = self.decoder(z_q)
        return x

    @torch.no_grad()
    def aux_semantic_decode(self, codes):
        # codes is [K, B, T], with T frames, K nb of codebooks.
        z_q_semantic = self.quantizer.decode_semantic(codes)
        x_semantic = self.aux_decoder(z_q_semantic)
        return x_semantic

    def forward(
        self,
        audio_data: torch.Tensor,
        **kwargs
    ):
        input_length = audio_data.size(-1)
        duration_sec = input_length / self.sample_rate
        output_length = int(duration_sec * self.output_sample_rate)
        semantic_recon_length = int(duration_sec * self.aux_decoder.sample_rate)

        z_semantic = self.semantic_encoder(audio_data)
        z_acoustic = self.acoustic_encoder(audio_data)

        z_q, codes, vq_info, [z_q_semantic, _] = self.quantizer(z_semantic, z_acoustic)

        x = self.decoder(z_q)
        semantic_x = self.aux_decoder(z_q_semantic)

        x = pad_to_length(x, output_length)
        semantic_x = pad_to_length(semantic_x, semantic_recon_length)

        semantic_edict = edict(
            {
                "x": semantic_x,
                "penalty": vq_info.semantic_commit_loss,
                "vq/codebook_loss": vq_info.semantic_codebook_loss,
                "metrics": {},
            }
        )

        acoustic_edict = edict(
            {
                "x": x,
                "codes": codes,
                "penalty": vq_info.acoustic_commit_loss,
                "vq/codebook_loss": vq_info.acoustic_codebook_loss,
                "metrics": vq_info.metrics,
            }
        )

        return acoustic_edict, semantic_edict



class LMSPTConfig(PretrainedConfig):
    model_type = "lmspt"

    def __init__(
        self,
        encoder_dim: int = 64,
        semantic_encoder_dim: Optional[int] = None,
        acoustic_encoder_dim: Optional[int] = None,
        encoder_rates: List[int] = None,
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = None,
        n_codebooks: int = 8,
        n_semantic_codebooks: int = 1,
        codebook_size: int = 2048,
        codebook_dim: Union[int, list] = 8,
        acoustic_codebook_size: int = None,
        semantic_codebook_size: int = None,
        acoustic_codebook_dim: Union[int, list] = None,
        semantic_codebook_dim: int = None,
        quantizer_dropout: float = 1.0,
        semantic_quantizer: str = "rvq",
        semantic_quantizer_kwargs: dict = None,
        sample_rate: int = 44100,
        output_sample_rate: Optional[int] = None,
        use_shared_encoder: bool = False,
        semantic_subtraction: bool = False,
        dilation: List[int] = None,
        residual_kernel_size: int = 7,
        is_causal: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder_dim = encoder_dim
        self.semantic_encoder_dim = semantic_encoder_dim if semantic_encoder_dim is not None else encoder_dim
        self.acoustic_encoder_dim = acoustic_encoder_dim if acoustic_encoder_dim is not None else encoder_dim
        self.encoder_rates = encoder_rates or [2, 4, 8, 8]
        self.latent_dim = latent_dim
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates or [8, 8, 4, 2]
        self.n_codebooks = n_codebooks
        self.n_semantic_codebooks = n_semantic_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.acoustic_codebook_size = acoustic_codebook_size
        self.semantic_codebook_size = semantic_codebook_size
        self.acoustic_codebook_dim = acoustic_codebook_dim
        self.semantic_codebook_dim = semantic_codebook_dim
        self.quantizer_dropout = quantizer_dropout
        self.semantic_quantizer = semantic_quantizer
        self.semantic_quantizer_kwargs = semantic_quantizer_kwargs or {}
        self.sample_rate = sample_rate
        self.output_sample_rate = output_sample_rate if output_sample_rate is not None else sample_rate
        self.use_shared_encoder = use_shared_encoder
        self.semantic_subtraction = semantic_subtraction
        self.dilation = dilation or [1, 3, 9]
        self.residual_kernel_size = residual_kernel_size
        self.is_causal = is_causal


class LMSPTPreTrainedModel(PreTrainedModel):
    config_class = LMSPTConfig
    base_model_prefix = "lmspt"
    supports_gradient_checkpointing = True
    _supports_cache_class = True
    _supports_static_cache = True

    def __init__(self, config: LMSPTConfig):
        super().__init__(config)
        self.sample_rate = config.sample_rate
        self.output_sample_rate = getattr(config, "output_sample_rate", None) or config.sample_rate
        self.is_causal = getattr(config, "is_causal", False)

        from lmspt.model_codec.dac_model import DACEncoder, DACEncoderConfig

        latent_dim = config.latent_dim
        if latent_dim is None:
            latent_dim = config.encoder_dim * (2 ** len(config.encoder_rates))

        semantic_encoder_dim = getattr(config, "semantic_encoder_dim", None) or config.encoder_dim
        acoustic_encoder_dim = getattr(config, "acoustic_encoder_dim", None) or config.encoder_dim
        dilation = getattr(config, "dilation", None) or [1, 3, 9]
        residual_kernel_size = getattr(config, "residual_kernel_size", 7)

        semantic_encoder_config = DACEncoderConfig(
            d_model=semantic_encoder_dim,
            strides=config.encoder_rates,
            d_latent=latent_dim,
            is_causal=self.is_causal,
            dilation=dilation,
            residual_kernel_size=residual_kernel_size,
        )
        self.semantic_encoder = DACEncoder(semantic_encoder_config)

        if config.use_shared_encoder:
            self.acoustic_encoder = self.semantic_encoder
        else:
            acoustic_encoder_config = DACEncoderConfig(
                d_model=acoustic_encoder_dim,
                strides=config.encoder_rates,
                d_latent=latent_dim,
                is_causal=self.is_causal,
                dilation=dilation,
                residual_kernel_size=residual_kernel_size,
            )
            self.acoustic_encoder = DACEncoder(acoustic_encoder_config)

        self.decoder = Decoder(
            latent_dim,
            config.decoder_dim,
            config.decoder_rates,
            dilation=dilation,
            residual_kernel_size=residual_kernel_size,
            is_causal=self.is_causal,
        )

        self.encoder_rates = config.encoder_rates

        semantic_codebook_size = config.semantic_codebook_size or config.codebook_size
        semantic_codebook_dim = config.semantic_codebook_dim or config.codebook_dim
        acoustic_codebook_size = config.acoustic_codebook_size or config.codebook_size
        acoustic_codebook_dim = config.acoustic_codebook_dim or config.codebook_dim

        self.quantizer = DualResidualVectorQuantize(
            input_dim=latent_dim,
            n_codebooks=config.n_codebooks,
            n_semantic_codebooks=config.n_semantic_codebooks,
            semantic_codebook_size=semantic_codebook_size,
            semantic_codebook_dim=semantic_codebook_dim,
            acoustic_codebook_size=acoustic_codebook_size,
            acoustic_codebook_dim=acoustic_codebook_dim,
            semantic_quantizer=config.semantic_quantizer,
            quantizer_dropout=config.quantizer_dropout,
            semantic_quantizer_kwargs=config.semantic_quantizer_kwargs,
            semantic_subtraction=config.semantic_subtraction
        )

        self.gradient_checkpointing = False
        self.post_init()

    def reset_vocab_usage(self):
        self.quantizer.reset_vocab_usage()

    @torch.no_grad()
    def encode(self, audio_data):
        z_semantic = self.semantic_encoder(audio_data)
        z_acoustic = self.acoustic_encoder(audio_data)
        codes = self.quantizer.encode(z_semantic, z_acoustic)
        return codes

    @torch.no_grad()
    def decode(self, codes, acoustic_only=False):
        z_q = self.quantizer.decode(codes, acoustic_only=acoustic_only)
        x = self.decoder(z_q)
        return x

    def forward(self, audio_data: torch.Tensor, **kwargs):
        input_length = audio_data.size(-1)
        duration_sec = input_length / self.sample_rate
        output_length = int(duration_sec * self.output_sample_rate)

        if self.gradient_checkpointing and self.training:
            z_semantic = torch.utils.checkpoint.checkpoint(
                self.semantic_encoder, audio_data, use_reentrant=False
            )
            z_acoustic = torch.utils.checkpoint.checkpoint(
                self.acoustic_encoder, audio_data, use_reentrant=False
            )
        else:
            z_semantic = self.semantic_encoder(audio_data)
            z_acoustic = self.acoustic_encoder(audio_data)

        z_q, codes, vq_info, [z_q_semantic, _] = self.quantizer(z_semantic, z_acoustic)

        x = self.decoder(z_q)
        x = pad_to_length(x, output_length)

        semantic_edict = edict({
            "penalty": vq_info.semantic_commit_loss,
            "vq/codebook_loss": vq_info.semantic_codebook_loss,
            "metrics": {},
        })

        acoustic_edict = edict({
            "x": x,
            "codes": codes,
            "penalty": vq_info.acoustic_commit_loss,
            "vq/codebook_loss": vq_info.acoustic_codebook_loss,
            "metrics": vq_info.metrics,
        })

        return acoustic_edict, semantic_edict
