"""
From DAC: https://github.com/descriptinc/descript-audio-codec/blob/main/dac/model
"""
import math
from typing import List
from typing import Union

import numpy as np
import torch
from torch import nn

from .dac_layers import Snake1d, Snake1d_
from .dac_layers import WNConv1d
from .dac_layers import WNConvTranspose1d
from .dac_layers import CausalWNConv1d, CausalWNConvTranspose1d
import torch.nn.functional as F
from .cnn import ConvNeXtBlock


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


def pad_to_length(x, length, pad_value=0):
    # Get the current size along the last dimension
    current_length = x.shape[-1]

    # If the length is greater than current_length, we need to pad
    if length > current_length:
        pad_amount = length - current_length
        # Pad on the last dimension (right side), keeping all other dimensions the same
        x_padded = F.pad(x, (0, pad_amount), value=pad_value)
    else:
        # If no padding is required, simply slice the tensor
        x_padded = x[..., :length]

    return x_padded


class ResidualUnit(nn.Module):
    def __init__(
        self,
        dim: int = 16,
        dilation: int = 1,
        kernel_size: int = 7,
        use_jit: bool = True,
        is_causal: bool = False,
    ):
        super().__init__()
        self.is_causal = is_causal
        if is_causal:
            conv1 = CausalWNConv1d(
                dim, dim, kernel_size=kernel_size, dilation=dilation, is_causal=True
            )
            conv2 = CausalWNConv1d(dim, dim, kernel_size=1, is_causal=True)
        else:
            pad = ((kernel_size - 1) * dilation) // 2
            conv1 = WNConv1d(
                dim, dim, kernel_size=kernel_size, dilation=dilation, padding=pad
            )
            conv2 = WNConv1d(dim, dim, kernel_size=1)
        self.block = nn.Sequential(
            Snake1d(dim) if use_jit else Snake1d_(dim),
            conv1,
            Snake1d(dim) if use_jit else Snake1d_(dim),
            conv2,
        )

    def forward(self, x):
        y = self.block(x)
        diff = x.shape[-1] - y.shape[-1]
        if diff > 0:
            if self.is_causal:
                # Causal: drop oldest (leftmost) frames so y[t] aligns with x[t]
                x = x[..., diff:]
            else:
                pad = diff // 2
                x = x[..., pad:-pad] if pad > 0 else x
        return x + y


class EncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int = 16,
        stride: int = 1,
        use_jit: bool = True,
        dilation: list=[1, 3, 9],
        residual_kernel_size: int = 7,
        is_causal: bool = False,
    ):
        super().__init__()
        if is_causal:
            downsample = CausalWNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                is_causal=True,
            )
        else:
            downsample = WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            )
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=dilation[0], use_jit=use_jit, kernel_size=residual_kernel_size, is_causal=is_causal),
            ResidualUnit(dim // 2, dilation=dilation[1], use_jit=use_jit, kernel_size=residual_kernel_size, is_causal=is_causal),
            ResidualUnit(dim // 2, dilation=dilation[2], use_jit=use_jit, kernel_size=residual_kernel_size, is_causal=is_causal),
            Snake1d(dim // 2) if use_jit else Snake1d_(dim // 2),
            downsample,
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
        dilation: list = [1, 3, 9],
        residual_kernel_size: int = 7,
        is_causal: bool = False,
    ):
        super().__init__()
        # Create first convolution
        if is_causal:
            first_conv = CausalWNConv1d(1, d_model, kernel_size=7, is_causal=True)
        else:
            first_conv = WNConv1d(1, d_model, kernel_size=7, padding=3)
        self.block = [first_conv]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [
                EncoderBlock(
                    d_model,
                    stride=stride,
                    dilation=dilation,
                    residual_kernel_size=residual_kernel_size,
                    is_causal=is_causal,
                )
            ]

        # Create last convolution
        if is_causal:
            last_conv = CausalWNConv1d(d_model, d_latent, kernel_size=3, is_causal=True)
        else:
            last_conv = WNConv1d(d_model, d_latent, kernel_size=3, padding=1)
        self.block += [
            Snake1d(d_model),
            last_conv,
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


from transformers import PreTrainedModel, PretrainedConfig
class DACEncoderConfig(PretrainedConfig):
    model_type = "dac_encoder"

    def __init__(
        self,
        d_model: int = 64,
        strides: List[int] = None,
        d_latent: int = 64,
        is_causal: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.strides = strides or [2, 4, 8, 8]
        self.dilation = [1, 3, 9]
        self.d_latent = d_latent
        self.is_causal = is_causal

class DACEncoder(PreTrainedModel):
    config_class = DACEncoderConfig
    supports_gradient_checkpointing = True
    base_model_prefix = "encoder"
    _supports_cache_class = True
    _supports_static_cache = True
    def __init__(
        self,
        config: DACEncoderConfig
    ):
        super().__init__(config)
        is_causal = getattr(config, "is_causal", False)
        # Create first convolution
        if is_causal:
            first_conv = CausalWNConv1d(1, config.d_model, kernel_size=7, is_causal=True)
        else:
            first_conv = WNConv1d(1, config.d_model, kernel_size=7, padding=3)
        self.block = [first_conv]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        d_model = config.d_model
        for stride in config.strides:
            d_model *= 2
            self.block += [
                EncoderBlock(
                    d_model,
                    stride=stride,
                    dilation=config.dilation,
                    use_jit=False,
                    is_causal=is_causal,
                )
            ]

        # Create last convolution
        if is_causal:
            last_conv = CausalWNConv1d(d_model, config.d_latent, kernel_size=3, is_causal=True)
        else:
            last_conv = WNConv1d(d_model, config.d_latent, kernel_size=3, padding=1)
        self.block += [
            Snake1d_(d_model),
            last_conv,
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

        self.gradient_checkpointing = False
        self.post_init()

    def forward(self, x):
        if self.gradient_checkpointing and self.training:
            return self._gradient_checkpointing_func(
                self.block.__call__, x,
            )
        else:
            return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        stride: int = 1,
        dilation: list = [1, 3, 9],
        residual_kernel_size: int = 7,
        is_causal: bool = False,
    ):
        super().__init__()
        if is_causal:
            upsample = CausalWNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                is_causal=True,
            )
        else:
            upsample = WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            )
        self.block = nn.Sequential(
            Snake1d(input_dim),
            upsample,
            ResidualUnit(output_dim, dilation=dilation[0], kernel_size=residual_kernel_size, is_causal=is_causal),
            ResidualUnit(output_dim, dilation=dilation[1], kernel_size=residual_kernel_size, is_causal=is_causal),
            ResidualUnit(output_dim, dilation=dilation[2], kernel_size=residual_kernel_size, is_causal=is_causal),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
        sample_rate: int = None,
        dilation: list = [1, 3, 9],
        residual_kernel_size: int = 7,
        is_causal: bool = False,
    ):
        super().__init__()
        self.sample_rate = sample_rate

        # Add first conv layer
        if is_causal:
            first_conv = CausalWNConv1d(input_channel, channels, kernel_size=7, is_causal=True)
        else:
            first_conv = WNConv1d(input_channel, channels, kernel_size=7, padding=3)
        layers = [first_conv]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [
                DecoderBlock(
                    input_dim,
                    output_dim,
                    stride,
                    dilation=dilation,
                    residual_kernel_size=residual_kernel_size,
                    is_causal=is_causal,
                )
            ]

        # Add final conv layer
        if is_causal:
            final_conv = CausalWNConv1d(output_dim, d_out, kernel_size=7, is_causal=True)
        else:
            final_conv = WNConv1d(output_dim, d_out, kernel_size=7, padding=3)
        layers += [
            Snake1d(output_dim),
            final_conv,
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


