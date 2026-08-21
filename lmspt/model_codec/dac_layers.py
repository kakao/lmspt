import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class CausalWNConv1d(nn.Module):
    """Weight-normalized Conv1d with optional causal padding.

    Mirrors transformers.models.mimi.modeling_mimi.MimiConv1d's strategy:
    nn.Conv1d itself uses padding=0, and padding is applied externally in
    forward(). In causal mode all of `padding_total = (k_eff - stride)` is
    placed on the LEFT (past), so the kernel never reads future frames.
    Extra right padding is added only when needed to make the sequence
    length a multiple of stride; this trailing padding is masked out
    naturally because it sits beyond the last valid output frame.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        is_causal: bool = False,
        pad_mode: str = "constant",
    ):
        super().__init__()
        self.is_causal = is_causal
        self.pad_mode = pad_mode

        self.conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding=0,
            )
        )

        # Effective kernel accounts for dilation
        kernel_eff = (kernel_size - 1) * dilation + 1
        self._stride = stride
        self._kernel_eff = kernel_eff
        self._padding_total = kernel_eff - stride

        # Symmetric split for non-causal mode (asymmetric for odd totals)
        self._padding_right = self._padding_total // 2
        self._padding_left = self._padding_total - self._padding_right

    def _get_extra_padding(self, x: torch.Tensor) -> int:
        """Compute extra right padding so output length matches stride."""
        length = x.shape[-1]
        n_frames = (length - self._kernel_eff + self._padding_total) / self._stride + 1
        n_frames = math.ceil(n_frames) - 1
        ideal_length = n_frames * self._stride + self._kernel_eff - self._padding_total
        return max(0, ideal_length - length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        extra = self._get_extra_padding(x)
        if self.is_causal:
            x = F.pad(x, (self._padding_total, extra), mode=self.pad_mode)
        else:
            x = F.pad(
                x,
                (self._padding_left, self._padding_right + extra),
                mode=self.pad_mode,
            )
        return self.conv(x)


class CausalWNConvTranspose1d(nn.Module):
    """Weight-normalized ConvTranspose1d with optional causal trimming.

    Mirrors transformers.models.mimi.modeling_mimi.MimiConvTranspose1d:
    nn.ConvTranspose1d uses padding=0, and the raw output is trimmed
    externally. In causal mode (with trim_right_ratio=1.0), all
    `padding_total = kernel_size - stride` extra positions are removed
    from the RIGHT, so each output time step depends only on input
    positions <= itself.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
        is_causal: bool = False,
        trim_right_ratio: float = 1.0,
    ):
        super().__init__()
        self.is_causal = is_causal

        if not is_causal and trim_right_ratio != 1.0:
            raise ValueError(
                "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
            )

        self.conv = weight_norm(
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                groups=groups,
                bias=bias,
                padding=0,
            )
        )

        padding_total = kernel_size - stride
        if is_causal:
            # Trim everything from the right (default ratio = 1.0)
            self._padding_right = math.ceil(padding_total * trim_right_ratio)
        else:
            # Symmetric trim
            self._padding_right = padding_total // 2
        self._padding_left = padding_total - self._padding_right

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        end = x.shape[-1] - self._padding_right
        return x[..., self._padding_left:end]


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


class Snake1d_(Snake1d):
    def forward(self, x):
        shape = x.shape
        x = x.reshape(shape[0], shape[1], -1)
        x = x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2)
        x = x.reshape(shape)
        return x
