"""ConvNeXt-V2 building blocks and the pixel-shuffle upsampler used by F3."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LayerNorm(nn.Module):
    """LayerNorm over channels_last [B, H, W, C] or channels_first [B, C, H, W]."""

    def __init__(self, dim: int, eps: float = 1e-6, data_format: str = "channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (dim,)

    def forward(self, x: Tensor) -> Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class GRN(nn.Module):
    """Global response normalization over channels_last [B, H, W, C]."""

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: Tensor) -> Tensor:
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class Block(nn.Module):
    """ConvNeXt-V2 block: depthwise conv, norm, inverted bottleneck with GRN, residual."""

    def __init__(
        self, dim: int, kernel_size: int = 7, bottleneck: int = 4, dilation: int = 1
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2 * dilation,
            dilation=dilation,
            groups=dim,
        )
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, bottleneck * dim)
        self.act = nn.GELU()
        self.grn = GRN(bottleneck * dim)
        self.pwconv2 = nn.Linear(bottleneck * dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        h = self.dwconv(x).permute(0, 2, 3, 1)
        h = self.pwconv2(self.grn(self.act(self.pwconv1(self.norm(h)))))
        return x + h.permute(0, 3, 1, 2)


class PixelShuffleUpsample(nn.Module):
    """Upsample by `upscale_factor` via a channel-expanding conv and a pixel shuffle."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upscale_factor: int = 2,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * upscale_factor**2,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x: Tensor) -> Tensor:
        return self.pixel_shuffle(self.conv(x))
