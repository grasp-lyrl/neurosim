"""DINOv2 transformer layers, after facebookresearch/dinov2 `layers/`.

Trimmed to the path DepthAnythingV2 takes: no xformers, no nested tensors, no stochastic
depth, no SwiGLU, no dropout. Parameter names are unchanged so released checkpoints load.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class PatchEmbed(nn.Module):
    """Image [B, C, H, W] -> patch tokens [B, N, D]."""

    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class Attention(nn.Module):
    """Multi-head self-attention. Uses SDPA where the reference dispatches to xformers."""

    def __init__(
        self, dim: int, num_heads: int, qkv_bias: bool = True, proj_bias: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        proj_bias: bool,
        ffn_bias: bool,
        init_values: float,
        norm_layer,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, proj_bias)
        self.ls1 = LayerScale(dim, init_values)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), bias=ffn_bias)
        self.ls2 = LayerScale(dim, init_values)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))
