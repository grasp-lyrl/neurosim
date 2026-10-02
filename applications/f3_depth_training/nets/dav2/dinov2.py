"""DINOv2 vision transformer, after facebookresearch/dinov2 `vision_transformer.py`.

Only the intermediate-layer path DepthAnythingV2 uses is kept: no masking, no register
tokens, no block chunking, no classification head.
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .layers import Block, PatchEmbed

# depth, embed_dim, num_heads per DINOv2 size.
VIT_SIZES = {"vits": (12, 384, 6), "vitb": (12, 768, 12), "vitl": (24, 1024, 16)}


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        init_values: float = 1.0,
        interpolate_offset: float = 0.1,
    ):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.interpolate_offset = interpolate_offset

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)
        )
        self.blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                init_values=init_values,
                norm_layer=norm_layer,
            )
            for _ in range(depth)
        )
        self.norm = norm_layer(embed_dim)
        # Unused here, but present in the released checkpoint, so it must exist to load.
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)

    def interpolate_pos_encoding(self, x: Tensor, h: int, w: int) -> Tensor:
        """Position embeddings resampled from the pretrained square grid to h/14 by w/14."""
        n_patches = self.pos_embed.shape[1] - 1
        if x.shape[1] - 1 == n_patches and h == w:
            return self.pos_embed

        pos_embed = self.pos_embed.float()
        # The offset avoids a floating point error in the interpolation:
        # https://github.com/facebookresearch/dino/issues/8
        rows = h // self.patch_size + self.interpolate_offset
        cols = w // self.patch_size + self.interpolate_offset
        side = math.sqrt(n_patches)
        patch_pos_embed = F.interpolate(
            pos_embed[:, 1:]
            .reshape(1, int(side), int(side), self.embed_dim)
            .permute(0, 3, 1, 2),
            scale_factor=(rows / side, cols / side),
            mode="bicubic",
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(
            1, -1, self.embed_dim
        )
        out = torch.cat([pos_embed[:, :1], patch_pos_embed], dim=1)
        return out.to(x.dtype)

    def prepare_tokens(self, x: Tensor) -> Tensor:
        """Image [B, C, H, W] -> class token prepended and position embedded [B, 1+N, D]."""
        h, w = x.shape[2:]
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        return x + self.interpolate_pos_encoding(x, h, w)

    def get_intermediate_layers(
        self, x: Tensor, blocks_to_take: list[int]
    ) -> tuple[tuple[Tensor, Tensor], ...]:
        """Normed (patch tokens, class token) after each block in `blocks_to_take`."""
        x = self.prepare_tokens(x)
        out = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in blocks_to_take:
                normed = self.norm(x)
                out.append((normed[:, 1:], normed[:, 0]))
        return tuple(out)


def DINOv2(model_name: str) -> DinoVisionTransformer:
    depth, embed_dim, num_heads = VIT_SIZES[model_name]
    return DinoVisionTransformer(embed_dim=embed_dim, depth=depth, num_heads=num_heads)
