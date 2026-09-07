"""DepthAnythingV2: a DINOv2 encoder with a DPT decoder.

After DepthAnything/Depth-Anything-V2 `depth_anything_v2/dpt.py`. The image-loading path
(`infer_image`, `image2tensor`) is dropped: this model consumes feature fields, not images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .dinov2 import DINOv2

# features and per-stage projection widths for each encoder size.
MODEL_CONFIGS = {
    "vits": {"features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"features": 256, "out_channels": [256, 512, 1024, 1024]},
}

# Which encoder blocks the DPT head taps, per encoder size.
INTERMEDIATE_LAYERS = {
    "vits": [2, 5, 8, 11],
    "vitb": [2, 5, 8, 11],
    "vitl": [4, 11, 17, 23],
}


class ResidualConvUnit(nn.Module):
    """Two 3x3 convs on a pre-activated input, added back to it."""

    def __init__(self, features: int, activation: nn.Module):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(self.activation(x))
        return self.conv2(self.activation(out)) + x


class FeatureFusionBlock(nn.Module):
    """Add the skip path, refine, upsample, and project."""

    def __init__(
        self, features: int, activation: nn.Module, align_corners: bool = True
    ):
        super().__init__()
        self.align_corners = align_corners
        self.out_conv = nn.Conv2d(features, features, kernel_size=1)
        self.resConfUnit1 = ResidualConvUnit(features, activation)
        self.resConfUnit2 = ResidualConvUnit(features, activation)

    def forward(self, *xs: Tensor, size: tuple[int, int] | None = None) -> Tensor:
        output = xs[0]
        if len(xs) == 2:
            output = output + self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)

        modifier = {"scale_factor": 2} if size is None else {"size": size}
        output = F.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )
        return self.out_conv(output)


def _make_scratch(in_shape: list[int], out_shape: int) -> nn.Module:
    """The four 3x3 projections from encoder widths onto the fusion width."""
    scratch = nn.Module()
    for i, in_channels in enumerate(in_shape):
        conv = nn.Conv2d(in_channels, out_shape, kernel_size=3, padding=1, bias=False)
        setattr(scratch, f"layer{i + 1}_rn", conv)
    return scratch


class DPTHead(nn.Module):
    """Four encoder taps -> one disparity map at 14x the token grid.

    `head` is `relu` for DAv2's own parameterisation, or `exp` for DA3's: the last conv
    then emits a log disparity, so the activation stays O(1) whatever the range.
    """

    def __init__(
        self,
        in_channels: int,
        features: int,
        out_channels: list[int],
        head: str = "relu",
    ):
        super().__init__()
        self.projects = nn.ModuleList(
            nn.Conv2d(in_channels, out_channel, kernel_size=1)
            for out_channel in out_channels
        )
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    out_channels[0], out_channels[0], kernel_size=4, stride=4
                ),
                nn.ConvTranspose2d(
                    out_channels[1], out_channels[1], kernel_size=2, stride=2
                ),
                nn.Identity(),
                nn.Conv2d(
                    out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1
                ),
            ]
        )

        self.scratch = _make_scratch(out_channels, features)
        for i in range(1, 5):
            setattr(
                self.scratch,
                f"refinenet{i}",
                FeatureFusionBlock(features, nn.ReLU(False)),
            )

        self.scratch.output_conv1 = nn.Conv2d(
            features, features // 2, kernel_size=3, padding=1
        )
        # The trailing ReLU would floor a log disparity at exp(0) = 1, so `exp` drops it.
        # Both keep the weights at index 0 and 2, so the state dict is the same either way.
        # Construction order is load-bearing: it fixes which weights each conv draws.
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1),
            *([nn.ReLU(True), nn.Identity()] if head == "relu" else []),
        )
        if head == "exp":
            reset_log_emit(self)

    def forward(self, out_features: tuple, patch_h: int, patch_w: int) -> Tensor:
        out = []
        for i, (patch_tokens, _) in enumerate(out_features):
            x = patch_tokens.permute(0, 2, 1).unflatten(2, (patch_h, patch_w))
            out.append(self.resize_layers[i](self.projects[i](x)))

        layer_1_rn = self.scratch.layer1_rn(out[0])
        layer_2_rn = self.scratch.layer2_rn(out[1])
        layer_3_rn = self.scratch.layer3_rn(out[2])
        layer_4_rn = self.scratch.layer4_rn(out[3])

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(
            out, (patch_h * 14, patch_w * 14), mode="bilinear", align_corners=True
        )
        return self.scratch.output_conv2(out)


def reset_log_emit(depth_head: "DPTHead") -> None:
    """Small weights on the emit conv, so a log-disparity head starts near exp(0) = 1.

    DAv2's pretrained values give disparity directly; exponentiating them overflows.
    Call this again after loading a DAv2 checkpoint, which restores them.
    """
    emit = depth_head.scratch.output_conv2[2]
    nn.init.trunc_normal_(emit.weight, std=0.02)
    nn.init.zeros_(emit.bias)


class DepthAnythingV2(nn.Module):
    """Feature field [B, C, H, W] -> relative disparity [B, H, W]."""

    def __init__(
        self,
        encoder: str = "vitl",
        features: int = 256,
        out_channels=None,
        head: str = "relu",
    ):
        super().__init__()
        assert head in ("relu", "exp"), f"unknown head {head!r}"
        self.encoder = encoder
        self.head = head
        self.pretrained = DINOv2(encoder)
        self.depth_head = DPTHead(
            self.pretrained.embed_dim, features, out_channels, head
        )

    def forward(self, x: Tensor) -> Tensor:
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        features = self.pretrained.get_intermediate_layers(
            x, INTERMEDIATE_LAYERS[self.encoder]
        )
        out = self.depth_head(features, patch_h, patch_w)
        return (torch.exp(out) if self.head == "exp" else F.relu(out)).squeeze(1)
