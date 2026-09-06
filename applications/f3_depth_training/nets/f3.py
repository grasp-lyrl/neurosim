"""F3 event feature field: hash-encode events, scatter, encode, upsample back to full res.

In-house copy of `f3.event_FF.EventPatchFF` restricted to the `use_upsampling` feature
path the depth model uses. The event-prediction head (`pred`) is not built, so its
checkpoint keys are dropped on load.
"""

from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch import Tensor

from .blocks import Block, LayerNorm, PixelShuffleUpsample
from .hash_encoder import MultiResolutionHashEncoder

# The event-prediction head. Present in the released checkpoint, never built here.
HEAD_PREFIXES = ("pred.",)


def batch_index(counts: Tensor, n: int) -> Tensor:
    """Which batch element each of `n` concatenated events belongs to.

    `repeat_interleave(counts)` says the same thing, but its output size depends on the
    *values* in counts, which dynamo cannot know: it splits the compiled graph in two. Here
    the size is n, which the tracer already has.
    """
    return torch.searchsorted(
        counts.cumsum(0), torch.arange(n, device=counts.device), right=True
    )


def _fuse_skip(dim: int) -> nn.Sequential:
    """Merge an upsampled stage with its skip: 1x1 down to `dim`, then a depthwise conv."""
    return nn.Sequential(
        nn.GELU(),
        nn.Conv2d(2 * dim, dim, kernel_size=1),
        LayerNorm(dim, data_format="channels_first"),
        nn.GELU(),
        nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
        LayerNorm(dim, data_format="channels_first"),
    )


class F3(nn.Module):
    """Event frontend producing a full-resolution feature field.

    events   [N, 3]                (x, y, t) in [0,1], batch elements concatenated
       │ hash encoder, per event, no grid yet
    encoded  [N, L*F]
       │ scatter onto the sensor canvas, accumulate
    field    [B, L*F, W, H]        W before H: the field is indexed x-major
       │ downsample_layers[i] then stages[i], strides 4, 2, 2
    tokens   [B, dims[-1], W/16, H/16]
       │ upsample_layers[i] then upsample_process[i], with skips
    feature  [B, upsampling_dims, W, H]
    """

    def __init__(
        self,
        hash_encoder: MultiResolutionHashEncoder,
        frame_sizes: list[int],
        dims: list[int],
        convkernels: list[int],
        convdepths: list[int],
        convbtlncks: list[int],
        convdilations: list[int],
        dskernels: list[int],
        dsstrides: list[int],
        patch_size: int,
        upsampling_dims: int,
    ):
        super().__init__()
        # frame_sizes is [W, H, time_ctx / bucket]. The model never reads the third entry: it
        # is the divisor the loader normalizes event age by.
        self.frame_sizes = frame_sizes
        self.w, self.h = frame_sizes[:2]
        self.feature_size = upsampling_dims
        self.multi_hash_encoder = hash_encoder

        in_channels = hash_encoder.levels * hash_encoder.feature_size
        self.downsample_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        dims[0],
                        kernel_size=dskernels[0],
                        stride=dsstrides[0],
                        padding=(dskernels[0] - 1) // 2,
                    ),
                    LayerNorm(dims[0], data_format="channels_first"),
                )
            ]
            + [
                nn.Sequential(
                    LayerNorm(dims[i - 1], data_format="channels_first"),
                    nn.Conv2d(
                        dims[i - 1],
                        dims[i],
                        kernel_size=dskernels[i],
                        stride=dsstrides[i],
                        padding=(dskernels[i] - 1) // 2,
                    ),
                )
                for i in range(1, len(dims))
            ]
        )
        self.stages = nn.ModuleList(
            nn.Sequential(
                *[
                    Block(dims[i], convkernels[i], convbtlncks[i], convdilations[i])
                    for _ in range(convdepths[i])
                ]
            )
            for i in range(len(dims))
        )

        self.upsample_layers = nn.ModuleList(
            [
                PixelShuffleUpsample(dims[i], dims[i - 1], upscale_factor=dsstrides[i])
                for i in range(len(dims) - 1, 0, -1)
            ]
            + [
                PixelShuffleUpsample(
                    dims[0], upsampling_dims, upscale_factor=dsstrides[0] // patch_size
                )
            ]
        )
        self.upsample_process = nn.ModuleList(
            _fuse_skip(dims[i - 1]) for i in range(len(dims) - 1, 0, -1)
        )
        self.upsample_process.append(_fuse_skip(upsampling_dims))
        self.downsample_hash_to_patchsize = nn.Conv2d(
            in_channels,
            upsampling_dims,
            kernel_size=patch_size,
            stride=patch_size,
            padding=(patch_size - 1) // 2,
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Conv2d | nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    @classmethod
    def from_config(cls, config: str | Path) -> "F3":
        """Build from F3's own yaml."""
        conf = yaml.safe_load(Path(config).read_text())
        assert conf["use_upsampling"], "the depth model reads F3's upsampled field"
        return cls(
            hash_encoder=MultiResolutionHashEncoder(**conf["multi_hash_encoder"]),
            frame_sizes=conf["frame_sizes"],
            dims=conf["dims"],
            convkernels=conf["convkernels"],
            convdepths=conf["convdepths"],
            convbtlncks=conf["convbtlncks"],
            convdilations=conf["convdilations"],
            dskernels=conf["dskernels"],
            dsstrides=conf["dsstrides"],
            patch_size=conf["patch_size"],
            upsampling_dims=conf["upsampling_dims"],
        )

    def feature_field(self, events: Tensor, counts: Tensor) -> Tensor:
        """Events [N, 3] with per-sample counts [B] -> accumulated hash field [B, L*F, W, H]."""
        px = (events[:, 0] * self.w).round().int()
        py = (events[:, 1] * self.h).round().int()
        encoded = self.multi_hash_encoder(events.unsqueeze(0)).squeeze(0)

        field = torch.zeros(
            (counts.shape[0], self.w, self.h, encoded.shape[-1]),
            device=encoded.device,
            dtype=encoded.dtype,
        )
        batch = batch_index(counts, encoded.shape[0]).int()
        field.index_put_((batch, px, py), encoded, accumulate=True)
        return field.permute(0, 3, 1, 2)

    def encode(self, field: Tensor) -> Tensor:
        """Hash field [B, L*F, W, H] -> feature field [B, upsampling_dims, W, H]."""
        skips = [field]
        x = field
        for i, (downsample, stage) in enumerate(
            zip(self.downsample_layers, self.stages, strict=True)
        ):
            x = stage(downsample(x))
            if i < len(self.stages) - 1:
                skips.append(x)

        for i in range(len(self.upsample_layers) - 1):
            x = self.upsample_layers[i](x)
            x = self.upsample_process[i](torch.cat([skips[-(i + 1)], x], dim=1))
        x = self.upsample_layers[-1](x)
        x = torch.cat([x, self.downsample_hash_to_patchsize(field)], dim=1)
        return self.upsample_process[-1](x)

    def forward(self, events: Tensor, counts: Tensor) -> Tensor:
        """Events [N, 3] with per-sample counts [B] -> feature field [B, C, W, H]."""
        return self.encode(self.feature_field(events, counts))


def load_f3_weights(model: F3, checkpoint: str | Path) -> dict:
    """Load released F3 weights, dropping the event-prediction head. Returns its metadata."""
    ckpt = torch.load(checkpoint, weights_only=True, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    assert not missing, f"checkpoint has no weights for {missing}"
    stray = [k for k in unexpected if not k.startswith(HEAD_PREFIXES)]
    assert not stray, f"checkpoint keys match no module and are not head keys: {stray}"
    return {k: ckpt[k] for k in ("epoch", "loss", "acc") if k in ckpt}
