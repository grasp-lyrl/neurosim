"""F3 feature field into a DepthAnythingV2 decoder, for relative disparity from events."""

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import Tensor

from .dav2 import MODEL_CONFIGS, DepthAnythingV2
from .f3 import HEAD_PREFIXES, F3, load_f3_weights

logger = logging.getLogger(__name__)


def get_resize_shapes(
    h: int, w: int, smaller_edge: int, multiple: int
) -> tuple[int, int]:
    """Resize h to `smaller_edge` keeping aspect, rounding w up to a `multiple`."""
    return smaller_edge, math.ceil(int(w / h * smaller_edge) / multiple) * multiple


def batch_cropper(images: Tensor, cparams: Tensor) -> Tensor:
    """Crop each image of [B, C, H, W] by its own [y0, x0, y1, x1] row of `cparams`."""
    return torch.stack(
        [
            images[i, :, cparams[i, 0] : cparams[i, 2], cparams[i, 1] : cparams[i, 3]]
            for i in range(images.shape[0])
        ]
    )


def load_depth_weights(
    model: "EventFFDepthAnythingV2", state: dict, fresh: tuple[str, ...] = ()
) -> None:
    """Load a trainer checkpoint, before compiling: keys carry `_orig_mod.` once compiled.

    Args:
        state: checkpoint weights. f3-era ones also carry `eventff.pred.*`, the
            event-prediction head this port never builds; checkpoints written here do not.
        fresh: key prefixes the checkpoint is allowed not to cover, for modules this run
            initialises itself (the recurrent model warm-starting off a non-recurrent run).
    """
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    unfilled = [k for k in missing if not k.startswith(fresh)]
    assert not unfilled, f"checkpoint has no weights for {unfilled}"
    head = tuple(f"eventff.{p}" for p in HEAD_PREFIXES)
    stray = [k for k in unexpected if not k.startswith(head)]
    assert not stray, f"checkpoint keys match no module: {stray}"


def widen_patch_embed(dav2: DepthAnythingV2, in_chans: int) -> None:
    """Re-tile the RGB patch embedding over `in_chans` feature-field channels."""
    proj = dav2.pretrained.patch_embed.proj
    weight, bias = proj.weight, proj.bias
    widened = nn.Conv2d(
        in_chans,
        proj.out_channels,
        kernel_size=proj.kernel_size,
        stride=proj.stride,
        padding=proj.padding,
    )
    widened.weight.data = torch.cat(
        [weight.repeat(1, in_chans // 3, 1, 1), weight[:, : in_chans % 3]], dim=1
    )
    widened.bias.data = bias
    dav2.pretrained.patch_embed.proj = widened


class EventFFDepthAnythingV2(nn.Module):
    """Events -> F3 feature field -> DepthAnythingV2 -> relative disparity."""

    def __init__(self, eventff_config: str, dav2_config: dict, retrain: bool = False):
        super().__init__()
        self.eventff_config = eventff_config
        self.dav2_config = dav2_config
        self.size = dav2_config["size"]

        self.eventff = F3.from_config(eventff_config)
        if not retrain:
            self.eventff.requires_grad_(False)

        encoder = dav2_config["encoder"]
        self.dav2 = DepthAnythingV2(encoder=encoder, **MODEL_CONFIGS[encoder])
        if "ckpt" in dav2_config:
            self.dav2.load_state_dict(
                torch.load(dav2_config["ckpt"], map_location="cpu", weights_only=True)
            )
            logger.info("Loaded DepthAnythingV2 ckpt from %s", dav2_config["ckpt"])
        widen_patch_embed(self.dav2, self.eventff.feature_size)

    def load_eventff_weights(self, checkpoint: str | Path) -> None:
        load_f3_weights(self.eventff, checkpoint)
        logger.info("Loaded F3 ckpt from %s", checkpoint)

    def save_configs(self, path: str) -> None:
        with open(f"{path}/depth_config.yml", "w") as f:
            yaml.dump(
                {
                    "model": "EventFFDepthAnythingV2",
                    "dav2_config": self.dav2_config,
                    "eventff_config": self.eventff_config,
                },
                f,
                default_flow_style=None,
            )

    @classmethod
    def init_from_config(
        cls, path: str, retrain: bool = False
    ) -> "EventFFDepthAnythingV2":
        conf = yaml.safe_load(Path(path).read_text())
        return cls(conf["eventff_config"], conf["dav2_config"], retrain)

    def field(self, events: Tensor, counts: Tensor) -> Tensor:
        """Events [N, 4] with counts [B] -> feature field [B, C, H, W]."""
        return self.eventff(events[:, :3], counts).permute(0, 1, 3, 2)

    def decode(self, field: Tensor, size: tuple[int, int]) -> Tensor:
        """Feature field [B, C, h, w] -> disparity [B, h, w], DAv2 run at `size`."""
        resized = F.interpolate(field, size, mode="bilinear", align_corners=False)
        pred = self.dav2(resized).unsqueeze(1)
        return F.interpolate(
            pred, field.shape[2:], mode="bilinear", align_corners=True
        ).squeeze(1)

    def forward(
        self, events: Tensor, counts: Tensor, cparams: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Square-cropped training path: disparity [B, h, w] and the cropped field."""
        field = batch_cropper(self.field(events, counts), cparams)
        return self.decode(field, (self.size, self.size)), field

    @torch.no_grad()
    def infer_image(
        self, events: Tensor, counts: Tensor, cparams: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Whole-frame path: aspect preserved, no square crop. One sample at a time."""
        h, w = int(cparams[2] - cparams[0]), int(cparams[3] - cparams[1])
        field = self.field(events, counts)
        field = field[0, :, cparams[0] : cparams[2], cparams[1] : cparams[3]]
        pred = self.decode(field.unsqueeze(0), get_resize_shapes(h, w, self.size, 14))
        return pred[0], field
