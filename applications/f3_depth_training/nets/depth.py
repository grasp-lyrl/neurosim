"""F3 feature field into a DepthAnythingV2 decoder, for relative disparity from events."""

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import Tensor

from .dav2 import MODEL_CONFIGS, DepthAnythingV2, MetricDepthAnythingV2, reset_emit
from .f3 import ENCODER_PREFIX, HEAD_PREFIXES, F3, load_f3_weights
from .hash_encoder import TemporalHashEncoder
from .memory import LatentMemory

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


def warm_start(model: "EventFFDepthAnythingV2", checkpoint: str | Path) -> bool:
    """Load another run's weights; the emit conv is reset across the relative/metric boundary."""
    # best/last carry optimizer and scheduler state too; mmap skips what is dropped.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    load_depth_weights(model, state["model"])
    # f3-era checkpoints have no config beside them, and predate metric heads.
    config = Path(checkpoint).parent / "depth_config.yml"
    metric = (
        config.exists()
        and yaml.safe_load(config.read_text())["dav2_config"].get("head") == "sigmoid"
    )
    reset = metric != (model.dav2.head == "sigmoid")
    if reset:
        reset_emit(model.dav2.depth_head)
    return reset


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

        encoder, head = dav2_config["encoder"], dav2_config.get("head", "relu")
        if head == "sigmoid":
            self.dav2 = MetricDepthAnythingV2(
                encoder, max_depth=dav2_config["max_depth"], **MODEL_CONFIGS[encoder]
            )
        else:
            self.dav2 = DepthAnythingV2(encoder, head=head, **MODEL_CONFIGS[encoder])
        if "ckpt" in dav2_config:
            self.dav2.load_state_dict(
                torch.load(dav2_config["ckpt"], map_location="cpu", weights_only=True)
            )
            logger.info("Loaded DepthAnythingV2 ckpt from %s", dav2_config["ckpt"])
            if self.dav2.head != "relu":
                reset_emit(self.dav2.depth_head)
        widen_patch_embed(self.dav2, self.eventff.feature_size)

    def load_eventff_weights(self, checkpoint: str | Path) -> None:
        """Load an f3 checkpoint; a `temporal_hash` run takes the conv stack only."""
        swapped = isinstance(self.eventff.multi_hash_encoder, TemporalHashEncoder)
        load_f3_weights(self.eventff, checkpoint, (ENCODER_PREFIX,) if swapped else ())
        logger.info(
            "Loaded F3 ckpt from %s%s",
            checkpoint,
            " (conv stack only; the age table starts random)" if swapped else "",
        )

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


class RecurrentEventFFDepthAnythingV2(EventFFDepthAnythingV2):
    """The same model with a ConvGRU carried on F3's stride-16 latent.

    The memory sits at the narrowest point, between `contract` and `expand`, so the state
    is 10x smaller than the raw field and F3's conv stack can run without a graph when the
    backbone is frozen. Parameter names outside `memory.` are unchanged, so a
    non-recurrent checkpoint warm-starts it:
    `load_depth_weights(model, state, fresh=("memory.",))`.

    The decoder's skips and the final hash-field fusion still come from the current tick,
    so only the coarse latent is remembered; fine detail is re-derived every tick.
    """

    def __init__(self, eventff_config: str, dav2_config: dict, retrain: bool = False):
        super().__init__(eventff_config, dav2_config, retrain)
        self.memory = LatentMemory(self.eventff.latent_channels)

    def initial_state(self, batch_size: int, device, dtype=torch.float32) -> Tensor:
        """Empty memory on the latent grid."""
        f3 = self.eventff
        stride = f3.latent_stride
        return torch.zeros(
            batch_size,
            f3.latent_channels,
            f3.w // stride,
            f3.h // stride,
            device=device,
            dtype=dtype,
        )

    def remember(
        self, events: Tensor, counts: Tensor, state: Tensor
    ) -> tuple[Tensor, Tensor]:
        """One tick: events [N, 4] and memory -> feature field [B, C, H, W], new memory."""
        field = self.eventff.feature_field(events[:, :3], counts)
        latent, skips = self.eventff.contract(field)
        state = self.memory(latent, state)
        return self.eventff.expand(state, skips, field).permute(0, 1, 3, 2), state

    def forward(
        self, events: Tensor, counts: Tensor, cparams: Tensor, state: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """One training step: disparity [B, h, w], the cropped field, and the new memory."""
        field, state = self.remember(events, counts, state)
        field = batch_cropper(field, cparams)
        return self.decode(field, (self.size, self.size)), field, state

    @torch.no_grad()
    def infer_image(
        self, events: Tensor, counts: Tensor, cparams: Tensor, state: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Whole-frame path with memory: aspect preserved, no square crop, one sample."""
        h, w = int(cparams[2] - cparams[0]), int(cparams[3] - cparams[1])
        field, state = self.remember(events, counts, state)
        field = field[0, :, cparams[0] : cparams[2], cparams[1] : cparams[3]]
        pred = self.decode(field.unsqueeze(0), get_resize_shapes(h, w, self.size, 14))
        return pred[0], field, state
