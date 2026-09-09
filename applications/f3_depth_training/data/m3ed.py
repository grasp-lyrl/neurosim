"""M3ED sequences as a fixed real-data validation set: recorded events, LiDAR disparity.

The simulator's validation redraws scenes every time, so its loss moves with the sample as
much as with the model. These frames never change, and they are the sensor we deploy on.
"""

import logging
from pathlib import Path
from typing import Callable, NamedTuple

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ..utils import eval_relative_depth

logger = logging.getLogger(__name__)

# M3ED's left event camera, and the frame its LiDAR depth is already projected into.
SENSOR_W, SENSOR_H = 1280, 720
EVENTS, DEPTH = "prophesee/left", "depth/prophesee/left"


class DepthFrame(NamedTuple):
    events: Tensor  # [N, 4]  x/W, y/H, age/window, polarity
    disparity: Tensor  # [H, W]  zero where invalid
    mask: Tensor  # [H, W]  bool


class M3EDDepth(Dataset):
    """LiDAR-supervised frames from recorded M3ED flights, read from disk on demand.

    Events are cropped, never resampled: rescaling coordinates makes some target pixels
    collect more source pixels than others, which the feature field reads as structure.
    `min_coverage` is a fraction of pixels, not a depth; the beams meet the floor at
    grazing incidence and rarely return, so a frame is only a few percent covered.
    """

    def __init__(
        self,
        sequences: list[str | Path],
        width: int,
        height: int,
        window_ms: int,
        stride: int = 4,
        min_coverage: float = 0.01,
    ):
        self.roots = [Path(s) for s in sequences]
        self.width, self.height, self.window_ms = width, height, window_ms
        self.x0 = (SENSOR_W - width) // 2
        self.y0 = (SENSOR_H - height) // 2
        self.index: list[tuple[int, int, int]] = []  # sequence, depth frame, window end
        self._open: dict[int, tuple[h5py.File, h5py.File]] = {}

        for s, root in enumerate(self.roots):
            kept, seen = self._index_sequence(s, root, stride, min_coverage)
            logger.info(
                "%s: %d of %d frames at stride %d, the rest under %.0f%% LiDAR coverage",
                root.name,
                kept,
                seen,
                stride,
                100 * min_coverage,
            )

    def _index_sequence(
        self, s: int, root: Path, stride: int, min_coverage: float
    ) -> tuple[int, int]:
        """Record which frames are worth reading, keeping none of their pixels."""
        kept = seen = 0
        with (
            h5py.File(root / f"{root.name}_depth_gt.h5", "r") as gt,
            h5py.File(root / f"{root.name}_data.h5", "r") as data,
        ):
            depths, timestamps = gt[DEPTH], gt["ts"][:]
            span = len(data[f"{EVENTS}/ms_map_idx"])
            for i in range(0, len(timestamps), stride):
                seen += 1
                end_ms = int(timestamps[i]) // 1000
                if end_ms < self.window_ms or end_ms >= span:
                    continue
                crop = depths[
                    i, self.y0 : self.y0 + self.height, self.x0 : self.x0 + self.width
                ]
                # Too few returns to fit a scale and shift against, let alone score.
                if np.isfinite(crop).mean() < min_coverage:
                    continue
                self.index.append((s, i, end_ms))
                kept += 1
        return kept, seen

    def _handles(self, s: int) -> tuple[h5py.File, h5py.File]:
        """Per-worker file handles; h5py objects cannot cross a process boundary."""
        if s not in self._open:
            root = self.roots[s]
            self._open[s] = (
                h5py.File(root / f"{root.name}_depth_gt.h5", "r"),
                h5py.File(root / f"{root.name}_data.h5", "r"),
            )
        return self._open[s]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> DepthFrame:
        s, frame, end_ms = self.index[i]
        gt, data = self._handles(s)
        x0, y0, w, h = self.x0, self.y0, self.width, self.height

        depth = gt[DEPTH][frame, y0 : y0 + h, x0 : x0 + w]
        mask = np.isfinite(depth)  # a beam that found nothing reads +inf
        disparity = np.zeros_like(depth)
        np.divide(1.0, depth, out=disparity, where=mask)

        ms_index = data[f"{EVENTS}/ms_map_idx"]
        i0 = int(ms_index[end_ms - self.window_ms])
        i1 = int(ms_index[end_ms])
        events = data[EVENTS]
        x = events["x"][i0:i1].astype(np.int32)
        y = events["y"][i0:i1].astype(np.int32)
        inside = (x >= x0) & (x < x0 + w) & (y >= y0) & (y < y0 + h)

        window = np.empty((int(inside.sum()), 4), np.float32)
        window[:, 0] = (x[inside] - x0) / w
        window[:, 1] = (y[inside] - y0) / h
        window[:, 2] = (end_ms * 1000 - events["t"][i0:i1][inside]) / (
            self.window_ms * 1000
        )
        window[:, 3] = events["p"][i0:i1][inside]

        return DepthFrame(
            torch.from_numpy(window),
            torch.from_numpy(disparity),
            torch.from_numpy(mask),
        )


def collate_frames(frames: list[DepthFrame]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Concatenate ragged event windows into the (events, counts) pair the model takes."""
    return (
        torch.cat([f.events for f in frames]),
        torch.tensor([len(f.events) for f in frames], dtype=torch.int32),
        torch.stack([f.disparity for f in frames]),
        torch.stack([f.mask for f in frames]),
    )


def build_m3ed_loader(
    cfg: dict, width: int, height: int, window_ms: int
) -> DataLoader | None:
    """The `data.m3ed` config block -> a validation loader, or None when it is absent."""
    sequences = cfg.get("sequences", [])
    if not sequences:
        return None
    dataset = M3EDDepth(
        sequences,
        width,
        height,
        window_ms,
        int(cfg.get("stride", 4)),
        float(cfg.get("min_coverage", 0.01)),
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg.get("batch", 8)),
        shuffle=False,  # a fixed set is only comparable in a fixed order
        num_workers=int(cfg.get("num_workers", 4)),
        collate_fn=collate_frames,
        pin_memory=True,
    )


@torch.no_grad()
def evaluate_m3ed(
    model,
    loader: DataLoader,
    predict: Callable[..., Tensor],
    device,
    min_disparity: float,
    loss_fn,
) -> dict[str, float]:
    """Aligned relative-depth metrics and the training loss, averaged over batches.

    Returns the same keys as the simulator's `validate`, so either can drive selection.
    """
    totals: dict[str, float] = {}
    batches = 0
    for events, counts, disparity, mask in loader:
        events, counts = events.to(device), counts.to(device)
        disparity, mask = disparity.to(device), mask.to(device)

        height, width = disparity.shape[1:]
        pred = predict(model, events, counts, height, width).float()
        scores = eval_relative_depth(pred, disparity, mask, min_disparity)
        scores[loss_fn.name] = loss_fn(pred, disparity, mask).item()
        for name, value in scores.items():
            totals[name] = totals.get(name, 0.0) + value
        batches += 1

    return {name: value / batches for name, value in totals.items()}
