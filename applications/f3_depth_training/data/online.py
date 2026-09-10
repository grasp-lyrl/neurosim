"""The simulator stream: an OnlineDataLoader of time-aligned events and depth."""

from typing import Callable

import numpy as np
import torch
import yaml
from torch import Tensor

from neurosim.online_data import OnlineDataLoader, TimeAlignedSample


def usable_sample_filter(
    depth_uuid: str,
    event_sensor: str,
    max_disparity: float,
    min_events: int,
    min_valid_frac: float = 0.5,
) -> Callable[[TimeAlignedSample], bool]:
    """Predicate dropping samples the model cannot learn from.

    Rejects a mostly-invalid depth anchor (0 m reads, e.g. a camera facing open sky) and
    packets too sparse for the backbone. Applied before batching, so a rejected sample
    costs a row rather than shrinking the batch.
    """
    min_depth = 1.0 / max_disparity

    def keep(sample: TimeAlignedSample) -> bool:
        depth = sample.sensors[depth_uuid]
        if np.count_nonzero(depth > min_depth) < min_valid_frac * depth.size:
            return False
        return len(sample.sensors[event_sensor].get("x", ())) >= min_events

    return keep


def build_online_loader(
    data_cfg: dict,
    *,
    batch_size: int,
    log_dir: str | None = None,
    sample_filter: Callable[[TimeAlignedSample], bool] | None = None,
) -> OnlineDataLoader:
    """Build an OnlineDataLoader from the ``data.online_data`` config block.

    Delegates to :meth:`OnlineDataLoader.from_config` (the same YAML schema used
    everywhere — roles, scenes, DR and loader knobs live in ``data.online_data``), with
    an optional ``data.sim_time`` overriding the base settings' episode length.
    """
    od = dict(data_cfg["online_data"])

    sim_time = data_cfg.get("sim_time")
    if sim_time is not None:
        base = od.get("base_settings")
        if isinstance(base, str):
            with open(base, "r") as f:
                base = yaml.safe_load(f)
        base.setdefault("simulator", {})["sim_time"] = sim_time
        od["base_settings"] = base

    return OnlineDataLoader.from_config(
        {"online_data": od},
        batch_size=batch_size,
        log_dir=log_dir,
        sample_filter=sample_filter,
    )


def cap_events(events: np.ndarray, counts: np.ndarray, cap: int):
    """Thin each sample down to `cap` events, evenly across its window.

    Deployment runs a fixed event budget so the backbone fits the frame time, so training
    sees the same. Evenly spaced rather than newest-first: the model is trained on a whole
    window, and dropping the oldest events would quietly shorten it.
    """
    if cap <= 0 or counts.max() <= cap:
        return events, counts

    offsets = np.concatenate([[0], np.cumsum(counts)])
    keep = [
        np.arange(offsets[i], offsets[i + 1])
        if n <= cap
        else offsets[i] + np.linspace(0, n - 1, cap).astype(np.int64)
        for i, n in enumerate(counts)
    ]
    return events[np.concatenate(keep)], np.minimum(counts, cap).astype(counts.dtype)


def process_batch(batch, args, device):
    """One loader batch -> ``(events, counts, disparity, color_images)`` on `device`.

    ``batch[event_sensor]`` is ``(counts, events)`` with events raw as
    ``[x, y, t_anchor - t, p]`` in pixels and anchor-relative µs; the loader does not
    normalize, so this divides by ``args.event_norm = (W, H, window_us)``.
    """
    event_sensor, depth_sensor = args.event_sensor, args.depth_sensor
    assert event_sensor in batch, f"no '{event_sensor}' in batch: {list(batch)}"
    assert depth_sensor in batch, f"no '{depth_sensor}' in batch: {list(batch)}"

    counts, events = batch[event_sensor]
    events, counts = cap_events(events, counts, args.max_events)
    ff_events = torch.from_numpy(events).float().to(device)
    event_counts = torch.from_numpy(counts).to(device)
    ff_events[:, :3] /= torch.tensor(
        args.event_norm, device=device, dtype=torch.float32
    )

    # Invalid depths read 0.0; clip before inverting to disparity.
    depth = torch.from_numpy(batch[depth_sensor]).to(device, torch.float32)
    disparity = 1.0 / depth.clamp(0.5 / args.max_disparity, 1 / args.min_disparity)

    color_sensor = args.color_sensor
    color_images = (
        batch[color_sensor].astype(np.uint8)
        if color_sensor and color_sensor in batch
        else None
    )
    return ff_events, event_counts, disparity, color_images


def usable_samples(valid_mask: Tensor) -> Tensor:
    """Rows with at least half their depth pixels valid."""
    return valid_mask.flatten(1).sum(1) * 2 >= valid_mask[0].numel()
