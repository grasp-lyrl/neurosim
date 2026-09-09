"""Turning disparity and raw events into images for the run's prediction panels."""

import numpy as np
import torch
from torch import Tensor


def get_disparity_image(disparity: Tensor, mask: Tensor, cmap) -> np.ndarray:
    """Disparity [H, W] -> RGB uint8, normalized over `mask` and black outside it."""
    lo, hi = disparity[mask].min().item(), disparity[mask].max().item()
    normalized = ((disparity.clamp(lo, hi) - lo) / (hi - lo)).squeeze().cpu().numpy()
    coloured = (cmap(normalized)[:, :, :3] * 255).astype(np.uint8)
    coloured[~mask.cpu().numpy()] = 0
    return coloured


def ev_to_frames_with_polarity(
    events: Tensor, counts: Tensor, w: int, h: int
) -> Tensor:
    """Normalized events -> RGB frames [B, h, w, 3], positive red and negative blue."""
    scale = torch.tensor([w, h, 1, 1], device=events.device)
    events = (events * scale).round().to(torch.int32)

    frames = torch.zeros(
        counts.shape[0], h, w, 3, dtype=torch.uint8, device=events.device
    )
    edges = torch.cumsum(torch.cat((torch.zeros(1).to(counts.device), counts)), 0).to(
        torch.int32
    )

    for i in range(counts.shape[0]):
        x, y = events[edges[i] : edges[i + 1], 0], events[edges[i] : edges[i + 1], 1]
        positive = events[edges[i] : edges[i + 1], 3] == 1
        frames[i, y[positive], x[positive], 0] = 255
        frames[i, y[~positive], x[~positive], 2] = 255

    return frames
