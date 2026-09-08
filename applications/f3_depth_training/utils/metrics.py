"""Disparity evaluation metrics and colourisation, after f3's depth task utils."""

import math

import numpy as np
import torch
from torch import Tensor

# Metrics a larger value is better for. Everything else here is an error.
HIGHER_IS_BETTER = frozenset({"d1", "d2", "d3"})


def align_least_squares(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Per-sample scale and shift carrying `pred` onto `target` over `mask`, closed form."""
    # Centred: the normal-equation form differences two sums that agree to every digit.
    b = pred.shape[0]
    m = mask.view(b, -1).double()
    n = m.sum(1, keepdim=True).clamp(min=1)
    p, g = pred.view(b, -1).double(), target.view(b, -1).double()

    mean_p, mean_g = (p * m).sum(1, keepdim=True) / n, (g * m).sum(1, keepdim=True) / n
    dp, dg = (p - mean_p) * m, (g - mean_g) * m

    var = (dp * dp).sum(1, keepdim=True)
    scale = torch.where(
        var > 0, (dp * dg).sum(1, keepdim=True) / var.clamp(min=1e-30), 0
    )
    return (scale * (p - mean_p) + mean_g).view_as(pred).to(pred.dtype)


def eval_relative_depth(
    pred: Tensor, target: Tensor, mask: Tensor, min_disparity: float = 0.05
) -> dict[str, float]:
    """The affine-invariant protocol: align disparity, invert, measure in metres."""
    aligned = align_least_squares(pred, target, mask).clamp(min=min_disparity)
    depth, truth = 1.0 / aligned, 1.0 / target.clamp(min=min_disparity)

    ratio = torch.maximum(depth / truth, truth / depth)
    error, log_error = depth - truth, torch.log(depth) - torch.log(truth)
    counts = mask.flatten(1).sum(1)

    def per_image(values: Tensor) -> Tensor:
        """Mean over each image's valid pixels, images without any dropped."""
        return ((values * mask).flatten(1).sum(1) / counts.clamp(min=1))[counts > 0]

    mean_sq_log = per_image(log_error**2)
    per_metric = {
        "abs_rel": per_image(error.abs() / truth),
        "sq_rel": per_image(error**2 / truth),
        **{f"d{i}": per_image((ratio < 1.25**i).float()) * 100 for i in (1, 2, 3)},
        "rmse": per_image(error**2).sqrt(),
        "rmse_log": mean_sq_log.sqrt(),
        "log10": per_image(log_error.abs()) / math.log(10),
        "silog": (mean_sq_log - per_image(log_error) ** 2).clamp(min=0).sqrt() * 100,
    }
    return {name: value.mean().item() for name, value in per_metric.items()}


def get_disparity_image(disparity: Tensor, mask: Tensor, cmap) -> np.ndarray:
    """Disparity [H, W] -> RGB uint8, normalized over `mask` and black outside it."""
    lo, hi = disparity[mask].min().item(), disparity[mask].max().item()
    normalized = ((disparity.clamp(lo, hi) - lo) / (hi - lo)).squeeze().cpu().numpy()
    coloured = (cmap(normalized)[:, :, :3] * 255).astype(np.uint8)
    coloured[~mask.cpu().numpy()] = 0
    return coloured


def set_best_results(best: dict, new: dict) -> None:
    """Keep each metric's better value in place, per `HIGHER_IS_BETTER`."""
    for k in best:
        value = new[k]
        value = value.item() if isinstance(value, Tensor) else value
        best[k] = max(best[k], value) if k in HIGHER_IS_BETTER else min(best[k], value)
