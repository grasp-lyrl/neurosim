"""Depth metrics in metres, the affine fit relative predictions need first, and best tracking."""

import math

import torch
from torch import Tensor

NEAR_FIELD = 5.0  # metres, the deployment brief's working range
METRICS = (
    "abs_rel",
    "sq_rel",
    "d1",
    "d2",
    "d3",
    "rmse",
    "rmse_log",
    "log10",
    "silog",
    "abs_rel_near",
    "d1_near",
)
HIGHER_IS_BETTER = frozenset(
    {"d1", "d2", "d3", "d1_near", "d1_aligned", "d1_near_aligned"}
)


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


def depth_metrics(depth: Tensor, truth: Tensor, mask: Tensor) -> dict[str, float]:
    """Per-image means over `mask`; `*_near` over truth under NEAR_FIELD. NaN when no image has a pixel."""
    ratio = torch.maximum(depth / truth, truth / depth)
    error, log_error = depth - truth, torch.log(depth) - torch.log(truth)
    near = mask & (truth < NEAR_FIELD)

    def per_image(values: Tensor, mask: Tensor) -> Tensor:
        counts = mask.flatten(1).sum(1)
        return ((values * mask).flatten(1).sum(1) / counts.clamp(min=1))[counts > 0]

    mean_sq_log = per_image(log_error**2, mask)
    per_metric = {
        "abs_rel": per_image(error.abs() / truth, mask),
        "sq_rel": per_image(error**2 / truth, mask),
        **{
            f"d{i}": per_image((ratio < 1.25**i).float(), mask) * 100 for i in (1, 2, 3)
        },
        "rmse": per_image(error**2, mask).sqrt(),
        "rmse_log": mean_sq_log.sqrt(),
        "log10": per_image(log_error.abs(), mask) / math.log(10),
        "silog": (mean_sq_log - per_image(log_error, mask) ** 2).clamp(min=0).sqrt()
        * 100,
        "abs_rel_near": per_image(error.abs() / truth, near),
        "d1_near": per_image((ratio < 1.25).float(), near) * 100,
    }
    return {name: value.mean().item() for name, value in per_metric.items()}


def mean_scores(batches: list[dict[str, float]]) -> dict[str, float]:
    """Mean of each score over batches, skipping the batches where it was NaN."""
    return {
        k: torch.tensor([b[k] for b in batches], dtype=torch.float64).nanmean().item()
        for k in batches[0]
    }


def improved(name: str, value: float, best: float) -> bool:
    """Whether `value` beats `best`, in that metric's own direction."""
    return value > best if name in HIGHER_IS_BETTER else value < best


def set_best_results(best: dict, new: dict) -> None:
    """Keep each metric's better value in place, per `HIGHER_IS_BETTER`."""
    for k in best:
        value = new[k]
        value = value.item() if isinstance(value, Tensor) else value
        if improved(k, value, best[k]):
            best[k] = value
