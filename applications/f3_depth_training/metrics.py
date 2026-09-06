"""Disparity evaluation metrics and colourisation, after f3's depth task utils."""

import numpy as np
import torch
from torch import Tensor


def eval_disparity(pred: Tensor, target: Tensor) -> dict[str, float]:
    """Percent-error, RMSE and log metrics over already-masked disparity values."""
    pred = torch.clamp(pred, min=1e-6)
    target = torch.clamp(target, min=1e-6)

    diff = torch.abs(pred - target)
    diff_log = torch.abs(torch.log(pred) - torch.log(target))

    return {
        "1pe": (torch.mean((diff > 1).float()) * 100).item(),
        "2pe": (torch.mean((diff > 2).float()) * 100).item(),
        "3pe": (torch.mean((diff > 3).float()) * 100).item(),
        "rmse": torch.sqrt(torch.mean(diff**2)).item(),
        "rmse_log": torch.sqrt(torch.mean(diff_log**2)).item(),
        "log10": torch.mean(torch.abs(torch.log10(pred) - torch.log10(target))).item(),
        "silog": torch.sqrt((diff_log**2).mean() - 0.5 * diff_log.mean() ** 2).item(),
    }


def get_disparity_image(disparity: Tensor, mask: Tensor, cmap) -> np.ndarray:
    """Disparity [H, W] -> RGB uint8, normalized over `mask` and black outside it."""
    lo, hi = disparity[mask].min().item(), disparity[mask].max().item()
    normalized = ((disparity.clamp(lo, hi) - lo) / (hi - lo)).squeeze().cpu().numpy()
    coloured = (cmap(normalized)[:, :, :3] * 255).astype(np.uint8)
    coloured[~mask.cpu().numpy()] = 0
    return coloured


def set_best_results(best: dict, new: dict) -> None:
    """Keep the lower value of each metric in place (all metrics here are error-like)."""
    for k in best:
        value = new[k]
        best[k] = min(best[k], value.item() if isinstance(value, Tensor) else value)
