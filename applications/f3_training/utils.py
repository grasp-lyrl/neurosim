"""
Utility functions for F3 training.

Includes loss functions, metrics, and helper functions adapted from:
https://github.com/grasp-lyrl/fast-feature-fields
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import cm


# ============================================================================
# Depth Loss Functions
# Reference: https://github.com/grasp-lyrl/fast-feature-fields/blob/main/src/f3/tasks/depth/utils/losses.py
# ============================================================================


def gradient_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute gradient matching loss.

    Args:
        prediction: (B, H, W) predicted disparity
        target: (B, H, W) target disparity
        mask: (B, H, W) valid mask

    Returns:
        Gradient matching loss
    """
    M = torch.sum(mask, (1, 2))
    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))
    batch_loss = torch.sum(image_loss) / torch.sum(M)
    return batch_loss


def compute_gradient_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, scales: int = 4
) -> torch.Tensor:
    """Multi-scale gradient loss."""
    total = 0.0
    for step in range(1, scales + 1):
        total += gradient_loss(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
        )
    return total


def compute_scale_and_shift_mae(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scale and shift invariant normalization using MAE.

    Following Eqn 5,6,7 from https://arxiv.org/pdf/1907.01341

    Args:
        prediction: (B, H, W) predicted disparity
        target: (B, H, W) target disparity
        mask: (B, H, W) valid mask

    Returns:
        (normalized_prediction, normalized_target)
    """
    B, H, W = prediction.shape

    prediction_masked = prediction.masked_fill(~mask, torch.nan).view(B, -1)
    t_p = torch.nanmedian(prediction_masked, dim=-1, keepdim=True)[0]
    s_p = torch.nanmean(torch.abs(prediction_masked - t_p), dim=-1)[:, None, None]

    target_masked = target.masked_fill(~mask, torch.nan).view(B, -1)
    t_t = torch.nanmedian(target_masked, dim=-1, keepdim=True)[0]
    s_t = torch.nanmean(torch.abs(target_masked - t_t), dim=-1)[:, None, None]

    d_p_hat = (prediction - t_p[:, :, None]) / (s_p + 1e-8)
    d_t_hat = (target - t_t[:, :, None]) / (s_t + 1e-8)

    return d_p_hat, d_t_hat


class SiLogLoss(nn.Module):
    """Scale-Invariant Logarithmic Loss."""

    def __init__(self, lambd: float = 0.5):
        super().__init__()
        self.name = "SiLogLoss"
        self.lambd = lambd

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask] + 1e-6)
        return torch.sqrt(
            torch.pow(diff_log, 2).mean() - self.lambd * torch.pow(diff_log.mean(), 2)
        )


class ScaleAndShiftInvariantLoss(nn.Module):
    """
    Scale and Shift Invariant Loss with MAE normalization.

    This is the primary loss for monocular depth training with pseudo labels.
    """

    def __init__(self, alpha: float = 0.5, scales: int = 4):
        super().__init__()
        self.alpha = alpha
        self.scales = scales
        self.name = "SSIMAELoss"

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        d_p_hat, d_t_hat = compute_scale_and_shift_mae(prediction, target, mask)
        l_ssimae = F.l1_loss(d_p_hat[mask], d_t_hat[mask])
        l_reg = compute_gradient_loss(d_p_hat, d_t_hat, mask, scales=self.scales)
        return l_ssimae + self.alpha * l_reg


class SiLogGradLoss(SiLogLoss):
    """SiLog Loss with gradient matching from pseudo labels."""

    def __init__(self, lambd: float = 0.5, alpha: float = 0.5, scales: int = 4):
        super().__init__(lambd)
        self.name = "SiLogGradLoss"
        self._alpha = alpha
        self._scales = scales

    def forward(
        self,
        pred: torch.Tensor,
        target_gt: torch.Tensor,
        grad_gt: torch.Tensor,
        target_valid_mask: torch.Tensor,
        grad_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: Network prediction
            target_gt: Metric disparity from LIDAR (sparse)
            grad_gt: Non-metric depth from pseudo model (dense)
            target_valid_mask: Mask for target
            grad_valid_mask: Mask for gradient (optional)
        """
        l_silog = super().forward(pred, target_gt, target_valid_mask)

        if grad_valid_mask is None:
            grad_valid_mask = torch.ones_like(grad_gt, dtype=torch.bool)

        pred_hat, grad_gt_hat = compute_scale_and_shift_mae(
            pred, grad_gt, grad_valid_mask
        )
        l_reg = compute_gradient_loss(
            pred_hat, grad_gt_hat, grad_valid_mask, scales=self._scales
        )

        return l_silog + self._alpha * l_reg


# Depth loss registry
DEPTH_LOSSES = {
    "ssimae": ScaleAndShiftInvariantLoss,
    "silog": SiLogLoss,
    "siloggrad": SiLogGradLoss,
}


# ============================================================================
# Depth Evaluation Metrics
# Reference: https://github.com/grasp-lyrl/fast-feature-fields/blob/main/src/f3/tasks/depth/utils/utils.py
# ============================================================================


def eval_disparity(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """
    Evaluate disparity prediction.

    Args:
        pred: Predicted disparity (already masked/flattened)
        target: Target disparity (already masked/flattened)

    Returns:
        Dictionary of metrics: 1pe, 2pe, 3pe, rmse, rmse_log, log10, silog
    """
    assert pred.shape == target.shape

    pred = torch.clamp(pred, min=1e-6)
    target = torch.clamp(target, min=1e-6)

    diff = torch.abs(pred - target)
    diff_log = torch.abs(torch.log(pred) - torch.log(target))

    one_pe = torch.mean((diff > 1).float()) * 100  # Percentage
    two_pe = torch.mean((diff > 2).float()) * 100
    three_pe = torch.mean((diff > 3).float()) * 100

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log, 2)))
    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(
        torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2)
    )

    return {
        "1pe": one_pe.item(),
        "2pe": two_pe.item(),
        "3pe": three_pe.item(),
        "rmse": rmse.item(),
        "rmse_log": rmse_log.item(),
        "log10": log10.item(),
        "silog": silog.item(),
    }


def eval_depth(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """
    Evaluate depth prediction using standard metrics.

    Args:
        pred: Predicted depth
        target: Target depth

    Returns:
        Dictionary of metrics: d1, d2, d3, abs_rel, sq_rel, rmse, rmse_log, log10, silog
    """
    assert pred.shape == target.shape

    thresh = torch.max((target / pred), (pred / target))
    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25**2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25**3).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / torch.pow(target, 2))
    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log, 2)))
    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(
        torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2)
    )

    return {
        "d1": d1.item(),
        "d2": d2.item(),
        "d3": d3.item(),
        "abs_rel": abs_rel.item(),
        "sq_rel": sq_rel.item(),
        "rmse": rmse.item(),
        "rmse_log": rmse_log.item(),
        "log10": log10.item(),
        "silog": silog.item(),
    }


def get_disparity_image(
    disparity: torch.Tensor, mask: torch.Tensor, cmap=None
) -> np.ndarray:
    """Convert disparity to colorized image for visualization."""
    if cmap is None:
        cmap = cm.get_cmap("magma")

    if isinstance(disparity, torch.Tensor):
        min_disp = disparity[mask].min().item()
        max_disp = disparity[mask].max().item()
        disp_colored = torch.clamp(disparity, min_disp, max_disp)
        disp_colored = (disp_colored - min_disp) / (max_disp - min_disp + 1e-8)
        disp_colored = disp_colored.squeeze().cpu().numpy()
        disp_colored = (cmap(disp_colored)[:, :, :3] * 255).astype(np.uint8)
        disp_colored[~mask.cpu().numpy()] = 0
    else:
        min_disp = disparity[mask].min()
        max_disp = disparity[mask].max()
        disp_colored = np.clip(disparity, min_disp, max_disp)
        disp_colored = (disp_colored - min_disp) / (max_disp - min_disp + 1e-8)
        disp_colored = (cmap(disp_colored)[:, :, :3] * 255).astype(np.uint8)
        disp_colored[~mask] = 0

    return disp_colored


def set_best_results(best_results: dict, val_results: dict) -> None:
    """Update best results dictionary with better values."""
    for key in best_results:
        if key in val_results:
            if isinstance(val_results[key], torch.Tensor):
                val = val_results[key].item()
            else:
                val = val_results[key]
            best_results[key] = min(best_results[key], val)


def log_dict(logger, results: dict) -> None:
    """Log a dictionary of results."""
    for key, val in results.items():
        if isinstance(val, torch.Tensor):
            val = val.item()
        logger.info(f"  {key}: {val:.4f}")


# ============================================================================
# Event Processing Utilities
# ============================================================================


def unnormalize_events(
    events: torch.Tensor, resolution: tuple[int, int, int]
) -> torch.Tensor:
    """
    Convert normalized events [0, 1] to pixel coordinates.

    Args:
        events: (N, 3) or (N, 4) tensor with normalized coordinates
        resolution: (W, H, T) target resolution

    Returns:
        Events with integer pixel coordinates
    """
    w, h, t = resolution
    events = events.clone()
    events[..., 0] = (events[..., 0] * w).round().long().clamp(0, w - 1)
    events[..., 1] = (events[..., 1] * h).round().long().clamp(0, h - 1)
    if events.shape[-1] > 2:
        events[..., 2] = (events[..., 2] * t).round().long().clamp(0, t - 1)
    return events


@torch.compile
def ev_to_grid(
    events: torch.Tensor, counts: torch.Tensor, w: int, h: int, t: int
) -> torch.Tensor:
    """
    Convert events to voxel grid representation.

    Args:
        events: (N, 3) tensor of events [x, y, t] (normalized or pixel coords)
        counts: (B,) tensor of event counts per batch
        w, h, t: Grid dimensions

    Returns:
        (B, W, H, T) voxel grid
    """
    if events.max() <= 1.0:
        events = unnormalize_events(events, (w, h, t))

    B = counts.shape[0]
    device = events.device

    pred_event_grid = torch.zeros(B, w, h, t, device=device, dtype=torch.float32)
    c = torch.cumsum(torch.cat((torch.zeros(1, device=device), counts)), 0).long()

    for i in range(B):
        ev = events[c[i] : c[i + 1]]
        pred_event_grid[i, ev[:, 0].long(), ev[:, 1].long(), ev[:, 2].long()] = 1

    return pred_event_grid


def ev_to_frames(
    events: torch.Tensor, counts: torch.Tensor, w: int, h: int
) -> torch.Tensor:
    """
    Convert events to frame representation (summed over time).

    Args:
        events: (N, 3) tensor of events [x, y, t]
        counts: (B,) tensor of event counts per batch
        w, h: Frame dimensions

    Returns:
        (B, W, H) frame tensor
    """
    if events.max() <= 1.0:
        events = unnormalize_events(events, (w, h, 1))

    B = counts.shape[0]
    device = events.device

    frames = torch.zeros(B, w, h, dtype=torch.uint8, device=device)
    c = torch.cumsum(torch.cat((torch.zeros(1, device=device), counts)), 0).long()

    for i in range(B):
        ev = events[c[i] : c[i + 1]]
        frames[i, ev[:, 0].long(), ev[:, 1].long()] = 255

    return frames
