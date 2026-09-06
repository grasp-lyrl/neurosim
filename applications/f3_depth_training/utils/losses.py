"""Scale- and shift-invariant relative disparity loss, after MiDaS (arXiv 1907.01341)."""

import torch
from torch import Tensor, nn


def normalize_scale_shift(prediction: Tensor, target: Tensor, mask: Tensor):
    """Per-sample median-centred, MAD-scaled disparity, eqns 5-7 of the MiDaS paper."""
    B = prediction.shape[0]

    masked = prediction.masked_fill(~mask, torch.nan).view(B, -1)
    t_p = torch.nanmedian(masked, dim=-1, keepdim=True)[0]
    s_p = torch.nanmean(torch.abs(masked - t_p), dim=-1)[:, None, None]

    masked = target.masked_fill(~mask, torch.nan).view(B, -1)
    t_t = torch.nanmedian(masked, dim=-1, keepdim=True)[0]
    s_t = torch.nanmean(torch.abs(masked - t_t), dim=-1)[:, None, None]

    return (prediction - t_p[:, :, None]) / s_p, (target - t_t[:, :, None]) / s_t


def gradient_loss(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Mean absolute difference of horizontal and vertical gradients over valid pixels."""
    diff = mask * (prediction - target)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1]) * (
        mask[:, :, 1:] * mask[:, :, :-1]
    )
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :]) * (
        mask[:, 1:, :] * mask[:, :-1, :]
    )
    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))
    return torch.sum(image_loss) / torch.sum(mask, (1, 2)).sum()


def multiscale_gradient_loss(
    prediction: Tensor, target: Tensor, mask: Tensor, scales: int = 4
) -> Tensor:
    """Eqn 11 of the MiDaS paper: gradient loss summed over `scales` decimations."""
    return sum(
        gradient_loss(
            prediction[:, :: 2**s, :: 2**s],
            target[:, :: 2**s, :: 2**s],
            mask[:, :: 2**s, :: 2**s],
        )
        for s in range(scales)
    )


class ScaleAndShiftInvariantLoss(nn.Module):
    """L1 on normalized disparity plus a multi-scale gradient term."""

    def __init__(self, alpha: float = 0.5, scales: int = 4):
        super().__init__()
        self.name = "SSIMAELoss"
        self.alpha = alpha
        self.scales = scales

    def forward(self, prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        pred_hat, target_hat = normalize_scale_shift(prediction, target, mask)
        l_ssimae = nn.functional.l1_loss(pred_hat[mask], target_hat[mask])
        l_reg = multiscale_gradient_loss(pred_hat, target_hat, mask, self.scales)
        return l_ssimae + self.alpha * l_reg
