"""Relative disparity loss after MiDaS (arXiv 1907.01341) and metric depth loss after DAv2."""

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
        self.parts = {}

    def forward(self, prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        pred_hat, target_hat = normalize_scale_shift(prediction, target, mask)
        l_ssimae = nn.functional.l1_loss(pred_hat[mask], target_hat[mask])
        l_reg = multiscale_gradient_loss(pred_hat, target_hat, mask, self.scales)
        self.parts = {"data": l_ssimae.detach(), "grad": (self.alpha * l_reg).detach()}
        return l_ssimae + self.alpha * l_reg


class SiLogLoss(nn.Module):
    """SiLog after f3's, plus `alpha` times the gradient term on log depth (f3 uses a pseudo-label; sim GT is dense)."""

    def __init__(self, lambd: float = 0.5, alpha: float = 0.0, scales: int = 4):
        super().__init__()
        self.name = "SiLogGradLoss" if alpha else "SiLogLoss"
        self.lambd = lambd
        self.alpha = alpha
        self.scales = scales
        self.parts = {}

    def forward(self, prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        # log(0) at invalid pixels would reach the gradient term as 0 * inf.
        log_pred = torch.log(prediction + 1e-6)
        log_target = torch.log(target.masked_fill(~mask, 1.0))
        diff = (log_target - log_pred)[mask]
        l_silog = torch.sqrt((diff**2).mean() - self.lambd * diff.mean() ** 2)
        l_reg = multiscale_gradient_loss(log_pred, log_target, mask, self.scales)
        self.parts = {"data": l_silog.detach(), "grad": (self.alpha * l_reg).detach()}
        return l_silog + self.alpha * l_reg
