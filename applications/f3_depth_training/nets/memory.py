"""Convolutional GRU carried on F3's stride-16 latent."""

import torch
import torch.nn as nn
from torch import Tensor


def reset_state(state: Tensor, is_first: Tensor) -> Tensor:
    """Zero the rows whose episode restarts; `is_first` is [B] bool from the batch meta."""
    return state * ~is_first.view(-1, 1, 1, 1)


class LatentMemory(nn.Module):
    """ConvGRU over the stride-16 latent, initialised to pass this tick through unchanged.

    x    [B, C, W/16, H/16]   this tick's latent, straight out of F3's conv stack
    h    [B, C, W/16, H/16]   the latent carried from the previous tick
       │ z, r = sigmoid(gates([x, h]))
       │ n    = x + candidate([x, r * h])
    h'   [B, C, W/16, H/16]   = (1 - z) * h + z * n

    The candidate is residual around `x` and zero-init, and `z` is zero-init with a +6
    bias, so the first tick emits `x` up to sigmoid(6): a run warm-started from a
    non-recurrent checkpoint begins at that checkpoint's loss.
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.channels = channels
        padding = kernel_size // 2
        self.gates = nn.Conv2d(2 * channels, 2 * channels, kernel_size, padding=padding)
        self.candidate = nn.Conv2d(2 * channels, channels, kernel_size, padding=padding)

        nn.init.zeros_(self.candidate.weight)
        nn.init.zeros_(self.candidate.bias)
        nn.init.zeros_(self.gates.weight[:channels])
        nn.init.constant_(self.gates.bias[:channels], 6.0)

    def forward(self, latent: Tensor, state: Tensor) -> Tensor:
        """This tick's latent and the carried state -> the new state, which is the output."""
        update, reset = (
            self.gates(torch.cat([latent, state], dim=1)).sigmoid().chunk(2, dim=1)
        )
        candidate = latent + self.candidate(torch.cat([latent, reset * state], dim=1))
        return (1 - update) * state + update * candidate
