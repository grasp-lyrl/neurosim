"""Instant-NGP style multi-resolution hash encoding of raw events."""

import torch
import torch.nn as nn
from torch import Tensor

PI1, PI2, PI3 = 1, 2654435761, 805459861


class MultiResolutionHashEncoder(nn.Module):
    """Per-event lookup: (x, y, t) in [0,1]^3 -> [L*F] features. No grid, a table read per event.

    Level resolutions are a geometric series from coarsest to finest. A level whose cells fit
    in 2^log2_entries_per_level is indexed directly; finer levels collide into the table by
    hash. The split is a property of the config: the 640x480x20 checkpoint lands on L_H = 0,
    all direct.
    """

    def __init__(
        self,
        coarsest_resolution: list[int],
        finest_resolution: list[int],
        levels: int,
        feature_size: int,
        log2_entries_per_level: int,
    ):
        super().__init__()
        self.levels = levels
        self.feature_size = feature_size
        self.log2_entries_per_level = log2_entries_per_level

        coarsest = torch.tensor(coarsest_resolution, dtype=torch.float32)
        finest = torch.tensor(finest_resolution, dtype=torch.float32)
        ratio = (finest.log() - coarsest.log()) / (levels - 1)
        resolutions = torch.exp(
            coarsest.log() + torch.arange(levels)[:, None] * ratio
        ).int()

        self.L_H = int(
            (torch.log2(resolutions + 1.0).sum(1) > log2_entries_per_level).sum()
        )
        self.L_NH = levels - self.L_H

        hashmap = torch.empty(levels, 1 << log2_entries_per_level, feature_size)
        self.hashmap = nn.Parameter(hashmap.uniform_(-1e-4, 1e-4))

        # Non-persistent: `.to(device)` must carry these, but they are not checkpoint state.
        self.register_buffer("resolutions", resolutions, persistent=False)
        bits = torch.arange(8)[:, None] >> torch.tensor([2, 1, 0])
        self.register_buffer("is_ceil", (bits & 1).bool(), persistent=False)

    def _corners(self, events: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Events [B, N, 3] -> direct and hashed table indices, plus the eight corner weights."""
        scaled = events.unsqueeze(-2) * self.resolutions  # [B, N, L, 3]
        floor = scaled.int()
        corners = torch.where(  # [B, N, L, 8, 3]
            self.is_ceil,
            torch.min(floor + 1, self.resolutions).unsqueeze(-2),
            floor.unsqueeze(-2),
        )
        weights = torch.prod(1 - (corners - scaled.unsqueeze(-2)).abs(), dim=-1)

        direct, hashed = corners[:, :, : self.L_NH], corners[:, :, self.L_NH :]
        res = self.resolutions[: self.L_NH]
        direct_idx = (
            direct[..., 0]
            + direct[..., 1] * res[:, 0, None]
            + direct[..., 2] * (res[:, 0] * res[:, 1])[:, None]
        )
        hashed_idx = (
            hashed[..., 0] * PI1 ^ hashed[..., 1] * PI2 ^ hashed[..., 2] * PI3
        ) % (1 << self.log2_entries_per_level)
        return direct_idx, hashed_idx, weights

    def forward(self, events: Tensor) -> Tensor:
        """Events [B, N, 3] normalized to [0,1] -> interpolated features [B, N, L*F]."""
        B, N, _ = events.shape
        direct_idx, hashed_idx, weights = self._corners(events)

        # Per level rather than one gather: gathering across levels needs an expand over N or
        # F, which goes OOM in the backward pass.
        feats = torch.zeros(
            (B, N, self.levels, 8, self.feature_size),
            dtype=self.hashmap.dtype,
            device=events.device,
        )
        for i in range(self.L_NH):
            feats[:, :, i] = self.hashmap[i][direct_idx[:, :, i]]
        for i in range(self.L_H):
            feats[:, :, i + self.L_NH] = self.hashmap[i + self.L_NH][
                hashed_idx[:, :, i]
            ]

        # flatten, not reshape(B, N, -1): a tick can hold no events at all, and then the -1 is
        # ambiguous rather than zero.
        return (weights.unsqueeze(-1) * feats).sum(-2).flatten(2)
