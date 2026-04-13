"""Task interfaces for Neurosim RL environments."""

import math
from typing import Any
from abc import ABC, abstractmethod

import torch
import numpy as np


class EventRepresentationManager:
    """Owns event representation state and update logic for RL observations."""

    # Reference count level in log normalization: output maps ~linearly near this
    # accumulation (replaces the former ``event_clip`` scale in the denominator).
    _EVENT_LOG_COUNT_REFERENCE = 10.0

    def __init__(
        self,
        representation: str,
        raw_height: int,
        raw_width: int,
        downsample_factor: int,
        event_log_compression: float,
        ts_tau_seconds: float,
        event_device: str | torch.device,
    ):
        self.representation = representation
        self.raw_height = int(raw_height)
        self.raw_width = int(raw_width)
        self.downsample_factor = max(int(downsample_factor), 1)
        self.event_log_compression = float(event_log_compression)
        self.ts_tau_seconds = float(ts_tau_seconds)

        self._device = torch.device(event_device)

        self._raw = torch.zeros(
            (2, self.raw_height, self.raw_width),
            dtype=torch.float32,
            device=self._device,
        )
        self.step_event_count = 0
        self._last_update_time_s: float | None = None

    def reset_episode(self) -> None:
        self._raw.zero_()
        self.step_event_count = 0
        self._last_update_time_s = None

    def begin_step(self) -> None:
        self.step_event_count = 0
        if self.representation != "time_surface":
            self._raw.zero_()

    def _downsample_torch(self, event_rep: torch.Tensor) -> torch.Tensor:
        if self.downsample_factor == 1:
            return event_rep

        factor = self.downsample_factor
        h = (event_rep.shape[1] // factor) * factor
        w = (event_rep.shape[2] // factor) * factor
        cropped = event_rep[:, :h, :w]
        reshaped = cropped.view(2, h // factor, factor, w // factor, factor)
        return reshaped.mean(dim=(2, 4), dtype=torch.float32)

    def _normalize_torch(self, event_rep: torch.Tensor) -> torch.Tensor:
        k = self.event_log_compression
        ref = self._EVENT_LOG_COUNT_REFERENCE
        out = torch.log1p(k * event_rep) / math.log1p(k * ref)
        return torch.clamp(out, 0.0, 1.0)

    def accumulate(self, events: Any | None) -> None:
        if events is None:
            return

        x, y, t, p = events.x, events.y, events.t, events.p

        assert x.device == self._device
        assert y.device == self._device
        assert t.device == self._device
        assert p.device == self._device

        #! Convert from uint16 to int32
        x = x.to(dtype=torch.int32, non_blocking=True)
        y = y.to(dtype=torch.int32, non_blocking=True)
        #! t is uint64, so no need to convert
        p = p.to(dtype=torch.int32, non_blocking=True)

        if x.numel() == 0:
            return

        self.step_event_count += int(x.numel())

        if self.representation == "histogram":
            hw = self.raw_height * self.raw_width
            flat_idx = p * hw + y * self.raw_width + x
            counts = torch.bincount(flat_idx, minlength=2 * hw).to(
                dtype=torch.float32, device=self._device
            )
            self._raw += counts.view(2, self.raw_height, self.raw_width)
            return

        if self.representation == "event_frame":
            self._raw[p, y, x] = 1.0
            return

        if self.representation == "time_surface":
            latest_time_s = float((t[-1].float() * 1e-6).item())
            if self._last_update_time_s is not None:
                dt_seconds = latest_time_s - self._last_update_time_s
                if dt_seconds > 0.0:
                    decay = math.exp(-dt_seconds / self.ts_tau_seconds)
                    self._raw.mul_(decay)
            self._last_update_time_s = latest_time_s
            self._raw[p, y, x] += 1.0
            return

        raise ValueError(f"Unsupported event_representation: {self.representation}")

    def observation(self) -> np.ndarray:
        buf = self._raw
        if self.representation == "histogram":
            buf = self._normalize_torch(self._raw)
        obs = self._downsample_torch(buf)
        return obs.detach().cpu().numpy().astype(np.float32, copy=False)

    def to_rgb(self, event_rep: np.ndarray) -> np.ndarray:
        """Pack event polarity channels into an RGB uint8 image (R=neg, B=pos, G=0)."""
        neg, pos = event_rep[0], event_rep[1]
        rgb = np.zeros((neg.shape[0], neg.shape[1], 3), dtype=np.uint8)
        rgb[..., 0] = np.rint(255.0 * neg).astype(np.uint8)
        rgb[..., 2] = np.rint(255.0 * pos).astype(np.uint8)
        return rgb


class RLTask(ABC):
    """Interface for task-specific reward, success, and termination logic."""

    @property
    @abstractmethod
    def crash_penalty(self) -> float:
        """Penalty applied by the environment when the episode terminates unsafely."""

    @abstractmethod
    def on_reset(self) -> None:
        """Reset task-specific episode state."""

    @abstractmethod
    def compute_reward(
        self,
        *,
        state: dict[str, np.ndarray],
        action: np.ndarray,
        prev_action: np.ndarray | None,
        event_manager: EventRepresentationManager,
        obs_mode: str,
    ) -> tuple[float, dict[str, float]]:
        """Compute reward and structured reward terms."""

    @abstractmethod
    def check_success(self, *, state: dict[str, np.ndarray]) -> bool:
        """Return whether the current state satisfies task success criteria."""

    def check_terminated(self, *, state: dict[str, np.ndarray]) -> tuple[bool, str]:
        """Optional task-specific termination checks."""
        _ = state
        return False, ""
