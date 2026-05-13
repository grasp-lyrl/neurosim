"""CTBR vehicle backed by RotorPy multirotor dynamics."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from gymnasium import spaces

from neurosim.core.dynamics.rotorpy_wrapper import get_vehicle_params

from .base import RLVehicle


@dataclass
class RateLimits:
    roll: float
    pitch: float
    yaw: float


class RotorpyCtbrVehicle(RLVehicle):
    """CTBR policy head for RotorPy multirotor dynamics (body rates + thrust)."""

    def __init__(
        self,
        *,
        dynamics: Any,
        vehicle: str,
        rate_limits: RateLimits,
        domain_randomization: dict[str, Any] | None = None,
    ):
        self._dynamics = dynamics
        self._multirotor = dynamics._multirotor
        self.vehicle = str(vehicle)
        self._rate_limits = rate_limits
        self._domain_randomization: dict[str, Any] | None = domain_randomization

        self.reference_vehicle_params = get_vehicle_params(self.vehicle)

        self._base_dynamic_params = {
            "mass": float(self._multirotor.mass),
            "k_eta": float(self._multirotor.k_eta),
            "k_m": float(self._multirotor.k_m),
            "rotor_speed_min": float(self._multirotor.rotor_speed_min),
            "rotor_speed_max": float(self._multirotor.rotor_speed_max),
        }

        self._action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        self._refresh_from_multirotor()

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    @property
    def hover_thrust(self) -> float:
        return self._hover_thrust

    @property
    def control_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        low = np.asarray(
            [
                self._cmd_thrust_min,
                -self._rate_limits.roll,
                -self._rate_limits.pitch,
                -self._rate_limits.yaw,
            ],
            dtype=np.float64,
        )
        high = np.asarray(
            [
                self._cmd_thrust_max,
                self._rate_limits.roll,
                self._rate_limits.pitch,
                self._rate_limits.yaw,
            ],
            dtype=np.float64,
        )
        return low, high

    @staticmethod
    def _minmax_scale(
        x: np.ndarray,
        min_values: float | np.ndarray,
        max_values: float | np.ndarray,
    ) -> np.ndarray:
        x_scaled = (x + 1.0) * 0.5 * (max_values - min_values) + min_values
        return np.clip(x_scaled, min_values, max_values)

    def _refresh_from_multirotor(self) -> None:
        m = self._multirotor
        self._hover_thrust = float(m.mass * m.g)
        self._cmd_thrust_min = float(m.num_rotors * m.k_eta * m.rotor_speed_min**2)
        self._cmd_thrust_max = float(m.num_rotors * m.k_eta * m.rotor_speed_max**2)
        if not (self._cmd_thrust_min < self._hover_thrust < self._cmd_thrust_max):
            raise ValueError(
                "Hover thrust must be within thrust bounds after vehicle parameter updates"
            )

    def _apply_domain_randomization(self, rng: np.random.Generator) -> None:
        """Resample multirotor fields; ``randomize`` must only call when ``scales`` is present."""
        scales = self._domain_randomization["scales"]
        for key, base_value in self._base_dynamic_params.items():
            low, high = scales[key]
            sampled = float(base_value) * float(rng.uniform(float(low), float(high)))
            setattr(self._multirotor, key, sampled)

    def randomize(self, episode_count: int, rng: np.random.Generator) -> None:
        dr = self._domain_randomization
        if dr is None or not dr["enabled"]:
            return
        if episode_count % dr["resample_every"] != 0:
            return

        self._apply_domain_randomization(rng)
        self._refresh_from_multirotor()

    def action_to_control(self, action: np.ndarray) -> dict[str, np.ndarray | float]:
        action = np.asarray(action, dtype=np.float32)

        cmd_thrust = float(
            self._minmax_scale(action[0], self._cmd_thrust_min, self._cmd_thrust_max)
        )
        cmd_roll_br = float(
            self._minmax_scale(
                action[1], -self._rate_limits.roll, self._rate_limits.roll
            )
        )
        cmd_pitch_br = float(
            self._minmax_scale(
                action[2], -self._rate_limits.pitch, self._rate_limits.pitch
            )
        )
        cmd_yaw_br = float(
            self._minmax_scale(action[3], -self._rate_limits.yaw, self._rate_limits.yaw)
        )
        return {
            "cmd_thrust": cmd_thrust,
            "cmd_w": np.asarray(
                [cmd_roll_br, cmd_pitch_br, cmd_yaw_br], dtype=np.float64
            ),
        }

    def clip_control(
        self, control: dict[str, np.ndarray | float]
    ) -> dict[str, np.ndarray | float]:
        low, high = self.control_bounds
        cmd = np.asarray(
            [
                float(control["cmd_thrust"]),
                *np.asarray(control["cmd_w"], dtype=np.float64).reshape(3),
            ],
            dtype=np.float64,
        )
        clipped = np.clip(cmd, low, high)
        merged = dict(control)
        merged["cmd_thrust"] = float(clipped[0])
        merged["cmd_w"] = clipped[1:].astype(np.float64, copy=False)
        return merged

    def control_to_normalized(
        self, control: dict[str, np.ndarray | float]
    ) -> np.ndarray:
        low, high = self.control_bounds
        cmd = np.asarray(
            [
                float(control["cmd_thrust"]),
                *np.asarray(control["cmd_w"], dtype=np.float64).reshape(3),
            ],
            dtype=np.float64,
        )
        normalized = 2.0 * (cmd - low) / (high - low) - 1.0
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def apply_ctbr_delta(
        self,
        nominal_control: dict[str, np.ndarray | float],
        action: np.ndarray,
        *,
        delta_thrust_fraction: float,
        delta_rate_limits: np.ndarray,
    ) -> dict[str, np.ndarray | float]:
        action = np.asarray(action, dtype=np.float32)
        delta_rate_limits = np.asarray(delta_rate_limits, dtype=np.float64).reshape(3)
        delta_thrust = (
            float(action[0]) * float(delta_thrust_fraction) * self._hover_thrust
        )
        delta_w = np.asarray(action[1:4], dtype=np.float64) * delta_rate_limits

        merged = dict(nominal_control)
        merged["cmd_thrust"] = float(nominal_control["cmd_thrust"]) + delta_thrust
        merged["cmd_w"] = (
            np.asarray(nominal_control["cmd_w"], dtype=np.float64) + delta_w
        )
        return self.clip_control(merged)

    def apply_gated_ctbr_delta(
        self,
        nominal_control: dict[str, np.ndarray | float],
        gate: float,
        action: np.ndarray,
        *,
        delta_thrust_fraction: float,
        delta_rate_limits: np.ndarray,
    ) -> dict[str, np.ndarray | float]:
        """Apply a CTBR residual scaled by a non-negative dodge gate."""
        gate = float(np.clip(gate, 0.0, 1.0))
        gated_action = gate * np.asarray(action, dtype=np.float32)
        return self.apply_ctbr_delta(
            nominal_control,
            gated_action,
            delta_thrust_fraction=delta_thrust_fraction,
            delta_rate_limits=delta_rate_limits,
        )
