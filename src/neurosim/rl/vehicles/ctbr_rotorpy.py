"""CTBR vehicle backed by RotorPy multirotor dynamics."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from gymnasium import spaces

from neurosim.core.dynamics.rotorpy_wrapper import get_vehicle_params

from .base import RLVehicle


@dataclass
class _RateLimits:
    roll: float
    pitch: float
    yaw: float


class RotorpyCtbrVehicle(RLVehicle):
    """CTBR vehicle for RotorPy multirotor dynamics."""

    def __init__(
        self,
        *,
        multirotor: Any,
        vehicle_name: str,
        rate_limits: _RateLimits,
        domain_randomization: dict[str, Any] | None = None,
    ):
        self._multirotor = multirotor
        self.vehicle_name = str(vehicle_name)
        self._rate_limits = rate_limits
        self._domain_randomization: dict[str, Any] | None = domain_randomization

        self.reference_vehicle_params = get_vehicle_params(self.vehicle_name)

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
        dr = self._domain_randomization
        if dr is None or not bool(dr.get("enabled", False)):
            return

        scales = dr["scales"]
        for key, base_value in self._base_dynamic_params.items():
            low, high = scales[key]
            sampled = float(base_value) * float(rng.uniform(float(low), float(high)))
            setattr(self._multirotor, key, sampled)

    def on_reset(self, rng: np.random.Generator) -> None:
        self._apply_domain_randomization(rng)
        self._refresh_from_multirotor()

    def action_to_control(self, action: np.ndarray) -> dict[str, np.ndarray | float]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self._action_space.low, self._action_space.high)

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
