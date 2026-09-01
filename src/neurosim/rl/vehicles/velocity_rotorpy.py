"""Velocity-command vehicle: the policy shapes a world-frame velocity.

Chosen over the CTBR/SE3 position-tracking path because the latter can lose
the vehicle outright. ``SE3Control`` builds

    F_des = m * (-kp_pos * (x - x_ref) - kd_pos * (v - v_ref) + a_ref + g)

and then takes ``b3_des = normalize(F_des)`` with nothing keeping the
vehicle upright. With ``kp_z = 15`` an altitude error of ~0.65 m alone
cancels gravity, ``F_des`` rotates below horizontal, the controller
commands an inverted attitude, thrust projects to ~zero and the vehicle
free-falls without recovering. Measured in ~10-15% of episodes with *zero*
policy action.

``cmd_vel`` has no position term:

    F_des = m * (-k_v * (v - cmd_v) + g)

so tipping needs a velocity error beyond ``g / k_v`` (~1 m/s downward),
which the commanded velocity is clamped well inside.
"""

from typing import Any

import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from .base import RLVehicle


class RotorpyVelocityVehicle(RLVehicle):
    """Passes a world-frame velocity command through to rotorpy's ``cmd_vel``.

    Deliberately thin: there is no thrust/rate allocation to clip here, and
    the action semantics live in the task, so this only owns the speed limit
    and the optional dynamics randomization.
    """

    def __init__(
        self,
        dynamics: Any,
        max_speed_mps: float = 3.0,
        max_acceleration_mps2: float = 0.0,
        domain_randomization: dict[str, Any] | None = None,
    ):
        self._dynamics = dynamics
        self._multirotor = dynamics._multirotor
        self._max_speed = float(max_speed_mps)
        self._max_acceleration = float(max_acceleration_mps2 or 0.0)
        if self._max_acceleration < 0.0:
            raise ValueError("max_acceleration_mps2 must be non-negative")
        self._domain_randomization = domain_randomization or {}
        self._action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self._base_dynamic_params = {
            key: getattr(self._multirotor, key)
            for key in self._domain_randomization.get("scales", {})
            if hasattr(self._multirotor, key)
        }

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    @property
    def max_speed(self) -> float:
        return self._max_speed

    def randomize(self, episode_count: int, rng: np.random.Generator) -> None:
        cfg = self._domain_randomization
        if not cfg.get("enabled", False) or not self._base_dynamic_params:
            return
        every = max(int(cfg.get("resample_every", 1)), 1)
        if episode_count % every:
            return
        scales = cfg["scales"]
        for key, base in self._base_dynamic_params.items():
            low, high = scales[key]
            setattr(self._multirotor, key, float(base) * float(rng.uniform(low, high)))

    def clip_control(
        self, control: dict[str, np.ndarray | float]
    ) -> dict[str, np.ndarray | float]:
        """Clamp commanded speed. This is the whole safety story for cmd_vel.

        Keeping the command inside the vehicle's achievable envelope keeps
        the velocity error, and therefore ``F_des``, well away from the
        orientation at which thrust would stop opposing gravity.
        """
        cmd = np.asarray(control["cmd_v"], dtype=np.float64).reshape(3)
        speed = float(np.linalg.norm(cmd))
        if speed > self._max_speed:
            cmd = cmd * (self._max_speed / speed)
        if self._max_acceleration > 0.0:
            current = np.asarray(self._dynamics.state["v"], dtype=np.float64)
            gain = np.asarray(self._multirotor.k_v, dtype=np.float64)
            desired_acceleration = gain * (cmd - current)
            acceleration_norm = float(np.linalg.norm(desired_acceleration))
            if acceleration_norm > self._max_acceleration:
                desired_acceleration *= self._max_acceleration / acceleration_norm
                cmd = current + desired_acceleration / gain
        merged = dict(control)
        merged["cmd_v"] = cmd
        # Direct vehicle users do not have a trajectory yaw to provide. Hold
        # their current heading instead of falling back to RotorPy's implicit
        # world-+X reference. ReactiveDodgeEnv supplies nominal yaw explicitly.
        merged.setdefault(
            "cmd_yaw",
            float(
                Rotation.from_quat(self._dynamics.state["q"]).as_euler("xyz")[2]
            ),
        )
        return merged

    def action_to_control(self, action: np.ndarray) -> dict[str, np.ndarray | float]:
        """Normalized action straight to a world-frame velocity command."""
        cmd = np.clip(np.asarray(action, dtype=np.float64).reshape(3), -1.0, 1.0)
        return self.clip_control({"cmd_v": cmd * self._max_speed})

    def control_to_normalized(
        self, control: dict[str, np.ndarray | float]
    ) -> np.ndarray:
        """Inverse of :meth:`action_to_control`, for the nominal-action channel."""
        cmd = np.asarray(control["cmd_v"], dtype=np.float64).reshape(3)
        return np.clip(cmd / self._max_speed, -1.0, 1.0).astype(np.float32)
