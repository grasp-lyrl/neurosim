"""Vehicle builders for neurosim RL."""

from typing import Any

from .base import RLVehicle
from .ctbr_rotorpy import RateLimits, RotorpyCtbrVehicle
from .velocity_correction import VelocityCorrectionVehicle


def build_vehicle(*, sim: Any, dynamics_config: dict[str, Any]) -> RLVehicle:
    abstraction = str(dynamics_config["control_abstraction"]).strip().lower()

    if abstraction == "cmd_ctbr":
        rate_limits_cfg = dynamics_config["ctbr_rate_limits"]
        rate_limits = RateLimits(
            roll=float(rate_limits_cfg["roll"]),
            pitch=float(rate_limits_cfg["pitch"]),
            yaw=float(rate_limits_cfg["yaw"]),
        )
        return RotorpyCtbrVehicle(
            dynamics=sim.dynamics,
            vehicle=str(dynamics_config["vehicle"]),
            rate_limits=rate_limits,
            domain_randomization=dynamics_config.get("domain_randomization"),
        )

    if abstraction == "velocity_correction":
        return VelocityCorrectionVehicle(
            max_correction_mps=float(
                dynamics_config.get("max_correction_mps", 2.0)
            ),
        )

    raise ValueError(
        f"Unsupported control_abstraction: {dynamics_config['control_abstraction']!r}; "
        "expected 'cmd_ctbr' or 'velocity_correction'"
    )


__all__ = [
    "RLVehicle",
    "RateLimits",
    "RotorpyCtbrVehicle",
    "VelocityCorrectionVehicle",
    "build_vehicle",
]
