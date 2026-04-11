"""Vehicle builders for neurosim RL."""

from typing import Any

from .base import RLVehicle
from .ctbr_rotorpy import RateLimits, RotorpyCtbrVehicle


def build_vehicle(*, sim: Any, dynamics_config: dict[str, Any]) -> RLVehicle:
    abstraction = str(dynamics_config["control_abstraction"]).strip().lower()
    if abstraction != "cmd_ctbr":
        raise ValueError(
            "Unsupported control_abstraction for RL CTBR vehicle; "
            f"expected cmd_ctbr, got {dynamics_config['control_abstraction']!r}"
        )

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


__all__ = ["RLVehicle", "RateLimits", "RotorpyCtbrVehicle", "build_vehicle"]
