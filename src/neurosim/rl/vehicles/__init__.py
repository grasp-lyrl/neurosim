"""Vehicle builders for neurosim RL."""

from typing import Any

from .base import RLVehicle
from .ctbr_rotorpy import RateLimits, RotorpyCtbrVehicle
from .velocity_rotorpy import RotorpyVelocityVehicle


def build_vehicle(*, sim: Any, dynamics_config: dict[str, Any]) -> RLVehicle:
    abstraction = str(dynamics_config["control_abstraction"]).strip().lower()
    if abstraction == "cmd_vel":
        # Velocity command: no position term in the force law, so the
        # controller cannot rotate the thrust axis past horizontal the way
        # the SE3 position path can (see velocity_rotorpy for the failure).
        return RotorpyVelocityVehicle(
            dynamics=sim.dynamics,
            max_speed_mps=float(dynamics_config.get("max_speed_mps", 3.0)),
            domain_randomization=dynamics_config.get("domain_randomization"),
        )
    if abstraction != "cmd_ctbr":
        raise ValueError(
            "Unsupported control_abstraction for RL vehicles; expected "
            f"cmd_ctbr or cmd_vel, got {dynamics_config['control_abstraction']!r}"
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


__all__ = [
    "RLVehicle",
    "RateLimits",
    "RotorpyCtbrVehicle",
    "RotorpyVelocityVehicle",
    "build_vehicle",
]
