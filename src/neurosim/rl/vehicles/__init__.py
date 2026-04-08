"""Vehicle builders for neurosim RL."""

from typing import Any

from .base import RLVehicle
from .ctbr_rotorpy import RotorpyCtbrVehicle, _RateLimits


def build_vehicle(*, sim: Any, vehicle_config: dict[str, Any]) -> RLVehicle:
    control_mode = str(vehicle_config["control_mode"]).strip().lower()
    if control_mode != "ctbr":
        raise ValueError(
            f"Unsupported control_mode for now: {vehicle_config['control_mode']}"
        )

    rate_limits_cfg = vehicle_config["ctbr_rate_limits"]
    rate_limits = _RateLimits(
        roll=float(rate_limits_cfg["roll"]),
        pitch=float(rate_limits_cfg["pitch"]),
        yaw=float(rate_limits_cfg["yaw"]),
    )
    return RotorpyCtbrVehicle(
        multirotor=sim.dynamics._multirotor,
        vehicle_name=str(vehicle_config["vehicle_name"]),
        rate_limits=rate_limits,
        domain_randomization=vehicle_config["domain_randomization"],
    )


__all__ = ["RLVehicle", "RotorpyCtbrVehicle", "build_vehicle"]
