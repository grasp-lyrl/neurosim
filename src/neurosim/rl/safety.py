"""Safety checking for RL environments using Habitat pathfinder.

Safety policy in this module:
- In-bounds check against full 3D pathfinder bounds.
- Optional direct `pathfinder.is_navigable(point)` check.
"""

from typing import Any

import numpy as np


class HabitatSafetyChecker:
    """Habitat-backed safety checker for a flying drone.

    Args:
        pathfinder: ``habitat_sim`` pathfinder object (must be loaded).
        pos_transform: 3×3 dynamics-to-Habitat position transform matrix.
        enable_navigable_check: If False, skip direct navigability checks.
    """

    def __init__(
        self,
        pathfinder: Any,
        pos_transform: np.ndarray,
        enable_navigable_check: bool = True,
    ):
        self._pathfinder = pathfinder
        self._pos_transform = np.asarray(pos_transform, dtype=np.float64)
        self._enable_navigable = enable_navigable_check

        # Full 3D bounds in Habitat coordinates.
        lo, hi = pathfinder.get_bounds()
        self._lo_x = float(lo[0])
        self._hi_x = float(hi[0])
        self._lo_y = float(lo[1])
        self._hi_y = float(hi[1])
        self._lo_z = float(lo[2])
        self._hi_z = float(hi[2])

    def dynamics_to_habitat(self, x: np.ndarray) -> np.ndarray:
        return self._pos_transform @ np.asarray(x, dtype=np.float64)

    def is_in_bounds(self, habitat_pos: np.ndarray) -> bool:
        hx, hy, hz = habitat_pos
        return (
            self._lo_x <= hx <= self._hi_x
            and self._lo_y <= hy <= self._hi_y
            and self._lo_z <= hz <= self._hi_z
        )

    def is_navigable(self, habitat_pos: np.ndarray) -> bool:
        """Check direct navigability at the provided Habitat-space point."""
        if not self._enable_navigable:
            return True
        return bool(
            self._pathfinder.is_navigable(np.asarray(habitat_pos, dtype=np.float64))
        )

    def check(self, dynamics_position: np.ndarray) -> tuple[bool, str]:
        hp = self.dynamics_to_habitat(dynamics_position)
        if not self.is_in_bounds(hp):
            return False, "out_of_bounds"
        if not self.is_navigable(hp):
            return False, "not_navigable"
        return True, ""

    def sample_habitat_start(self, max_tries: int = 200) -> np.ndarray:
        """Sample a random valid starting position in Habitat space."""
        for _ in range(max_tries):
            nav_pt = np.array(
                self._pathfinder.get_random_navigable_point(), dtype=np.float64
            )
            ok, _ = self.check(np.linalg.solve(self._pos_transform, nav_pt))
            if ok:
                return nav_pt
        raise RuntimeError(
            f"Could not sample a safe starting position in {max_tries} tries. "
            "Check scene bounds and pathfinder navigability."
        )


def build_safety_checker(
    sim: Any,
    *,
    enable_navigable_check: bool = True,
) -> HabitatSafetyChecker:
    """Build a HabitatSafetyChecker from a SynchronousSimulator."""
    pathfinder = sim.visual_backend._sim.pathfinder
    if pathfinder is None or not pathfinder.is_loaded:
        raise RuntimeError("HabitatSafetyChecker requires a loaded Habitat pathfinder.")

    return HabitatSafetyChecker(
        pathfinder=pathfinder,
        pos_transform=sim.coord_trans.pos_transform,
        enable_navigable_check=enable_navigable_check,
    )
