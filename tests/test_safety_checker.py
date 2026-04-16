"""Tests for HabitatSafetyChecker against real Habitat runtime.

Covers the three main safety checks produced by ``check()``:
  1. out_of_bounds
  2. not_navigable
  3. obstacle_collision
Plus basic sanity (init, safe point, sample_habitat_start).
"""

import copy
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("habitat_sim", reason="habitat_sim not available")

from neurosim.core.visual_backend.dynamic_obstacles import ActiveObstacle
from neurosim.rl.safety import HabitatSafetyChecker
from neurosim.sims.synchronous_simulator.simulator import SynchronousSimulator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_settings() -> dict:
    return {
        "simulator": {
            "world_rate": 100,
            "control_rate": 20,
            "sim_time": 2.0,
            "coord_transform": "rotorpy_to_hm3d",
            "sensor_rates": {},
            "viz_rates": {},
        },
        "visual_backend": {
            "gpu_id": 0,
            "scene": "data/scene_datasets/habitat-test-scenes/apartment_1.glb",
            "scene_dataset_config_file": "default",
            "clear_color": [0.0, 0.0, 0.0, 1.0],
            "default_agent": 0,
            "agent_height": 1.0,
            "agent_radius": 0.3,
            "agent_max_climb": 1.0,
            "agent_max_slope": 90.0,
            "enable_hbao": False,
            "frustum_culling": True,
            "seed": 324,
            "physics_config_file": "data/default.physics_config.json",
            "enable_physics": True,
            "sensors": {},
            "dynamic_obstacles": {
                "enabled": False,
                "spawn_interval_s": 0.1,
                "max_concurrent": 2,
                "throw_speed_range_mps": [4.0, 6.0],
                "angular_speed_range_radps": [0.0, 1.0],
                "radial_distance_range_m": [1.5, 2.5],
                "relative_height_range_m": [0.0, 1.0],
                "aim_noise_std_m": 0.05,
                "templates": [],
            },
        },
        "dynamics": {
            "model": "rotorpy_multirotor_euler",
            "vehicle": "crazyflie",
            "control_abstraction": "cmd_ctbr",
        },
    }


def _to_dynamics(checker, habitat_pos):
    """Convert a Habitat-space position back to dynamics space."""
    return np.linalg.solve(
        checker._pos_transform, np.asarray(habitat_pos, dtype=np.float64)
    )


# ---------------------------------------------------------------------------
# Module-scoped sim — shared by tests that don't mutate obstacle state.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sim():
    s = SynchronousSimulator(_base_settings(), visualizer_disabled=True)
    yield s
    s.close()


@pytest.fixture(scope="module")
def checker(sim):
    return HabitatSafetyChecker(sim, enable_navigable_check=True)


# ===========================================================================
# Sanity
# ===========================================================================


def test_init_bounds_are_non_degenerate(checker):
    assert checker._hi_x > checker._lo_x
    assert checker._hi_y > checker._lo_y
    assert checker._hi_z > checker._lo_z


def test_sample_habitat_start_returns_in_bounds_point(checker):
    pt = checker.sample_habitat_start()
    assert pt.shape == (3,)
    assert checker.is_in_bounds(pt)


def test_check_safe_point(checker):
    """A sampled navigable point (no obstacles) should pass all checks."""
    nav_pt = checker.sample_habitat_start()
    ok, reason = checker.check(_to_dynamics(checker, nav_pt))
    assert ok is True
    assert reason == ""


# ===========================================================================
# out_of_bounds
# ===========================================================================


def test_check_out_of_bounds(checker):
    far_habitat = np.array([9999.0, 9999.0, 9999.0])
    ok, reason = checker.check(_to_dynamics(checker, far_habitat))
    assert ok is False
    assert reason == "out_of_bounds"


# ===========================================================================
# not_navigable
# ===========================================================================


def test_check_not_navigable(checker):
    """A point near the ceiling is in bounds but not on the navmesh."""
    mid_x = (checker._lo_x + checker._hi_x) / 2
    mid_z = (checker._lo_z + checker._hi_z) / 2
    ceiling_y = checker._hi_y - 0.01
    ceiling_pos = np.array([mid_x, ceiling_y, mid_z])

    assert checker.is_in_bounds(ceiling_pos), "sanity: point should be in bounds"

    ok, reason = checker.check(_to_dynamics(checker, ceiling_pos))
    assert ok is False
    assert reason == "not_navigable"


# ===========================================================================
# obstacle_collision
# ===========================================================================


def _inject_fake_obstacle(sim, position, collision_radius=0.5, obj_id=9999):
    """Place a fake obstacle into the manager's active dict."""
    fake_obj = MagicMock()
    fake_obj.translation = np.asarray(position, dtype=np.float32)
    sim.visual_backend._dynamic_obstacles._active[obj_id] = ActiveObstacle(
        object_id=obj_id,
        obj=fake_obj,
        born_time=0.0,
        motion_mode="kinematic_line",
        ttl_s=100.0,
        spawn_position=np.asarray(position, dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        gravity_mps2=9.81,
        collision_radius=float(collision_radius),
    )


def test_check_obstacle_collision():
    """Inject a fake obstacle at a navigable point; check() must return obstacle_collision."""
    sim = SynchronousSimulator(_base_settings(), visualizer_disabled=True)
    try:
        # Navigable check disabled so the test isolates the collision branch.
        checker = HabitatSafetyChecker(sim, enable_navigable_check=False)
        nav_pt = checker.sample_habitat_start()

        _inject_fake_obstacle(sim, nav_pt, collision_radius=0.5)

        ok, reason = checker.check(_to_dynamics(checker, nav_pt))
        assert ok is False
        assert reason == "obstacle_collision"
    finally:
        sim.visual_backend._dynamic_obstacles._active.pop(9999, None)
        sim.close()


def test_check_obstacle_collision_with_spawned_obstacle():
    """Step the sim until a real obstacle spawns, then verify the checker detects it."""
    base = _base_settings()
    sim = SynchronousSimulator(base, visualizer_disabled=True)
    try:
        # Pick a real template handle available at runtime.
        otm = sim.visual_backend._sim.get_object_template_manager()
        handles = list(otm.get_template_handles())
        if not handles:
            pytest.skip("No object templates available in Habitat runtime")
        handle = handles[0]
        for token in ("cube", "sphere", "capsule"):
            for h in handles:
                if token in h.lower():
                    handle = h
                    break

        cfg = copy.deepcopy(base)
        cfg["visual_backend"]["dynamic_obstacles"] = {
            "enabled": True,
            "spawn_interval_s": 0.05,
            "max_concurrent": 3,
            "throw_speed_range_mps": [4.0, 6.0],
            "angular_speed_range_radps": [0.0, 1.0],
            "radial_distance_range_m": [1.5, 2.5],
            "relative_height_range_m": [0.0, 1.0],
            "aim_noise_std_m": 0.05,
            "seed": 42,
            "templates": [
                {
                    "handle": handle,
                    "motion_mode": "kinematic_line",
                    "ttl_s": 5.0,
                    "kinematic_speed_mps": 4.0,
                }
            ],
        }
        sim.reconfigure(cfg)

        checker = HabitatSafetyChecker(sim, enable_navigable_check=False)
        ctrl = sim.dynamics._default_control()

        for _ in range(30):
            sim.step(ctrl)
            if sim.visual_backend._dynamic_obstacles._active:
                break

        active = sim.visual_backend._dynamic_obstacles._active
        assert len(active) > 0, "No obstacles spawned after 30 steps"

        item = next(iter(active.values()))
        obstacle_pos = np.asarray(item.obj.translation, dtype=np.float32)

        # Query at the obstacle's own position — must collide.
        assert checker.has_obstacle_collision(obstacle_pos) is True

        # Query far away — no collision.
        far = obstacle_pos + np.array([1000.0, 0.0, 0.0], dtype=np.float32)
        assert checker.has_obstacle_collision(far) is False
    finally:
        sim.close()
