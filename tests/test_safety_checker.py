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
from neurosim.core.visual_backend.safety import HabitatSafetyChecker
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
                "azimuth_range_deg": [-45.0, 45.0],
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
    # Active obstacle translations are stored with visual-backend agent-height offset.
    agent_height = float(sim.visual_backend.settings["agent_height"])
    pos = np.asarray(position, dtype=np.float32).copy()
    pos[1] += agent_height
    fake_obj.translation = pos
    sim.visual_backend._dynamic_obstacles._active[obj_id] = ActiveObstacle(
        object_id=obj_id,
        obj=fake_obj,
        born_time=0.0,
        motion_mode="kinematic_line",
        ttl_s=100.0,
        spawn_position=pos,
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
            "azimuth_range_deg": [-45.0, 45.0],
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

        assert sim.visual_backend._dynamic_obstacles._agent_height == pytest.approx(
            cfg["visual_backend"]["agent_height"]
        )
        assert sim.visual_backend._dynamic_obstacles._agent_radius == pytest.approx(
            cfg["visual_backend"]["agent_radius"]
        )

        checker = HabitatSafetyChecker(sim, enable_navigable_check=False)
        ctrl = sim.dynamics._default_control()

        for _ in range(30):
            sim.step(ctrl)
            if sim.visual_backend._dynamic_obstacles._active:
                break

        active = sim.visual_backend._dynamic_obstacles._active
        assert len(active) > 0, "No obstacles spawned after 30 steps"

        item = next(iter(active.values()))
        obstacle_pos = np.asarray(item.obj.translation, dtype=np.float32).copy()
        obstacle_pos[1] -= float(cfg["visual_backend"]["agent_height"])

        # Query in floor-referenced Habitat coordinates — must collide.
        assert checker.has_obstacle_collision(obstacle_pos) is True

        # Query far away — no collision.
        far = obstacle_pos + np.array([1000.0, 0.0, 0.0], dtype=np.float32)
        assert checker.has_obstacle_collision(far) is False
    finally:
        sim.close()


# ---------------------------------------------------------------------------
# The roof-and-ground check that keeps the drone out of the garden
# ---------------------------------------------------------------------------
def _checker_with_rays(hits_up: bool, hits_down: bool) -> HabitatSafetyChecker:
    """A checker whose raycasts answer as told, so the logic is testable off-scene."""
    checker = HabitatSafetyChecker.__new__(HabitatSafetyChecker)
    checker._enable_sky = True
    checker._sky_probe_m = 4.0

    def cast_ray(ray, max_distance):
        result = MagicMock()
        # Habitat is Y up, so the sign of the ray's y component says which probe this is.
        result.has_hits.return_value = hits_up if ray.direction.y > 0 else hits_down
        return result

    checker._sim = MagicMock()
    checker._sim.cast_ray.side_effect = cast_ray

    # Enough of the rest for check() to run: identity frames, roomy bounds, no obstacles.
    checker._pos_transform = np.eye(3)
    checker._lo_x = checker._lo_y = checker._lo_z = -10.0
    checker._hi_x = checker._hi_y = checker._hi_z = 10.0
    checker._enable_navigable = False
    checker._dynamic_obstacles = None
    return checker


def test_a_point_with_roof_and_ground_is_indoors():
    assert _checker_with_rays(hits_up=True, hits_down=True).is_indoors(np.zeros(3))


def test_open_sky_overhead_is_not_indoors():
    """A yard: floor underneath, nothing above."""
    assert not _checker_with_rays(hits_up=False, hits_down=True).is_indoors(np.zeros(3))


def test_nothing_underneath_is_not_indoors():
    """A balcony edge or a ledge over a void."""
    assert not _checker_with_rays(hits_up=True, hits_down=False).is_indoors(np.zeros(3))


def test_the_sky_check_can_be_switched_off():
    """Outdoor scenes should still be usable when someone asks for them."""
    checker = _checker_with_rays(hits_up=False, hits_down=False)
    checker._enable_sky = False
    assert checker.is_indoors(np.zeros(3))


def test_check_reports_open_sky_as_not_indoors():
    """Its own reason, so a yard is distinguishable from leaving the scene."""
    safe, reason = _checker_with_rays(hits_up=False, hits_down=True).check(np.zeros(3))
    assert (safe, reason) == (False, "not_indoors")


def test_check_still_reports_leaving_the_scene_as_out_of_bounds():
    """The two must not be conflated: one is a garden, the other is off the map."""
    checker = _checker_with_rays(hits_up=True, hits_down=True)
    safe, reason = checker.check(np.array([999.0, 0.0, 0.0]))
    assert (safe, reason) == (False, "out_of_bounds")


def test_check_passes_a_point_indoors_and_in_bounds():
    safe, reason = _checker_with_rays(hits_up=True, hits_down=True).check(np.zeros(3))
    assert (safe, reason) == (True, "")
