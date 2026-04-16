"""Habitat-runtime integration tests for dynamic obstacles.

These tests intentionally exercise the real HabitatWrapper +
SynchronousSimulator path (not fakes/mocks) so obstacle spawning behavior is
validated end-to-end.
"""

import copy

import pytest

from neurosim.sims.synchronous_simulator.simulator import SynchronousSimulator

# Skip the entire module if Habitat runtime is unavailable.
pytest.importorskip("habitat_sim")


def _base_settings() -> dict:
    """Minimal simulator settings for real Habitat runtime tests."""
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


def _pick_runtime_template_handle(sim: SynchronousSimulator) -> str:
    """Pick a valid runtime object template handle from Habitat."""
    otm = sim.visual_backend._sim.get_object_template_manager()
    handles = list(otm.get_template_handles())
    if not handles:
        raise RuntimeError("No object templates available in Habitat runtime")

    preferred_tokens = ("cube", "sphere", "capsule")
    for token in preferred_tokens:
        for handle in handles:
            if token in handle.lower():
                return handle

    return handles[0]


def _active_obstacle_count(sim: SynchronousSimulator) -> int:
    """Return the number of active dynamic obstacles."""
    return len(sim.visual_backend._dynamic_obstacles._active)


def _enable_dynamic_obstacles(
    settings: dict,
    *,
    handle: str,
    motion_mode: str,
    ttl_s: float,
    spawn_interval_s: float,
    max_concurrent: int,
) -> dict:
    cfg = copy.deepcopy(settings)
    cfg["visual_backend"]["dynamic_obstacles"] = {
        "enabled": True,
        "spawn_interval_s": spawn_interval_s,
        "max_concurrent": max_concurrent,
        "throw_speed_range_mps": [4.0, 6.0],
        "angular_speed_range_radps": [0.0, 1.0],
        "azimuth_range_deg": [-45.0, 45.0],
        "radial_distance_range_m": [1.5, 2.5],
        "relative_height_range_m": [0.0, 1.0],
        "aim_noise_std_m": 0.05,
        "seed": 7,
        "templates": [
            {
                "handle": handle,
                "motion_mode": motion_mode,
                "ttl_s": ttl_s,
                "kinematic_speed_mps": 4.0,
                "parabola_gravity_mps2": 9.81,
            }
        ],
    }
    return cfg


def _default_control(sim: SynchronousSimulator) -> dict:
    # RotorpyDynamics provides this helper and it matches configured abstraction.
    return sim.dynamics._default_control()


def test_dynamic_obstacles_spawn_in_real_habitat_runtime():
    base = _base_settings()
    sim = SynchronousSimulator(base, visualizer_disabled=True)
    try:
        handle = _pick_runtime_template_handle(sim)
        sim.reconfigure(
            _enable_dynamic_obstacles(
                base,
                handle=handle,
                motion_mode="dynamic_throw",
                ttl_s=1.0,
                spawn_interval_s=0.08,
                max_concurrent=3,
            )
        )

        counts = []
        for _ in range(25):
            sim.step(_default_control(sim))
            counts.append(_active_obstacle_count(sim))

        assert max(counts) > 0, "No dynamic obstacle was spawned in runtime"
    finally:
        sim.close()


def test_dynamic_obstacles_ttl_despawns_in_real_habitat_runtime():
    base = _base_settings()
    sim = SynchronousSimulator(base, visualizer_disabled=True)
    try:
        handle = _pick_runtime_template_handle(sim)
        sim.reconfigure(
            _enable_dynamic_obstacles(
                base,
                handle=handle,
                motion_mode="kinematic_line",
                ttl_s=0.15,
                spawn_interval_s=10.0,
                max_concurrent=1,
            )
        )

        # First step should spawn one obstacle (interval check from -inf).
        sim.step(_default_control(sim))
        assert _active_obstacle_count(sim) == 1

        # Advance beyond TTL and ensure obstacle is removed.
        for _ in range(40):
            sim.step(_default_control(sim))

        assert _active_obstacle_count(sim) == 0
    finally:
        sim.close()


def test_dynamic_obstacles_reconfigure_rebinds_with_valid_handle():
    base = _base_settings()
    sim = SynchronousSimulator(base, visualizer_disabled=True)
    try:
        handle = _pick_runtime_template_handle(sim)
        enabled = _enable_dynamic_obstacles(
            base,
            handle=handle,
            motion_mode="kinematic_parabola",
            ttl_s=0.4,
            spawn_interval_s=0.05,
            max_concurrent=2,
        )
        sim.reconfigure(enabled)

        for _ in range(10):
            sim.step(_default_control(sim))
        assert _active_obstacle_count(sim) > 0

        # Reconfigure should reset time/sim state and clear old runtime obstacle ids.
        sim.reconfigure(enabled)
        assert _active_obstacle_count(sim) == 0

        # Obstacles should spawn again under the reconfigured runtime.
        for _ in range(10):
            sim.step(_default_control(sim))

        assert _active_obstacle_count(sim) > 0
    finally:
        sim.close()
