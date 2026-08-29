"""Habitat-runtime integration tests for dynamic obstacles.

These tests intentionally exercise the real HabitatWrapper +
SynchronousSimulator path (not fakes/mocks) so obstacle spawning behavior is
validated end-to-end.
"""

import copy

import numpy as np
import pytest

from neurosim.core.visual_backend.dynamic_obstacles import (
    DynamicObstacleManager,
    DynamicObstaclesConfig,
)
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


def test_reset_episode_allows_spawning_again_after_the_clock_restarts():
    """Obstacles must keep spawning in every episode, not just the first.

    Spawn/TTL timers are absolute ``sim_time`` values while the episode
    clock restarts at 0. Without ``reset_episode`` the spawn test
    ``sim_time - last_spawn >= interval`` stays false for the whole of
    every later episode, so obstacles appear once and never again -- which
    makes any encounter-gated success metric silently unreachable.
    """
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
        manager = sim.visual_backend._dynamic_obstacles

        spawns_per_episode = []
        for _ in range(2):
            # Restart the episode clock the way BaseNeurosimRLEnv.reset does.
            sim.time = 0.0
            sim.simsteps = 0
            manager.reset_episode(seed=123)

            seen = set()
            for _ in range(25):
                sim.step(_default_control(sim))
                seen |= set(manager._active.keys())
            spawns_per_episode.append(len(seen))

        assert all(n > 0 for n in spawns_per_episode), (
            f"obstacles stopped spawning after the first episode: {spawns_per_episode}"
        )
    finally:
        sim.close()


def test_explicit_collision_radius_overrides_the_aabb_estimate():
    """An explicit radius must win over the derived bounding sphere.

    The AABB estimate is the corner distance, overestimating a rounded
    obstacle by up to sqrt(3), and it silently sets the clearance a dodge
    has to achieve -- so the task needs to be able to state it directly.
    """
    base = _base_settings()
    sim = SynchronousSimulator(base, visualizer_disabled=True)
    try:
        handle = _pick_runtime_template_handle(sim)
        settings = _enable_dynamic_obstacles(
            base,
            handle=handle,
            motion_mode="dynamic_throw",
            ttl_s=5.0,
            spawn_interval_s=0.08,
            max_concurrent=2,
        )
        for template in settings["visual_backend"]["dynamic_obstacles"]["templates"]:
            template["collision_radius"] = 0.1
        sim.reconfigure(settings)

        manager = sim.visual_backend._dynamic_obstacles
        for _ in range(15):
            sim.step(_default_control(sim))

        assert manager._active, "expected at least one obstacle to spawn"
        radii = [item.collision_radius for item in manager._active.values()]
        assert radii == pytest.approx([0.1] * len(radii)), radii
    finally:
        sim.close()


def test_reset_episode_clears_active_obstacles_and_timer():
    base = _base_settings()
    sim = SynchronousSimulator(base, visualizer_disabled=True)
    try:
        handle = _pick_runtime_template_handle(sim)
        sim.reconfigure(
            _enable_dynamic_obstacles(
                base,
                handle=handle,
                motion_mode="dynamic_throw",
                ttl_s=5.0,
                spawn_interval_s=0.08,
                max_concurrent=3,
            )
        )
        manager = sim.visual_backend._dynamic_obstacles
        for _ in range(15):
            sim.step(_default_control(sim))
        assert manager._active, "expected obstacles to be active before reset"

        manager.reset_episode(seed=7)

        assert manager._active == {}
        assert manager._last_spawn_time == -float("inf")
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


def test_obstacle_spawn_candidates_have_line_of_sight_to_drone():
    """A spawn candidate blocked by scene geometry must be rejected.

    Without this, the organic spawner (a pure geometric offset from the
    drone with no scene-mesh awareness) can place an obstacle behind a
    wall or in an adjacent room, and it then flies straight through that
    wall to reach the drone.
    """
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
        dyn = sim.visual_backend._dynamic_obstacles
        # Wide enough that, without the line-of-sight check, many candidates
        # in apartment_1 would land behind a wall or in another room.
        dyn.cfg.radial_distance_range_m = (2.0, 15.0)
        dyn.cfg.azimuth_range_deg = (-180.0, 180.0)

        # A navigable interior point, not wherever the default (possibly
        # out-of-mesh) agent state happens to be, so the wide radius above
        # reliably reaches real walls in every direction.
        drone_position = np.asarray(
            sim.visual_backend._sim.pathfinder.get_random_navigable_point(),
            dtype=np.float32,
        )
        identity_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        results = [
            dyn._sample_spawn_position(drone_position, identity_quat)
            for _ in range(60)
        ]
        accepted = [r for r in results if r is not None]
        rejected = [r for r in results if r is None]

        assert accepted, "line-of-sight check rejected every candidate"
        assert rejected, (
            "expected some wide-radius candidates in apartment_1 to be "
            "blocked by walls -- if none were, this test isn't exercising "
            "the rejection path"
        )
        for pos in accepted:
            assert dyn._has_line_of_sight(pos, drone_position), (
                f"accepted spawn position {pos} does not actually have "
                "line-of-sight to the drone"
            )
    finally:
        sim.close()


def test_aim_override_makes_spawns_independent_of_actual_drone_position():
    """With an aim override, the same seed yields the same encounter.

    The organic spawner aims at the live drone position, so two policies
    that fly differently see different obstacles from the same seed --
    which makes success rates from different checkpoints not comparable.
    Overriding the aim with a seed-determined reference position removes
    that coupling.
    """
    base = _base_settings()
    sim = SynchronousSimulator(base, visualizer_disabled=True)
    try:
        handle = _pick_runtime_template_handle(sim)
        sim.reconfigure(
            _enable_dynamic_obstacles(
                base,
                handle=handle,
                motion_mode="kinematic_line",
                ttl_s=5.0,
                spawn_interval_s=0.05,
                max_concurrent=1,
            )
        )
        dyn = sim.visual_backend._dynamic_obstacles
        identity_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        def spawn_with(drone_position, reference):
            dyn.reset_episode(seed=1234)
            dyn.set_aim_override(reference)
            dyn.step(1.0, np.asarray(drone_position, dtype=np.float64), identity_quat)
            if not dyn._active:
                return None
            item = next(iter(dyn._active.values()))
            return (
                np.asarray(item.spawn_position, dtype=np.float64),
                np.asarray(item.velocity, dtype=np.float64),
            )

        # Open enough that the line-of-sight spawn check finds a candidate.
        pathfinder = sim.visual_backend._sim.pathfinder
        for _ in range(25):
            reference = np.asarray(
                pathfinder.get_random_navigable_point(), dtype=np.float64
            )
            if spawn_with(reference, reference) is not None:
                break
        else:
            pytest.skip("no navigable point with a spawnable neighbourhood found")

        drone_a = reference + np.array([1.5, 0.0, -1.0])
        drone_b = reference + np.array([-2.0, 0.5, 2.5])

        # Two very different "actual drone" positions, same aim override.
        pos_a, vel_a = spawn_with(drone_a, reference)
        pos_b, vel_b = spawn_with(drone_b, reference)

        np.testing.assert_allclose(pos_a, pos_b, atol=1e-6)
        np.testing.assert_allclose(vel_a, vel_b, atol=1e-6)

        # Without the override the same two drone positions must diverge,
        # otherwise this test would pass even if the override did nothing.
        def spawn_without(drone_position):
            dyn.reset_episode(seed=1234)
            dyn.clear_aim_override()
            dyn.step(1.0, np.asarray(drone_position, dtype=np.float64), identity_quat)
            if not dyn._active:
                return None
            item = next(iter(dyn._active.values()))
            return np.asarray(item.spawn_position, dtype=np.float64)

        free_a = spawn_without(drone_a)
        free_b = spawn_without(drone_b)
        assert free_a is None or free_b is None or not np.allclose(
            free_a, free_b, atol=1e-6
        ), "spawns matched without the override -- test cannot detect its effect"
    finally:
        sim.close()


def test_intercept_time_solves_for_a_moving_target():
    """The lead solve must actually meet the drone, not trail it."""
    mgr = DynamicObstacleManager.__new__(DynamicObstacleManager)
    spawn = np.array([0.0, 0.0, 0.0])
    drone = np.array([10.0, 0.0, 0.0])
    vel = np.array([0.0, 0.0, 2.0])
    speed = 5.0

    t = mgr._intercept_time(spawn, drone, vel, speed)
    assert t is not None
    meet = drone + vel * t
    # The projectile covers exactly speed*t getting there.
    assert np.linalg.norm(meet - spawn) == pytest.approx(speed * t, rel=1e-6)
    # And it leads: the aim point is displaced along the drone's heading.
    assert meet[2] > drone[2]

    # A drone outrunning the throw has no intercept; the caller falls back
    # to aiming at the current position rather than extrapolating wildly.
    assert mgr._intercept_time(spawn, drone, np.array([0.0, 0.0, 50.0]), 5.0) is None

    # A stationary drone degenerates to the straight-line travel time.
    t_still = mgr._intercept_time(spawn, drone, np.zeros(3), speed)
    assert t_still == pytest.approx(10.0 / speed, rel=1e-6)


def test_lead_aim_probability_mixes_both_aiming_behaviours():
    """Neither aiming mode may dominate, or the policy can overfit to it."""
    base = _base_settings()
    sim = SynchronousSimulator(base, visualizer_disabled=True)
    try:
        handle = _pick_runtime_template_handle(sim)
        sim.reconfigure(
            _enable_dynamic_obstacles(
                base,
                handle=handle,
                motion_mode="kinematic_line",
                ttl_s=5.0,
                spawn_interval_s=0.05,
                max_concurrent=1,
            )
        )
        dyn = sim.visual_backend._dynamic_obstacles
        dyn.cfg.aim_noise_std_m = 0.0
        reference = np.asarray(
            sim.visual_backend._sim.pathfinder.get_random_navigable_point(),
            dtype=np.float64,
        )
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        velocity = np.array([0.0, 0.0, 3.0], dtype=np.float64)

        def spawn_headings(probability, trials=40):
            dyn.cfg.lead_aim_probability = probability
            headings = []
            for i in range(trials):
                dyn.reset_episode(seed=100 + i)
                # Two steps: the manager derives velocity from consecutive
                # positions, so the first call only primes it.
                dyn.step(0.0, reference - velocity * 0.1, quat)
                dyn.step(0.1, reference, quat)
                if dyn._active:
                    item = next(iter(dyn._active.values()))
                    headings.append(np.asarray(item.velocity, dtype=np.float64))
            return headings

        none_lead = spawn_headings(0.0)
        all_lead = spawn_headings(1.0)
        mixed = spawn_headings(0.5)
        assert none_lead and all_lead and mixed

        # Leading bends the throw along the drone's direction of travel, so
        # the two populations must be distinguishable at all.
        def mean_component(hs):
            return float(np.mean([h[2] / np.linalg.norm(h) for h in hs]))

        assert mean_component(all_lead) != pytest.approx(
            mean_component(none_lead), abs=1e-3
        )

        # And a 0.5 mix must actually produce some of each, not silently
        # collapse to one branch.
        lead_ref = mean_component(all_lead)
        plain_ref = mean_component(none_lead)
        per_throw = [h[2] / np.linalg.norm(h) for h in mixed]
        n_lead = sum(abs(v - lead_ref) < abs(v - plain_ref) for v in per_throw)
        assert 0 < n_lead < len(per_throw)
    finally:
        sim.close()


def test_spawn_interval_jitter_breaks_the_clock():
    """A fixed spawn period is a signal the policy can ride instead of seeing.

    Measured on a trained policy at a hard 2.5 s cadence: vehicle speed
    varied 78% of its mean with phase in the spawn cycle while commanded
    action magnitude varied only 2%. Nothing in the observation encodes
    time, so that structure could only come from the period itself.
    """
    cfg = DynamicObstaclesConfig.from_dict(
        {
            "enabled": True,
            "spawn_interval_s": 2.5,
            "spawn_interval_jitter": 0.6,
            "seed": 7,
            "templates": [{"handle": "x"}],
        }
    )
    mgr = DynamicObstacleManager.__new__(DynamicObstacleManager)
    mgr.cfg = cfg
    mgr._rng = np.random.default_rng(7)

    waits = np.array([mgr._sample_spawn_wait() for _ in range(400)])
    assert waits.min() >= 2.5 * 0.4 - 1e-9
    assert waits.max() <= 2.5 * 1.6 + 1e-9
    # Mean rate preserved, but the value is genuinely spread.
    assert abs(waits.mean() - 2.5) < 0.15
    assert waits.std() > 0.3
    assert len(np.unique(np.round(waits, 3))) > 100

    # Jitter off reproduces the old fixed cadence exactly.
    cfg.spawn_interval_jitter = 0.0
    fixed = np.array([mgr._sample_spawn_wait() for _ in range(20)])
    np.testing.assert_allclose(fixed, 2.5)


def test_aim_scatter_decouples_throws_from_the_drone():
    """Scattered throws remove position itself as a survival strategy.

    Throws are solved once at spawn and never re-aimed, so an aimed field
    lets a policy survive by drifting: measured threat fraction was 51.5%
    with a deterministic (still) policy against 20.8% with a stochastic
    (wandering) one, for the same obstacle field. If a throw is not aimed at
    you, moving does not make it miss.
    """
    from neurosim.core.visual_backend.dynamic_obstacles import (
        DynamicObstaclesConfig,
    )

    assert DynamicObstaclesConfig(enabled=False).aim_scatter_m == 0.0
    parsed = DynamicObstaclesConfig.from_dict(
        {"enabled": True, "aim_scatter_m": 3.5}
    )
    assert parsed.aim_scatter_m == pytest.approx(3.5)


def test_aim_scatter_is_uniform_inside_the_sphere():
    """Isotropic scatter, not concentrated on a shell.

    Sampling the radius uniformly would pile throws near the centre; scaling
    by the cube root spreads them through the volume, whose mean radius is
    3R/4.
    """
    rng = np.random.default_rng(0)
    radius_m = 4.0
    radii = np.array(
        [radius_m * float(rng.random()) ** (1.0 / 3.0) for _ in range(20000)]
    )
    assert radii.max() <= radius_m
    assert radii.mean() == pytest.approx(0.75 * radius_m, rel=0.02)


class _FakeObj:
    def __init__(self, translation):
        self.translation = np.asarray(translation, dtype=np.float64)
        self.object_id = 1


def _manager_with(commit_s, turn_rate=100.0):
    """A manager with just enough state for _reaim to run."""
    from neurosim.core.visual_backend.dynamic_obstacles import (
        DynamicObstacleManager,
        DynamicObstaclesConfig,
    )

    mgr = DynamicObstacleManager.__new__(DynamicObstacleManager)
    mgr.cfg = DynamicObstaclesConfig.from_dict(
        {
            "enabled": True,
            "reaim_commit_time_s": commit_s,
            "reaim_max_turn_rate_radps": turn_rate,
        }
    )
    mgr._active = {}
    mgr._previous_drone_time = 0.0
    return mgr


def _throw(position, velocity, aim_offset=None):
    from neurosim.core.visual_backend.dynamic_obstacles import ActiveObstacle

    return ActiveObstacle(
        object_id=1,
        obj=_FakeObj(position),
        born_time=0.0,
        motion_mode="kinematic_line",
        ttl_s=4.0,
        spawn_position=np.asarray(position, dtype=np.float64),
        velocity=np.asarray(velocity, dtype=np.float64),
        gravity_mps2=0.0,
        collision_radius=0.15,
        aim_offset=aim_offset,
    )


def test_reaim_steers_a_throw_back_toward_the_drone():
    """Drifting must stop paying: throws follow the drone in flight.

    Aimed-once throws couple threat to evadability -- escaping needs only to
    out-move the aim error, and measured drift (~1.6 m) already exceeds the
    1.13 m mean scatter, so wandering beats a tighter aim rather than losing
    to it.
    """
    mgr = _manager_with(commit_s=0.2)
    item = _throw([10.0, 0.0, 0.0], [-5.0, 0.0, 0.0])
    mgr._active = {1: item}
    # Drone has moved 3 m sideways since the throw was aimed.
    mgr._reaim(0.1, np.array([0.0, 3.0, 0.0]))

    assert item.velocity[1] > 0.0, "should steer toward the drone's new position"
    assert float(np.linalg.norm(item.velocity)) == pytest.approx(5.0, rel=1e-6)


def test_reaim_preserves_each_throws_own_miss_distance():
    """Re-aiming must not turn every obstacle into a homing missile."""
    mgr = _manager_with(commit_s=0.2)
    offset = np.array([0.0, 2.0, 0.0])
    item = _throw([10.0, 0.0, 0.0], [-5.0, 0.0, 0.0], aim_offset=offset)
    mgr._active = {1: item}
    mgr._reaim(0.1, np.zeros(3))

    # Aim point is drone + offset, so it steers toward y=+2, not y=0.
    direction = item.velocity / np.linalg.norm(item.velocity)
    to_aim = (offset - np.array([10.0, 0.0, 0.0]))
    to_aim = to_aim / np.linalg.norm(to_aim)
    assert float(np.dot(direction, to_aim)) > 0.99


def test_reaim_stops_at_the_commit_time():
    """Inside the commit window the throw is ballistic, so a late dodge works."""
    mgr = _manager_with(commit_s=0.5)
    # 1.12 m out at 5 m/s is 0.22 s to arrival -- inside the commit window.
    # The drone is offset enough that an uncommitted throw would visibly turn.
    item = _throw([1.0, 0.0, 0.0], [-5.0, 0.0, 0.0])
    mgr._active = {1: item}
    before = item.velocity.copy()
    mgr._reaim(0.1, np.array([0.0, 0.5, 0.0]))
    assert np.allclose(item.velocity, before)

    # Same geometry outside the window does turn, confirming the test is
    # measuring the commit rule rather than a throw that never steers.
    far = _throw([4.0, 0.0, 0.0], [-5.0, 0.0, 0.0])
    mgr._active = {1: far}
    mgr._reaim(0.1, np.array([0.0, 0.5, 0.0]))
    assert far.velocity[1] > 0.0


def test_reaim_is_off_by_default():
    mgr = _manager_with(commit_s=0.0)
    item = _throw([10.0, 0.0, 0.0], [-5.0, 0.0, 0.0])
    mgr._active = {1: item}
    before = item.velocity.copy()
    mgr._reaim(0.1, np.array([0.0, 3.0, 0.0]))
    assert np.allclose(item.velocity, before)


def test_reaim_turn_rate_limits_the_correction():
    """A slow turn rate must only partially close the angle in one step."""
    mgr = _manager_with(commit_s=0.2, turn_rate=0.5)
    item = _throw([10.0, 0.0, 0.0], [-5.0, 0.0, 0.0])
    mgr._active = {1: item}
    mgr._reaim(0.05, np.array([0.0, 10.0, 0.0]))
    direction = item.velocity / np.linalg.norm(item.velocity)
    # 0.5 rad/s over 0.05 s is 0.025 rad -- a small nudge, not a snap.
    assert 0.0 < direction[1] < 0.1


def test_reaimed_throws_still_travel():
    """Re-aiming must not freeze obstacles in place.

    _reaim re-bases spawn_position/born_time to the current instant, so if it
    runs before _update_kinematic the displacement integrates over t = 0 and
    the throw never moves. That measured as 0.0 encounters per episode with a
    field of 8 obstacles -- lethal-looking config, nothing ever arriving.
    """
    mgr = _manager_with(commit_s=0.2)
    item = _throw([10.0, 0.0, 0.0], [-5.0, 0.0, 0.0])
    mgr._active = {1: item}

    positions = []
    for k in range(1, 6):
        now = 0.05 * k
        # Same order step() uses: advance, then re-aim.
        mgr._update_kinematic(now)
        mgr._reaim(now, np.array([0.0, 0.0, 0.0]))
        mgr._previous_drone_time = now
        positions.append(float(item.obj.translation[0]))

    assert positions[-1] < positions[0], "throw must make progress toward the drone"
    travelled = positions[0] - positions[-1]
    assert travelled > 0.5, f"expected real displacement, got {travelled:.3f} m"
