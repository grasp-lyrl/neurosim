"""Tests for the neurosim hover-stop RL environment.

All tests use the real Habitat backend (apartment_1 scene) since RL training
always runs with Habitat as the visual backend.
"""

import copy
from unittest.mock import Mock

import numpy as np
import pytest

from neurosim.rl import NeurosimRLEnv
from neurosim.rl.safety import HabitatSafetyChecker


def _test_env_config(
    *,
    obs_mode: str,
    episode_seconds: float,
    event_downsample_factor: int = 1,
) -> dict:
    return {
        "obs_mode": obs_mode,
        "episode_seconds": episode_seconds,
        "event_sensor_uuid": None,
        "event_downsample_factor": event_downsample_factor,
        "init_speed_range": [0.5, 1.0],
        "event_representation": "time_surface",
        "event_log_compression": 1.0,
        "event_ts_decay_ms": 10.0,
        "enable_navigable_check": True,
        "enable_visualization": False,
        "visualization_log_every_n_steps": 1,
        "task": {
            "name": "hover_stop",
            "config": {
                "w_velocity": 2.0,
                "w_events": 0.0,
                "w_angular": 0.05,
                "w_action": 1e-2,
                "w_survival": 0.5,
                "crash_penalty": 100.0,
                "success_velocity_threshold": 0.15,
                "success_steps_required": 20,
            },
        },
        "vehicle": {
            "dynamics_model": "rotorpy_multirotor_euler",
            "vehicle_name": "crazyflie",
            "control_mode": "ctbr",
            "ctbr_rate_limits": {
                "roll": 7.0,
                "pitch": 7.0,
                "yaw": 5.0,
            },
        },
        "scenes": [
            {
                "name": "apartment_1",
                "path": "data/scene_datasets/habitat-test-scenes/apartment_1.glb",
            },
        ],
        "sensors": {
            "event_camera_1": {
                "type": "event",
                "position": [0.0, 0.0, 0.05],
                "orientation": [0.0, 0.0, 0.0],
                "width": 640,
                "height": 480,
                "hfov": 90,
                "zfar": 20.0,
                "backend": "cuda",
                "contrast_threshold_neg": 0.15,
                "contrast_threshold_pos": 0.15,
                "anti_aliasing": 8,
            },
        },
        "simulator": {
            "world_rate": 1000,
            "control_rate": 100,
            "sim_time": 30,
            "coord_transform": "rotorpy_to_hm3d",
            "sensor_rates": {"event_camera_1": 1000},
            "viz_rates": {"event_camera_1": 20},
        },
    }


def _config_with_simulator_domain_randomization(
    *,
    enabled: bool,
    resample_on_reset: bool,
    sensors: dict | None = None,
) -> dict:
    cfg = copy.deepcopy(
        _test_env_config(obs_mode="state", episode_seconds=0.05),
    )
    sim = cfg.setdefault("simulator", {})
    dr = {
        "enabled": enabled,
        "resample_on_reset": resample_on_reset,
    }
    if sensors is not None:
        dr["sensors"] = sensors
    sim["domain_randomization"] = dr
    return cfg


class TestNeurosimRLEnvDomainRandomization:
    """NeurosimRLEnv wiring for :class:`~neurosim.sims.synchronous_simulator.randomized_simulator.RandomizedSimulator`."""

    def test_dr_disabled_does_not_call_randomize(self):
        cfg = _test_env_config(obs_mode="state", episode_seconds=0.05)
        env = NeurosimRLEnv(env_config=cfg)
        try:
            spy = Mock(wraps=env._rsim.randomize)
            env._rsim.randomize = spy
            env.reset(seed=1)
            env.reset(seed=2)
            spy.assert_not_called()
        finally:
            env.close()

    def test_dr_enabled_first_reset_calls_randomize(self):
        cfg = _config_with_simulator_domain_randomization(
            enabled=True,
            resample_on_reset=False,
            sensors={
                "event_camera_1": {
                    "contrast_threshold_neg": {"range": [0.41, 0.41]},
                },
            },
        )
        env = NeurosimRLEnv(env_config=cfg)
        try:
            spy = Mock(wraps=env._rsim.randomize)
            env._rsim.randomize = spy
            env.reset(seed=10)
            assert spy.call_count == 1
            env.reset(seed=11)
            assert spy.call_count == 1
        finally:
            env.close()

    def test_dr_resample_on_reset_randomizes_each_episode(self):
        cfg = _config_with_simulator_domain_randomization(
            enabled=True,
            resample_on_reset=True,
            sensors={
                "event_camera_1": {
                    "contrast_threshold_neg": {"range": [0.41, 0.41]},
                },
            },
        )
        env = NeurosimRLEnv(env_config=cfg)
        try:
            spy = Mock(wraps=env._rsim.randomize)
            env._rsim.randomize = spy
            env.reset(seed=20)
            assert spy.call_count == 1
            env.reset(seed=21)
            assert spy.call_count == 2
        finally:
            env.close()

    def test_simulator_dr_applies_sensor_sampling(self):
        cfg = _config_with_simulator_domain_randomization(
            enabled=True,
            resample_on_reset=False,
            sensors={
                "event_camera_1": {
                    "contrast_threshold_neg": {"range": [0.37, 0.37]},
                    "hfov": {"choices": [119]},
                },
            },
        )
        env = NeurosimRLEnv(env_config=cfg)
        try:
            env.reset(seed=0)
            s = env.sim.config.visual_sensors["event_camera_1"]
            assert abs(float(s["contrast_threshold_neg"]) - 0.37) < 1e-6
            assert int(s["hfov"]) == 119
        finally:
            env.close()

    def test_vehicle_dr_scales_dynamics_when_enabled(self):
        """Do not hold two Habitat-backed envs open: teardown can abort (IOT) in habitat_sim.close."""
        baseline_cfg = _test_env_config(obs_mode="state", episode_seconds=0.05)
        baseline = NeurosimRLEnv(env_config=baseline_cfg)
        try:
            baseline.reset(seed=5)
            m0 = float(baseline.sim.dynamics._multirotor.mass)
        finally:
            baseline.close()

        dr_cfg = _test_env_config(obs_mode="state", episode_seconds=0.05)
        dr_cfg["vehicle"] = {
            **dr_cfg["vehicle"],
            "domain_randomization": {
                "enabled": True,
                "scales": {
                    "mass": [1.25, 1.25],
                    "k_eta": [1.0, 1.0],
                    "k_m": [1.0, 1.0],
                    "rotor_speed_min": [1.0, 1.0],
                    "rotor_speed_max": [1.0, 1.0],
                },
            },
        }
        dr_env = NeurosimRLEnv(env_config=dr_cfg)
        try:
            dr_env.reset(seed=5)
            m1 = float(dr_env.sim.dynamics._multirotor.mass)
            assert abs(m1 / m0 - 1.25) < 1e-5
        finally:
            dr_env.close()

    def test_vehicle_config_dict_not_mutated(self):
        cfg = _test_env_config(obs_mode="state", episode_seconds=0.05)
        scales = {
            "mass": [1.1, 1.1],
            "k_eta": [1.0, 1.0],
            "k_m": [1.0, 1.0],
            "rotor_speed_min": [1.0, 1.0],
            "rotor_speed_max": [1.0, 1.0],
        }
        cfg["vehicle"]["domain_randomization"] = {"enabled": True, "scales": scales}
        scales_before = copy.deepcopy(scales)
        env = NeurosimRLEnv(env_config=cfg)
        try:
            env.reset(seed=0)
            assert cfg["vehicle"]["domain_randomization"]["scales"] == scales_before
        finally:
            env.close()


@pytest.fixture(scope="module")
def env():
    e = NeurosimRLEnv(
        env_config=_test_env_config(obs_mode="combined", episode_seconds=0.05),
    )
    yield e
    e.close()


class TestNeurosimRLEnv:
    """Smoke tests for the hover-stop RL environment."""

    def test_reset_returns_correct_obs_shape(self, env: NeurosimRLEnv):
        obs, info = env.reset(seed=7)
        assert "events" in obs
        assert "state" in obs
        assert obs["events"].shape == (2, env.event_height, env.event_width)
        assert obs["state"].shape == (13,)
        assert np.isfinite(obs["state"]).all()
        assert "time" in info
        assert "simsteps" in info

    def test_reset_gives_nonzero_velocity(self, env: NeurosimRLEnv):
        env.reset(seed=42)
        v = env.sim.dynamics.state["v"]
        assert float(np.linalg.norm(v)) > 0.1, "Initial velocity should be nonzero"

    def test_reset_gives_safe_initial_position(self, env: NeurosimRLEnv):
        for seed in range(5):
            env.reset(seed=seed)
            x = env.sim.dynamics.state["x"]
            ok, reason = env._safety.check(x)
            assert ok, f"Unsafe initial position (seed={seed}): {reason}, x={x}"

    def test_step_returns_valid_types(self, env: NeurosimRLEnv):
        env.reset(seed=7)
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)

        assert "events" in next_obs
        assert "state" in next_obs
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "reward_terms" in info
        assert "is_success" in info

    def test_reward_terms_present(self, env: NeurosimRLEnv):
        env.reset(seed=11)
        _, _, _, _, info = env.step(env.action_space.sample())
        rt = info["reward_terms"]
        for key in (
            "vel_norm",
            "event_activity",
            "event_activity_density",
            "r_velocity",
        ):
            assert key in rt

    def test_event_activity_density_is_normalized(self, env: NeurosimRLEnv):
        env.reset(seed=9)
        _, _, _, _, info = env.step(env.action_space.sample())
        density = info["reward_terms"]["event_activity_density"]
        assert np.isfinite(density)
        assert density >= 0.0

    def test_no_immediate_termination(self, env: NeurosimRLEnv):
        """After reset + 1 step with neutral normalized CTBR action, should not terminate."""
        env.reset(seed=3)
        # With min-max scaling, 0 maps to midpoint thrust and zero body rates.
        action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        _, _, terminated, _, info = env.step(action)
        assert not terminated, (
            f"Episode terminated on first step. "
            f"reason={info.get('termination_reason')}, "
            f"x={env.sim.dynamics.state['x']}"
        )

    def test_train_mode_forces_visualization_off(self):
        cfg = _test_env_config(obs_mode="state", episode_seconds=0.05)
        cfg["enable_visualization"] = True
        env = NeurosimRLEnv(env_config=cfg, train=True)
        try:
            assert env.enable_visualization is False
        finally:
            env.close()

    def test_downsampled_event_observation_shape(self):
        env = NeurosimRLEnv(
            env_config=_test_env_config(
                obs_mode="events",
                episode_seconds=0.05,
                event_downsample_factor=4,
            ),
        )
        try:
            obs, _ = env.reset(seed=5)
            assert obs.shape == (2, env.event_height, env.event_width)
            assert env.event_height == env._raw_event_height // 4
            assert env.event_width == env._raw_event_width // 4
        finally:
            env.close()


class TestHabitatSafetyCheckerWithRealScene:
    """Tests for HabitatSafetyChecker using the real Habitat apartment scene."""

    @pytest.fixture(scope="class")
    def checker(self) -> HabitatSafetyChecker:
        env = NeurosimRLEnv(
            env_config=_test_env_config(obs_mode="state", episode_seconds=1.0),
        )
        c = env._safety
        env.close()
        return c

    def test_bounds_are_available(self, checker: HabitatSafetyChecker):
        assert checker._lo_x < checker._hi_x
        assert checker._lo_y < checker._hi_y
        assert checker._lo_z < checker._hi_z

    def test_safe_navigable_position(self, checker: HabitatSafetyChecker):
        # A navigable habitat position: sample one and ask the checker.
        nav_pt = np.array(
            checker._pathfinder.get_random_navigable_point(), dtype=np.float64
        )
        # Convert to dynamics via inverse transform.
        dyn_pos = np.linalg.solve(checker._pos_transform, nav_pt)
        ok, reason = checker.check(dyn_pos)
        assert ok, f"Expected safe: reason={reason}"

    def test_out_of_horizontal_bounds(self, checker: HabitatSafetyChecker):
        # Place drone far outside horizontal scene bounds.
        ok, reason = checker.check(np.array([1000.0, 0.0, 0.0]))
        assert not ok
        assert reason == "out_of_bounds"

    def test_sample_habitat_start_returns_safe_position(
        self, checker: HabitatSafetyChecker
    ):
        rng = np.random.default_rng(42)
        for _ in range(5):
            hab_pt = checker.sample_habitat_start(rng)
            dyn_pos = np.linalg.solve(checker._pos_transform, hab_pt)
            ok, reason = checker.check(dyn_pos)
            assert ok, f"Sampled unsafe start: {reason}, hab={hab_pt}"
