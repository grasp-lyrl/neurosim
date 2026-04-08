"""Tests for the neurosim hover-stop RL environment.

All tests use the real Habitat backend (apartment_1 scene) since RL training
always runs with Habitat as the visual backend.
"""

import numpy as np
import pytest
import yaml

from neurosim.rl import NeurosimRLEnv
from neurosim.rl.safety import HabitatSafetyChecker

SETTINGS = "configs/apartment_1-settings.yaml"


def _test_settings() -> dict:
    with open(SETTINGS, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["visual_backend"]["sensors"]["event_camera_1"]["backend"] = "cuda"
    cfg["visual_backend"]["sensors"].pop("optical_flow_1", None)
    cfg["simulator"]["sensor_rates"].pop("optical_flow_1", None)
    cfg["simulator"]["viz_rates"].pop("optical_flow_1", None)
    return cfg


def _test_env_config(
    *,
    obs_mode: str,
    episode_seconds: float,
    event_downsample_factor: int = 1,
) -> dict:
    return {
        "settings": _test_settings(),
        "obs_mode": obs_mode,
        "episode_seconds": episode_seconds,
        "event_sensor_uuid": None,
        "event_clip": 10.0,
        "event_downsample_factor": event_downsample_factor,
        "init_speed_range": [0.5, 1.0],
        "event_representation": "time_surface",
        "event_log_compression": None,
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
            "domain_randomization": {
                "enabled": False,
                "scales": {
                    "mass": [1.0, 1.0],
                    "k_eta": [1.0, 1.0],
                    "k_m": [1.0, 1.0],
                    "rotor_speed_min": [1.0, 1.0],
                    "rotor_speed_max": [1.0, 1.0],
                },
            },
        },
    }


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
