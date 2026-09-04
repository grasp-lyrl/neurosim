from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from applications.rl.event_geometry_residual import EventGeometryResidualWrapper


class _DummyEnv(gym.Env):
    def __init__(self, event_value=1.0):
        self.event_value = float(event_value)
        self.observation_space = gym.spaces.Dict(
            {
                "events": gym.spaces.Box(0, 1, (2, 4, 5), dtype=np.float32),
                "state": gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
                "privileged": gym.spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32),
            }
        )
        self.action_space = gym.spaces.Box(-1, 1, (3,), dtype=np.float32)
        self.last_action = None

    def _obs(self):
        return {
            "events": np.full(
                (2, 4, 5), self.event_value, dtype=np.float32
            ),
            "state": np.asarray([1, 2, 3], dtype=np.float32),
            "privileged": np.arange(9, dtype=np.float32),
        }

    def reset(self, *, seed=None, options=None):
        return self._obs(), {}

    def step(self, action):
        self.last_action = np.asarray(action)
        return self._obs(), 1.0, False, False, {}


class _Tracker:
    def __init__(self, probability):
        self.probability = probability

    def reset(self):
        pass

    def step(self, events):
        return SimpleNamespace(
            probability=self.probability,
            centre=np.asarray([0.6, 0.4]),
            geometry=np.asarray([0.1, 0.7]),
            inbound_probability=0.8,
        )


class _Controller:
    hold_steps = 20

    def reset(self):
        self.pending_direction = None
        self.hold_remaining = 0
        self.track_age = 0
        self.refractory = False
        self.motion = np.zeros(2)
        self.direction = np.zeros(3)

    def step(self, prediction):
        return np.asarray([0.0, 0.5, 0.0], dtype=np.float32), False


def _wrapper(probability=0.9, event_value=1.0):
    return EventGeometryResidualWrapper(
        _DummyEnv(event_value),
        spatial_checkpoint="unused",
        temporal_checkpoint="unused",
        tracker=_Tracker(probability),
        controller=_Controller(),
        residual_scale=0.4,
    )


def test_wrapper_removes_events_but_preserves_privileged_critic_channel():
    env = _wrapper()
    observation, _ = env.reset()
    assert set(observation) == {"state", "privileged"}
    assert observation["state"].shape == (3 + len(env.FEATURE_NAMES),)
    np.testing.assert_array_equal(observation["privileged"], np.arange(9))
    assert env.observation_space.contains(observation)


def test_visible_track_adds_bounded_residual_to_baseline():
    env = _wrapper(probability=0.9)
    env.reset()
    _, _, _, _, info = env.step(np.asarray([1.0, -1.0, 0.5]))
    np.testing.assert_allclose(env.unwrapped.last_action, [0.4, 0.1, 0.2])
    assert info["event_geometry_residual"]["gate"] == 1.0


def test_blank_track_disables_residual_when_baseline_is_zero():
    controller = _Controller()
    controller.step = lambda prediction: (np.zeros(3, dtype=np.float32), False)
    env = EventGeometryResidualWrapper(
        _DummyEnv(event_value=0.0),
        spatial_checkpoint="unused",
        temporal_checkpoint="unused",
        tracker=_Tracker(0.1),
        controller=controller,
        residual_scale=0.4,
    )
    env.reset()
    env.step(np.ones(3, dtype=np.float32))
    np.testing.assert_array_equal(env.unwrapped.last_action, np.zeros(3))
