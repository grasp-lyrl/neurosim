from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from applications.rl.event_pulse_residual import EventPulseResidualWrapper
from applications.rl.train_event_tracker_action_head import LearnedPulseController


class _DummyEnv(gym.Env):
    def __init__(self, event_value=1.0):
        self.event_value = float(event_value)
        self.observation_space = gym.spaces.Dict(
            {
                "events": gym.spaces.Box(-1, 1, (2, 4, 5), dtype=np.float32),
                "state": gym.spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32),
                "privileged": gym.spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32),
            }
        )
        self.action_space = gym.spaces.Box(-1, 1, (3,), dtype=np.float32)
        self.last_action = None

    def _obs(self):
        return {
            "events": np.full((2, 4, 5), self.event_value, dtype=np.float32),
            "state": np.asarray([1, 2, 3, 4], dtype=np.float32),
            "privileged": np.arange(9, dtype=np.float32),
        }

    def reset(self, *, seed=None, options=None):
        return self._obs(), {}

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float32)
        return self._obs(), 1.0, False, False, {}


class _Policy:
    def __init__(self, *, gate=0.6, direction=(0.2, 0.2)):
        self.mean = np.zeros(4, dtype=np.float32)
        self.std = np.ones(4, dtype=np.float32)
        self.gate = float(gate)
        self.direction = np.asarray(direction, dtype=np.float32)
        self.pulse_controller = LearnedPulseController(
            confirm_steps=2,
            hold_steps=5,
            ramp_steps=2,
            refractory_absence_steps=2,
        )
        self.reset()

    def reset(self):
        self.pulse_controller.reset()
        self.last_prediction = None
        self.last_input = np.zeros(4, dtype=np.float32)
        self.last_gate_probability = 0.0
        self.last_direction = np.zeros(2, dtype=np.float32)
        self.last_event_active = False
        self.last_committed = False

    def infer(self, observation):
        self.last_event_active = bool(np.any(np.abs(observation["events"]) > 1e-6))
        probability = 0.9 if self.last_event_active else 0.1
        self.last_prediction = SimpleNamespace(
            probability=probability, inbound_probability=0.8
        )
        self.last_input = np.asarray(observation["state"], dtype=np.float32)
        self.last_gate_probability = self.gate
        self.last_direction = self.direction.copy()
        return self.last_gate_probability, self.last_direction.copy()

    def action_from_inference(self, gate_probability=None, direction=None):
        gate_probability = (
            self.last_gate_probability if gate_probability is None else gate_probability
        )
        direction = self.last_direction if direction is None else direction
        action = self.pulse_controller.step(
            gate_probability,
            direction,
            track_present=(
                self.last_event_active and self.last_prediction.probability >= 0.7
            ),
        )
        self.last_committed = self.pulse_controller.last_committed
        return action, None


def _wrapper(
    *, event_value=1.0, policy=None, direction_scale=0.25, gate_logit_scale=1.0
):
    return EventPulseResidualWrapper(
        _DummyEnv(event_value),
        action_checkpoint="unused",
        policy=policy or _Policy(),
        direction_scale=direction_scale,
        gate_logit_scale=gate_logit_scale,
    )


def test_wrapper_exposes_only_causal_actor_features_and_privileged_critic():
    env = _wrapper()
    observation, _ = env.reset()
    assert set(observation) == {"state", "privileged"}
    assert observation["state"].shape == (4 + len(env.FEATURE_NAMES),)
    np.testing.assert_array_equal(observation["privileged"], np.arange(9))
    assert env.observation_space.contains(observation)


def test_zero_residual_exactly_matches_unmodified_pulse_policy():
    wrapped_policy = _Policy()
    baseline_policy = _Policy()
    env = _wrapper(policy=wrapped_policy)
    observation, _ = env.reset()
    baseline_policy.reset()
    baseline_policy.infer(env.unwrapped._obs())

    for _ in range(9):
        expected, _ = baseline_policy.action_from_inference()
        observation, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
        np.testing.assert_array_equal(env.unwrapped.last_action, expected)
        np.testing.assert_array_equal(
            info["event_pulse_residual"]["baseline_action"], expected
        )
        assert env.observation_space.contains(observation)
        baseline_policy.infer(env.unwrapped._obs())


def test_blank_events_cannot_trigger_even_with_maximum_ppo_action():
    env = _wrapper(
        event_value=0.0, policy=_Policy(gate=0.99), gate_logit_scale=8.0
    )
    env.reset()
    for _ in range(8):
        _, _, _, _, info = env.step(np.ones(3, dtype=np.float32))
        np.testing.assert_array_equal(env.unwrapped.last_action, np.zeros(3))
        assert info["event_pulse_residual"]["gate"] == 0.0
        assert not info["event_pulse_residual"]["committed"]


def test_wide_guarded_logit_range_can_recover_low_visible_bc_trigger():
    env = _wrapper(policy=_Policy(gate=0.007), gate_logit_scale=8.0)
    env.reset()
    residual = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    env.step(residual)
    _, _, _, _, info = env.step(residual)
    assert info["event_pulse_residual"]["adjusted_gate_probability"] > 0.5
    assert info["event_pulse_residual"]["committed"]
    assert np.linalg.norm(env.unwrapped.last_action) > 0.0


def test_direction_adjustment_is_latched_and_cannot_reverse_mid_pulse():
    env = _wrapper(policy=_Policy(), direction_scale=0.5)
    env.reset()
    residual = np.asarray([1.0, -1.0, 1.0], dtype=np.float32)
    env.step(residual)
    _, _, _, _, info = env.step(residual)
    assert info["event_pulse_residual"]["committed"]
    committed_action = env.unwrapped.last_action.copy()
    assert committed_action[1] < 0.0 and committed_action[2] > 0.0

    env.step(-residual)
    continued_action = env.unwrapped.last_action
    assert continued_action[1] < 0.0 and continued_action[2] > 0.0


def test_consistent_late_threat_selectively_renews_same_latched_direction():
    controller = LearnedPulseController(
        confirm_steps=1,
        hold_steps=6,
        ramp_steps=2,
        renewal_extension_steps=3,
        renewal_confirm_steps=2,
        maximum_hold_steps=9,
        renewal_direction_cosine=0.0,
    )
    direction = np.asarray([1.0, 0.0], dtype=np.float32)
    actions = [
        controller.step(0.9, direction, track_present=True)
        for _ in range(7)
    ]

    assert controller.renewals == 1
    assert controller.renewal_pulse_steps == [3]
    # The original six-step pulse would already have returned zero. Renewal
    # stays in the exact initially latched half-plane without another ramp.
    assert actions[5][1] > 0.9
    assert actions[6][1] > 0.9
    np.testing.assert_array_equal(controller.direction, direction)


def test_opposite_late_proposal_cannot_renew_or_reverse_pulse():
    controller = LearnedPulseController(
        confirm_steps=1,
        hold_steps=6,
        ramp_steps=2,
        renewal_extension_steps=3,
        renewal_confirm_steps=2,
        maximum_hold_steps=9,
        renewal_direction_cosine=0.0,
    )
    forward = np.asarray([1.0, 0.0], dtype=np.float32)
    opposite = -forward
    actions = [controller.step(0.9, forward, track_present=True)]
    actions.extend(
        controller.step(0.9, opposite, track_present=True) for _ in range(5)
    )

    assert controller.renewals == 0
    assert controller.refractory
    np.testing.assert_array_equal(actions[-1], np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(controller.direction, forward)
