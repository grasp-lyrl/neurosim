import numpy as np
import torch
from copy import deepcopy

from applications.rl.collect_event_tracker_dagger_features import choose_rollout_action
from applications.rl.train_event_tracker_action_head import (
    CachedEpisode,
    EventTrackerActionHead,
    LearnedPulseController,
    causal_visibility_mask,
    checkpoint_selection_score,
    commitment_targets,
    gate_classification_loss,
    load_feature_episodes,
    rebase_input_normalization,
)


def test_visibility_gate_confirms_and_resets_causally():
    probability = np.asarray([0.1, 0.8, 0.9, 0.2, 0.8, 0.1, 0.1, 0.1])
    activity = np.ones_like(probability)
    mask = causal_visibility_mask(
        probability,
        activity,
        threshold=0.7,
        confirm_steps=2,
        reset_absence_steps=3,
    )
    np.testing.assert_array_equal(
        mask, [False, False, True, True, True, True, True, False]
    )


def test_zero_event_activity_cannot_open_visibility_gate():
    mask = causal_visibility_mask(
        np.ones(5), np.zeros(5), threshold=0.7, confirm_steps=2, reset_absence_steps=3
    )
    assert not mask.any()


def test_action_head_never_commands_forward_axis():
    model = EventTrackerActionHead(input_dim=12, hidden_size=16)
    output = model(torch.randn(3, 7, 12))
    assert output["action"].shape == (3, 7, 3)
    torch.testing.assert_close(output["action"][..., 0], torch.zeros(3, 7))


def test_gate_only_loss_does_not_backpropagate_through_direction_head():
    model = EventTrackerActionHead(input_dim=12, hidden_size=16)
    output = model(torch.randn(2, 5, 12))
    target = torch.zeros(2, 5, 3)
    target[0, 2:, 1] = 1.0
    loss = gate_classification_loss(
        output, target, torch.ones(2, 5, dtype=torch.bool)
    )
    loss.backward()
    assert model.gate_head.weight.grad is not None
    assert model.direction_head.weight.grad is None


def test_gate_only_checkpoint_selection_prefers_precision():
    higher_precision = {
        "trigger_precision": 0.8,
        "trigger_recall": 0.5,
    }
    higher_recall = {
        "trigger_precision": 0.5,
        "trigger_recall": 0.8,
    }
    assert checkpoint_selection_score(
        higher_precision, gate_only=True
    ) > checkpoint_selection_score(higher_recall, gate_only=True)


def test_compact_feature_shard_roundtrip(tmp_path):
    episode = {
        "key": "seed1",
        "inputs": np.zeros((4, 143), dtype=np.float32),
        "actions": np.zeros((4, 3), dtype=np.float32),
        "original_actions": np.ones((4, 3), dtype=np.float32),
        "probabilities": np.linspace(0, 1, 4, dtype=np.float32),
        "label_valid": np.asarray([True, False, True, True]),
    }
    path = tmp_path / "features.pt"
    torch.save({"input_dim": 143, "episodes": [episode]}, path)
    loaded = load_feature_episodes([str(path)])
    assert len(loaded) == 1
    assert isinstance(loaded[0], CachedEpisode)
    np.testing.assert_array_equal(loaded[0].inputs, episode["inputs"])
    np.testing.assert_array_equal(loaded[0].label_valid, episode["label_valid"])
    np.testing.assert_array_equal(loaded[0].action_valid, episode["label_valid"])

    gate_only = load_feature_episodes([str(path)], action_labels_valid=False)
    # An explicit action-valid mask takes precedence; this shard has none, so
    # all frames become gate-only supervision.
    assert not gate_only[0].action_valid.any()


def test_dagger_action_mixture_boundaries():
    expert = np.asarray([1.0, 2.0, 3.0])
    learner = np.asarray([-1.0, -2.0, -3.0])
    np.testing.assert_array_equal(
        choose_rollout_action(expert, learner, beta=1.0, draw=0.5), expert
    )
    np.testing.assert_array_equal(
        choose_rollout_action(expert, learner, beta=0.0, draw=0.5), learner
    )


def test_normalization_rebase_preserves_recurrent_function():
    torch.manual_seed(7)
    original = EventTrackerActionHead(input_dim=12, hidden_size=16).eval()
    rebased = deepcopy(original)
    old_mean = np.linspace(-0.2, 0.3, 12, dtype=np.float32)
    old_std = np.linspace(0.5, 1.5, 12, dtype=np.float32)
    new_mean = np.linspace(0.4, -0.1, 12, dtype=np.float32)
    new_std = np.linspace(1.4, 0.6, 12, dtype=np.float32)
    raw = torch.randn(2, 5, 12)
    old_input = (raw - torch.from_numpy(old_mean)) / torch.from_numpy(old_std)
    new_input = (raw - torch.from_numpy(new_mean)) / torch.from_numpy(new_std)
    rebase_input_normalization(rebased, old_mean, old_std, new_mean, new_std)
    expected = original(old_input)
    actual = rebased(new_input)
    torch.testing.assert_close(actual["gate_logit"], expected["gate_logit"])
    torch.testing.assert_close(actual["action"], expected["action"])


def test_commitment_targets_use_one_direction_per_active_run():
    actions = np.zeros((7, 3), dtype=np.float32)
    actions[1:4, 1:] = [[0.8, 0.1], [0.5, 0.3], [0.7, 0.2]]
    actions[5:, 1:] = [[-0.4, 0.2], [-0.7, 0.1]]
    target = commitment_targets(actions, np.ones(7, dtype=bool))
    np.testing.assert_allclose(target[1], target[2])
    np.testing.assert_allclose(target[2], target[3])
    np.testing.assert_allclose(np.linalg.norm(target[1, 1:]), 1.0)
    assert target[1, 1] > 0.0
    assert target[5, 1] < 0.0
    np.testing.assert_array_equal(target[[0, 4]], 0.0)


def test_learned_pulse_latches_direction_and_requires_absence_to_rearm():
    controller = LearnedPulseController(
        confirm_steps=2,
        hold_steps=5,
        ramp_steps=2,
        refractory_absence_steps=2,
    )
    first = controller.step(0.9, [1.0, 0.0], track_present=True)
    np.testing.assert_array_equal(first, np.zeros(3))
    second = controller.step(0.9, [1.0, 0.0], track_present=True)
    assert controller.last_committed
    assert 0.0 < second[1] < 1.0
    # Contradictory predictions cannot reverse an active pulse.
    third = controller.step(0.9, [-1.0, 0.0], track_present=True)
    assert third[1] > 0.0
    for _ in range(3):
        controller.step(0.9, [-1.0, 0.0], track_present=True)
    assert controller.refractory
    for _ in range(3):
        action = controller.step(0.9, [-1.0, 0.0], track_present=True)
        np.testing.assert_array_equal(action, np.zeros(3))
    controller.step(0.0, [0.0, 0.0], track_present=False)
    controller.step(0.0, [0.0, 0.0], track_present=False)
    controller.step(0.9, [-1.0, 0.0], track_present=True)
    action = controller.step(0.9, [-1.0, 0.0], track_present=True)
    assert controller.last_committed
    assert action[1] < 0.0


def test_learned_pulse_releases_early_after_confirmed_pass():
    controller = LearnedPulseController(
        confirm_steps=1,
        hold_steps=20,
        ramp_steps=3,
        minimum_hold_steps=8,
        release_inbound_threshold=0.35,
        release_confirm_steps=3,
    )
    actions = [
        controller.step(0.9, [1.0, 0.0], track_present=True)
    ]
    # Eight outward samples are guaranteed, even if a noisy temporal head
    # claims immediately that the object has passed.
    for _ in range(7):
        actions.append(
            controller.step(
                0.9, [1.0, 0.0], track_present=True, threat_passed=True
            )
        )
    assert controller.pulse_step == 8
    assert controller.release_confirmed == 0

    # Three confirmed frames start a smooth 2/3, 1/3, 0 release.
    for _ in range(3):
        actions.append(
            controller.step(
                0.9, [1.0, 0.0], track_present=True, threat_passed=True
            )
        )
    assert controller.early_releases == 1
    assert controller.early_release_pulse_steps == [10]
    np.testing.assert_allclose(actions[-1][1], 2.0 / 3.0)
    penultimate = controller.step(
        0.9, [1.0, 0.0], track_present=True, threat_passed=True
    )
    final = controller.step(
        0.9, [1.0, 0.0], track_present=True, threat_passed=True
    )
    np.testing.assert_allclose(penultimate[1], 1.0 / 3.0)
    np.testing.assert_array_equal(final, np.zeros(3))
    assert controller.refractory
    assert controller.pulse_step is None


def test_tracker_disappearance_does_not_end_pulse_early():
    controller = LearnedPulseController(
        confirm_steps=1,
        hold_steps=12,
        ramp_steps=2,
        minimum_hold_steps=4,
        release_inbound_threshold=0.35,
        release_confirm_steps=2,
    )
    controller.step(0.9, [1.0, 0.0], track_present=True)
    for _ in range(8):
        controller.step(
            0.0, [0.0, 0.0], track_present=False, threat_passed=False
        )
    assert controller.pulse_step is not None
    assert controller.early_releases == 0
