import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "applications" / "rl"))

from train_velocity_dodge_gru_bc import (  # noqa: E402
    HDF5SequenceDataset,
    build_gru_clone_net,
)
from train_velocity_dodge_compact_gru import (  # noqa: E402
    CommitmentController,
    PairedCompactSequenceDataset,
    build_compact_model,
    compact_targets,
)


def test_sequence_dataset_never_crosses_episode_and_builds_counterfactual(tmp_path):
    import h5py

    path = tmp_path / "sequences.h5"
    events = np.arange(10 * 2 * 4 * 4, dtype=np.float16).reshape(10, 2, 4, 4)
    state = np.arange(10 * 18, dtype=np.float32).reshape(10, 18)
    actions = np.zeros((10, 3), dtype=np.float32)
    actions[1:3, 1] = 0.5
    with h5py.File(path, "w") as file:
        file.create_dataset("observations_events", data=events)
        file.create_dataset("observations_state", data=state)
        file.create_dataset("actions", data=actions)
        file.create_dataset("episode_index", data=np.repeat([0, 1], 5))

    dataset = HDF5SequenceDataset(
        [str(path)],
        sequence_length=4,
        stride=2,
        downsample_events=2,
        zero_event_counterfactuals=True,
    )
    # Episode 0 contributes real+counterfactual; episode 1 contributes quiet.
    assert dataset.group_counts == [1, 1, 1]
    real, action = dataset[0]
    assert real["events"].shape == (4, 2, 2, 2)
    np.testing.assert_array_equal(action[:, 1], [0.0, 0.5, 0.5, 0.0])
    blank, blank_action = dataset[1]
    np.testing.assert_array_equal(blank["events"], 0.0)
    np.testing.assert_array_equal(blank_action, 0.0)
    quiet, _ = dataset[2]
    # The second real sequence begins at row 5, not in episode 0's tail.
    assert quiet["state"][0, 0] == state[5, 0]
    dataset.close()


def test_gru_clone_sequence_and_stateful_step_shapes():
    class Env:
        observation_space = gym.spaces.Dict(
            {
                "events": gym.spaces.Box(0.0, 1.0, (8, 60, 80), np.float32),
                "state": gym.spaces.Box(-np.inf, np.inf, (18,), np.float32),
            }
        )
        action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)

    cfg = {
        "env": {"obs_mode": "combined"},
        "ppo": {
            "log_std_init": -2.0,
            "event_presence_features": False,
            "event_high_resolution": False,
        },
    }
    model, action_dim = build_gru_clone_net(
        cfg,
        Env(),
        hidden_size=16,
        events_only_state=True,
        blank_previous_action=True,
    )
    assert action_dim == 3
    obs = {
        "events": torch.rand(2, 4, 8, 60, 80),
        "state": torch.rand(2, 4, 18),
    }
    action, raw, side, hidden = model.forward_sequence(obs)
    assert action.shape == raw.shape == (2, 4, 3)
    assert side.shape == (2, 4, 3)
    assert hidden.shape == (1, 2, 16)

    step_obs = {key: value[:, 0] for key, value in obs.items()}
    first, hidden = model.recurrent_step(step_obs)
    second, hidden = model.recurrent_step(step_obs, hidden)
    assert first.shape == second.shape == (2, 3)
    assert hidden.shape == (1, 2, 16)


def test_privileged_flat_sequence_dataset_and_gru(tmp_path):
    import h5py

    path = tmp_path / "privileged.h5"
    observations = np.arange(12 * 30, dtype=np.float32).reshape(12, 30)
    actions = np.zeros((12, 3), dtype=np.float32)
    actions[1:4, 1] = 0.75
    with h5py.File(path, "w") as file:
        file.create_dataset("observations", data=observations)
        file.create_dataset("actions", data=actions)
        file.create_dataset("episode_index", data=np.repeat([0, 1], 6))

    dataset = HDF5SequenceDataset(
        [str(path)], sequence_length=4, stride=2
    )
    obs, target = dataset[0]
    assert obs.shape == (4, 30)
    assert target.shape == (4, 3)
    dataset.close()

    class Env:
        observation_space = gym.spaces.Box(
            -np.inf, np.inf, (30,), np.float32
        )
        action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)

    cfg = {
        "env": {"obs_mode": "state"},
        "ppo": {
            "log_std_init": -2.0,
            "event_presence_features": False,
            "event_high_resolution": False,
        },
    }
    model, _ = build_gru_clone_net(
        cfg, Env(), hidden_size=16, blank_previous_action=True
    )
    tensor = torch.from_numpy(obs[None])
    masked = model.preprocess(tensor)
    torch.testing.assert_close(masked[..., :15], tensor[..., :15])
    torch.testing.assert_close(masked[..., 15:18], torch.zeros_like(masked[..., 15:18]))
    torch.testing.assert_close(masked[..., 18:], tensor[..., 18:])
    action, raw, side, hidden = model.forward_sequence(tensor)
    assert action.shape == raw.shape == (1, 4, 3)
    assert side.shape == (1, 4, 3)
    assert hidden.shape == (1, 1, 16)


def test_paired_compact_dataset_targets_model_and_commitment(tmp_path):
    import h5py

    event_path = tmp_path / "events.h5"
    privileged_path = tmp_path / "privileged.h5"
    rows = 32
    episode_index = np.repeat(np.arange(4), 8)
    episode_seed = np.repeat(np.arange(100, 104), 8)
    events = np.ones((rows, 8, 8, 8), dtype=np.float16)
    state = np.arange(rows * 18, dtype=np.float32).reshape(rows, 18)
    actions = np.zeros((rows, 3), dtype=np.float32)
    for start in range(0, rows, 8):
        actions[start : start + 4, 1] = 0.6
    threat = np.abs(actions[:, 1]) > 0.15
    privileged = np.zeros((rows, 30), dtype=np.float32)
    privileged[:, 18] = 1.0
    privileged[:, 19:22] = [3.0, 0.5, -0.25]
    privileged[:, 22:25] = [-4.0, 0.0, 0.0]
    privileged[:, 29] = 0.5
    with h5py.File(event_path, "w") as file:
        file.create_dataset("observations_events", data=events)
        file.create_dataset("observations_state", data=state)
        file.create_dataset("actions", data=actions)
        file.create_dataset("episode_index", data=episode_index)
        file.create_dataset("episode_seed", data=episode_seed)
        file.create_dataset("threat", data=threat)
    with h5py.File(privileged_path, "w") as file:
        file.create_dataset("observations", data=privileged)
        file.create_dataset("actions", data=actions)
        file.create_dataset("episode_index", data=episode_index)
        file.create_dataset("episode_seed", data=episode_seed)
        file.create_dataset("threat", data=threat)

    dataset = PairedCompactSequenceDataset(
        [str(event_path)],
        [str(privileged_path)],
        sequence_length=4,
        stride=4,
        downsample_events=2,
        holdout_fraction=0.25,
        holdout_side="train",
        counterfactuals=True,
    )
    assert dataset.group_counts == [3, 3, 3]
    obs, geometry, action = dataset[0]
    assert obs["events"].shape == (4, 8, 4, 4)
    target = compact_targets(
        torch.from_numpy(geometry[None]), torch.from_numpy(action[None])
    )
    torch.testing.assert_close(target["tca"], torch.full((1, 4), 1.0 / 3.0))
    torch.testing.assert_close(
        target["miss"], torch.tensor([[[0.2, -0.1]] * 4])
    )
    blank_obs, blank_geometry, blank_action = dataset[1]
    np.testing.assert_array_equal(blank_obs["events"], 0.0)
    np.testing.assert_array_equal(blank_geometry, 0.0)
    np.testing.assert_array_equal(blank_action, 0.0)
    dataset.close()

    # A DAgger shard can carry its own privileged supervision while exposing
    # only events/state to the rollout model.
    with h5py.File(event_path, "a") as file:
        file.create_dataset("observations_privileged", data=privileged[:, 18:30])
    self_paired = PairedCompactSequenceDataset(
        [str(event_path)], [str(event_path)], sequence_length=4, stride=4,
        holdout_fraction=0.25, holdout_side="train", counterfactuals=False,
    )
    _, self_geometry, _ = self_paired[0]
    np.testing.assert_array_equal(self_geometry, privileged[:4, 18:30])
    self_paired.close()

    class Env:
        observation_space = gym.spaces.Dict(
            {
                "events": gym.spaces.Box(0.0, 1.0, (8, 8, 8), np.float32),
                "state": gym.spaces.Box(-np.inf, np.inf, (18,), np.float32),
            }
        )
        action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)

    cfg = {
        "env": {"obs_mode": "combined"},
        "ppo": {
            "log_std_init": -2.0,
            "event_presence_features": False,
            "event_high_resolution": False,
        },
    }
    model = build_compact_model(cfg, Env(), downsample_events=2, hidden_size=16)
    model_obs = {
        "events": torch.rand(2, 3, 8, 4, 4),
        "state": torch.rand(2, 3, 18),
    }
    output, hidden = model.forward_sequence(model_obs)
    assert output["trigger_logit"].shape == output["tca"].shape == (2, 3)
    assert output["miss"].shape == (2, 3, 2)
    assert output["side_logits"].shape == (2, 3, 3)
    assert output["magnitude"].shape == (2, 3)
    assert hidden.shape == (1, 2, 16)
    masked = model.preprocess(model_obs)["state"]
    torch.testing.assert_close(masked[..., 0:3], model_obs["state"][..., 0:3])
    torch.testing.assert_close(masked[..., 12:15], model_obs["state"][..., 12:15])
    torch.testing.assert_close(masked[..., 3:12], torch.zeros_like(masked[..., 3:12]))
    torch.testing.assert_close(masked[..., 15:18], torch.zeros_like(masked[..., 15:18]))

    controller = CommitmentController(hold_steps=2, alpha=1.0)
    left = {
        "trigger_logit": torch.tensor([10.0]),
        "side_logits": torch.tensor([[10.0, 0.0, 0.0]]),
        "magnitude": torch.tensor([0.6]),
    }
    right = {
        "trigger_logit": torch.tensor([10.0]),
        "side_logits": torch.tensor([[0.0, 0.0, 10.0]]),
        "magnitude": torch.tensor([0.7]),
    }
    assert np.isclose(controller.action(left)[1], -0.6)
    # Opposite frame-level predictions cannot flip an active commitment.
    assert np.isclose(controller.action(right)[1], -0.7)

    miss_controller = CommitmentController(
        hold_steps=2, alpha=1.0, side_source="miss"
    )
    geometry = dict(left)
    geometry["miss"] = torch.tensor([[0.4, 0.0]])
    # A projectile predicted to pass on +body-y produces a -body-y dodge,
    # even when the imitation side head votes the other way.
    geometry["side_logits"] = torch.tensor([[0.0, 0.0, 10.0]])
    assert np.isclose(miss_controller.action(geometry)[1], -0.6)
