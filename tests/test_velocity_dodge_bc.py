import pytest
import torch
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "applications" / "rl"))
from train_velocity_dodge_bc import (
    HDF5ObservationDataset,
    balanced_sample_weights,
    gated_imitation_loss,
)


def test_gated_loss_ignores_quiet_direction_and_trains_gate():
    predicted = torch.tensor(
        [[-2.0, 10.0, -10.0, 4.0], [2.0, 0.0, 1.0, 0.0]],
        requires_grad=True,
    )
    target = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]]
    )
    loss, gate_loss, direction_loss = gated_imitation_loss(predicted, target)
    assert gate_loss.item() > 0.0
    assert direction_loss.item() == pytest.approx(0.0)
    loss.backward()
    assert predicted.grad[0, 0] != 0.0
    assert torch.count_nonzero(predicted.grad[0, 1:]) == 0


def test_gated_loss_rejects_non_gated_actions():
    with pytest.raises(ValueError, match="gated imitation"):
        gated_imitation_loss(torch.zeros(2, 3), torch.zeros(2, 3))


def test_lazy_hdf5_dataset_applies_virtual_onset_and_counterfactuals(tmp_path):
    import h5py

    path = tmp_path / "data.h5"
    actions = np.zeros((4, 4), dtype=np.float32)
    actions[1:3, 0] = 1.0
    actions[1:3, 2] = 0.5
    state = np.zeros((4, 8), dtype=np.float32)
    state[2, -4] = 1.0
    with h5py.File(path, "w") as data:
        data.create_dataset("observations_events", data=np.ones((4, 2, 3, 3), np.float16))
        data.create_dataset("observations_state", data=state)
        data.create_dataset("actions", data=actions)
        data.create_dataset("threat", data=np.array([0, 1, 1, 0], dtype=bool))
        data.create_dataset("episode_index", data=np.zeros(4, dtype=np.int32))

    dataset = HDF5ObservationDataset(
        [str(path)],
        gated_onset_window=2,
        zero_event_counterfactuals=True,
        gated_action_loss=True,
    )
    assert len(dataset) == 8  # 4 base + 2 onset copies + 2 counterfactuals
    onset_obs, onset_action = dataset[4]
    assert onset_action[0] == 1.0
    np.testing.assert_array_equal(onset_obs["state"][-4:], 0.0)
    counter_obs, counter_action = dataset[6]
    np.testing.assert_array_equal(counter_obs["events"], 0.0)
    np.testing.assert_array_equal(counter_obs["state"][-4:], 0.0)
    np.testing.assert_array_equal(counter_action, 0.0)
    dataset.close()


def test_flat_privileged_dataset_stays_flat_and_has_no_false_counterfactuals(tmp_path):
    """Blank-event augmentation is invalid when the observation is all state."""
    import h5py

    path = tmp_path / "flat.h5"
    observations = np.arange(120, dtype=np.float32).reshape(4, 30)
    actions = np.zeros((4, 3), dtype=np.float32)
    actions[1:3, 1] = 0.8
    with h5py.File(path, "w") as data:
        data.create_dataset("observations", data=observations)
        data.create_dataset("actions", data=actions)
        data.create_dataset("threat", data=np.array([0, 1, 1, 0], dtype=bool))
        data.create_dataset("episode_index", data=np.zeros(4, dtype=np.int32))

    dataset = HDF5ObservationDataset(
        [str(path)], gated_onset_window=2, zero_event_counterfactuals=True
    )
    assert len(dataset) == 6  # 4 base + 2 onset copies; no counterfactuals
    obs, action = dataset[1]
    np.testing.assert_array_equal(obs, observations[1])
    np.testing.assert_array_equal(action, actions[1])
    threat, counterfactual = dataset.sample_kinds
    assert not counterfactual.any()
    weights = balanced_sample_weights(threat, counterfactual)
    assert weights[threat].sum() == pytest.approx(0.5)
    assert weights[~threat].sum() == pytest.approx(0.5)
    onset_obs, onset_action = dataset[4]
    np.testing.assert_array_equal(onset_obs[15:18], 0.0)
    np.testing.assert_array_equal(onset_action, actions[1])
    dataset.close()
