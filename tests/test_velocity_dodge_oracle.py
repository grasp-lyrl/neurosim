import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "applications" / "rl"))

from evaluate_velocity_dodge_oracle import (
    HDF5ImitationWriter,
    oracle_action,
    should_include_episode,
)
from neurosim.rl.trajectory_dodge_expert import LocalTrajectoryExpert, QuinticAvoidancePlan


class _Sim:
    time = 0.5
    dynamics = type(
        "Dynamics",
        (),
        {
            "state": {
                "q": np.array([0.0, 0.0, 0.0, 1.0]),
                "x": np.zeros(3),
            }
        },
    )()


class _Task:
    threat_distance_m = 1.5
    threat_time_horizon_s = 5.0
    outer_position_gain_hz = np.array([1.5, 1.5, 2.5])
    delta_velocity_limits_mps = np.array([0.7, 0.7, 0.3])
    # This fixture's action_space is the 4-dim gated layout (gate +
    # 3-axis residual), not velocity_delta_integrated's ungated 3-dim one.
    residual_control_mode = "gated_cascaded_velocity"


class _Env:
    action_space = type("ActionSpace", (), {"shape": (4,)})()
    sim = _Sim()
    _task = _Task()
    _oracle_latched_actions = {}

    def __init__(self, closest_lateral):
        self._oracle_latched_actions = {}
        # At tca=1 the current lateral bearing is positive in both cases;
        # relative velocity determines which side it will actually pass on.
        self.row = {
            "object_id": 7,
            "rel_pos": np.array([2.0, 1.0, 0.0]),
            "rel_vel": np.array([-2.0, closest_lateral - 1.0, 0.0]),
            "predicted_clearance": abs(closest_lateral),
            "time_to_closest_approach": 1.0,
        }

    def _obstacle_relative_states(self, state):
        return [self.row]

    def _nominal_flat(self):
        return {
            "x": np.zeros(3),
            "x_dot": np.array([1.0, 0.0, 0.0]),
        }


def test_oracle_tracks_cached_smooth_trajectory_in_body_velocity_coordinates():
    env = _Env(0.2)
    env._trajectory_dodge_expert = LocalTrajectoryExpert()
    env._trajectory_dodge_expert.plan = QuinticAvoidancePlan(
        7,
        0.0,
        1.0,
        2.0,
        np.array([0.0, 0.5, 0.0]),
        0.2,
        validated_object_ids=(7,),
    )
    action = oracle_action(env)

    assert action[0] == 1.0
    assert action[2] > 0.0
    assert action[1] == 0.0


def test_include_all_keeps_timeouts_but_rejects_invalid_rollouts():
    assert should_include_episode(
        {"success": False, "termination_reason": "timeout"}, True
    )
    assert not should_include_episode(
        {"success": False, "termination_reason": "tracking_failure"}, True
    )
    assert not should_include_episode(
        {"success": False, "termination_reason": "out_of_bounds"}, True
    )
    assert should_include_episode(
        {"success": True, "termination_reason": "timeout"}, False
    )


def test_hdf5_writer_streams_episode_metadata_and_half_precision_events(tmp_path):
    import h5py

    path = tmp_path / "expert.h5"
    writer = HDF5ImitationWriter(path, "config.yaml")
    observations = [
        {
            "events": np.full((2, 3, 4), index, dtype=np.float32),
            "state": np.array([index, index + 1], dtype=np.float32),
        }
        for index in range(3)
    ]
    actions = [np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32)]
    actions.append(np.zeros(4, dtype=np.float32))
    writer.append_episode(observations, actions, episode=5, seed=123)
    writer.close()

    with h5py.File(path, "r") as data:
        assert data["observations_events"].dtype == np.float16
        assert data["observations_events"].shape == (3, 2, 3, 4)
        np.testing.assert_array_equal(data["episode_index"][:], 5)
        np.testing.assert_array_equal(data["episode_seed"][:], 123)
        assert int(data.attrs["samples"]) == 3
        assert int(data.attrs["threat_samples"]) == 1


def test_hdf5_writer_skips_empty_episode_and_rejects_misaligned_samples(tmp_path):
    import h5py

    path = tmp_path / "empty.h5"
    writer = HDF5ImitationWriter(path, "config.yaml")
    writer.append_episode([], [], episode=0, seed=1)
    with pytest.raises(ValueError, match="length mismatch"):
        writer.append_episode(
            [{"events": np.zeros((2, 3, 4), dtype=np.float32)}],
            [np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32)],
            episode=1,
            seed=2,
        )
    writer.close()

    with h5py.File(path, "r") as data:
        assert not list(data.keys())
        assert int(data.attrs["samples"]) == 0
        assert int(data.attrs["threat_samples"]) == 0
