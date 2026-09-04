import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "applications" / "rl"))

from evaluate_velocity_dodge_oracle import (
    HDF5ImitationWriter,
    _limit_mpc_action_acceleration,
    _mpc_visibility_gate_open,
    oracle_action,
    project_sphere_to_pinhole,
    should_include_episode,
)
from neurosim.rl.receding_horizon_dodge_expert import (
    RecedingHorizonConfig,
    RecedingHorizonDodgeExpert,
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


def test_project_sphere_to_pinhole_uses_habitat_camera_convention():
    identity_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
    centre = project_sphere_to_pinhole(
        [0.0, 0.0, -5.0], [0.0, 0.0, 0.0], identity_wxyz,
        width=320, height=240, hfov_deg=90, radius_m=0.5,
    )
    assert centre[0] == 1.0
    assert centre[1] == pytest.approx(0.5)
    assert centre[2] == pytest.approx(0.5)
    assert centre[3] == pytest.approx(0.05)
    assert centre[4] == pytest.approx(5.0)

    right_and_up = project_sphere_to_pinhole(
        [1.0, 1.0, -5.0], [0.0, 0.0, 0.0], identity_wxyz,
        width=320, height=240, hfov_deg=90, radius_m=0.5,
    )
    assert right_and_up[1] > 0.5
    assert right_and_up[2] < 0.5
    assert project_sphere_to_pinhole(
        [0.0, 0.0, 5.0], [0.0, 0.0, 0.0], identity_wxyz,
        width=320, height=240, hfov_deg=90, radius_m=0.5,
    )[0] == 0.0
    assert not should_include_episode(
        {"success": False, "termination_reason": "out_of_bounds"}, True
    )
    assert should_include_episode(
        {"success": True, "termination_reason": "timeout"}, False
    )


def test_mpc_execution_projects_velocity_target_onto_acceleration_ball():
    task = type(
        "Task",
        (),
        {
            "action_filter_tau_s": 0.05,
            "delta_velocity_limits_mps": np.array([0.6, 0.6, 0.4]),
        },
    )()
    sim = type("Sim", (), {"config": type("Config", (), {"world_rate": 1000})()})()
    env = type(
        "Env",
        (),
        {
            "steps_per_action": 10,
            "sim": sim,
            "_task": task,
            "_filtered_action": np.zeros(3),
        },
    )()
    expert = RecedingHorizonDodgeExpert(
        RecedingHorizonConfig(max_residual_command_acceleration_mps2=5.0)
    )

    limited = _limit_mpc_action_acceleration(
        env, expert, np.ones(3, dtype=np.float32)
    )
    physical_target_gap = limited * task.delta_velocity_limits_mps

    # alpha=control_dt/tau=0.2, so a 5 m/s^2 first inner-control update
    # permits a raw target gap of 5 * 0.01 / 0.2 = 0.25 m/s.
    assert np.linalg.norm(physical_target_gap) == pytest.approx(0.25)


def test_mpc_visibility_gate_releases_once_and_invalidates_early_plan():
    task = type(
        "Task",
        (),
        {"trajectory_config": {"visibility_release_distance_m": 5.0}},
    )()
    coord = type("Coord", (), {"pos_transform_inv": np.eye(3)})()
    sim = type("Sim", (), {"coord_trans": coord})()
    env = type("Env", (), {"_task": task, "sim": sim})()
    item = type(
        "Item",
        (),
        {
            "encounter_id": 17,
            "object_id": 3,
            "obj": type("Object", (), {"translation": np.array([5.1, 0.0, 0.0])})(),
        },
    )()
    expert = type("Expert", (), {"plan": object()})()
    state = {"x": np.zeros(3)}

    assert not _mpc_visibility_gate_open(env, expert, state, {3: item}, 1.0)
    assert expert.plan is None
    assert not getattr(env, "_mpc_visibility_release_events", [])

    item.obj.translation = np.array([4.9, 0.0, 0.0])
    expert.plan = object()
    assert _mpc_visibility_gate_open(env, expert, state, {3: item}, 1.1)
    assert expert.plan is None
    assert env._mpc_visibility_released_ids == {17}
    assert env._mpc_visibility_release_events == [
        {"encounter_id": 17, "time_s": 1.1, "center_distance_m": 4.9}
    ]

    # The same encounter stays released without duplicating the event.
    expert.plan = object()
    assert _mpc_visibility_gate_open(env, expert, state, {3: item}, 1.2)
    assert expert.plan is not None
    assert len(env._mpc_visibility_release_events) == 1


def test_mpc_temporal_costs_prefer_continuing_the_current_plan():
    """Opt-in continuity costs must rank a committed maneuver first."""
    cfg = RecedingHorizonConfig(
        horizon_s=0.2,
        step_dt_s=0.05,
        control_segments=2,
        max_slack_cost=0.0,
        integrated_slack_cost=0.0,
        clearance_risk_cost=0.0,
        offset_cost=0.0,
        along_track_cost=0.0,
        vertical_cost=0.0,
        relative_velocity_cost=0.0,
        control_cost=0.0,
        control_smoothness_cost=0.0,
        terminal_offset_cost=0.0,
        terminal_velocity_cost=0.0,
        initial_control_smoothness_cost=1.0,
        plan_change_cost=1.0,
    )
    expert = RecedingHorizonDodgeExpert(cfg)
    continuing = np.array([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    switching = -continuing
    controls = np.stack([continuing, switching])
    rollout_kwargs = {
        "initial_offset": np.zeros(3),
        "initial_relative_velocity": np.zeros(3),
        "initial_filtered_action": np.array([0.5, 0.0, 0.0]),
        "body_to_world": np.eye(3),
        "delta_velocity_limits": np.ones(3),
        "action_filter_alpha": 1.0,
        "max_target_delta_velocity_mps": np.inf,
        "return_gain_hz": 0.0,
        "max_return_speed_mps": 1.0,
    }
    objective, _, _ = expert._score(
        controls,
        nominal_positions=np.zeros((4, 3)),
        nominal_velocity=np.zeros(3),
        obstacles=(),
        rollout_kwargs=rollout_kwargs,
        previous_controls=continuing,
    )

    assert objective[0] == pytest.approx(0.0)
    assert objective[1] > objective[0]


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
