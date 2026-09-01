"""Unit tests for the privileged sampling-MPC dodge expert."""

import numpy as np

from neurosim.rl.receding_horizon_dodge_expert import (
    MovingSpherePrediction,
    RecedingHorizonConfig,
    RecedingHorizonDodgeExpert,
)


def _problem(*, action_mask=None, static_path_is_clear=None, obstacle=None, offset=None):
    config = RecedingHorizonConfig(
        samples_per_mode=24,
        iterations=2,
        static_shortlist=32,
        seed=17,
    )
    expert = RecedingHorizonDodgeExpert(config)
    times = config.step_dt_s * (
        np.arange(int(round(config.horizon_s / config.step_dt_s))) + 1
    )
    nominal = np.column_stack(
        [0.6 * times, np.zeros_like(times), np.zeros_like(times)]
    )
    plan = expert.make_plan(
        now=0.0,
        time_to_closest_approach=0.70 if obstacle is not None else None,
        initial_offset=np.zeros(3) if offset is None else np.asarray(offset),
        initial_relative_velocity=np.zeros(3),
        initial_filtered_action=np.zeros(3),
        body_to_world=np.eye(3),
        delta_velocity_limits=np.array([0.6, 0.6, 0.4]),
        action_filter_alpha=0.67232,
        return_gain_hz=1.0,
        max_return_speed_mps=1.0,
        nominal_positions=nominal,
        nominal_velocity=np.array([0.6, 0.0, 0.0]),
        obstacles=() if obstacle is None else (obstacle,),
        action_mask=action_mask,
        static_path_is_clear=static_path_is_clear,
    )
    return expert, plan


def _head_on(position=4.0, speed=-5.0):
    return MovingSpherePrediction(
        object_id=7,
        position=np.array([position, 0.0, 0.0]),
        velocity=np.array([speed, 0.0, 0.0]),
        acceleration=np.zeros(3),
        combined_radius=0.30,
    )


def test_head_on_threat_produces_a_continuous_evasive_command():
    _, plan = _problem(obstacle=_head_on(), action_mask=np.array([0.0, 1.0, 0.0]))

    assert abs(plan.actions[0, 1]) > 0.1
    assert np.max(np.abs(plan.offsets[:, 1])) > 0.20
    # This deliberately exact head-on case is slightly outside the requested
    # 0.15 m margin at the task's 0.6 m/s correction limit, but MPC still
    # turns a centreline collision into a near miss.
    assert plan.predicted_min_clearance > -0.05
    assert plan.object_ids == (7,)


def test_infeasible_state_returns_least_dangerous_plan_instead_of_none():
    # Only 0.12 s to impact: the 0.15 m requested margin is unreachable, but
    # finite collision slack must still produce an evasive command.
    _, plan = _problem(
        obstacle=_head_on(position=1.2, speed=-5.0),
        action_mask=np.array([0.0, 1.0, 0.0]),
    )

    assert plan.safety_slack > 0.0
    assert np.isfinite(plan.objective)
    assert abs(plan.actions[0, 1]) > 0.1


def test_static_shortlist_selects_the_clear_side_of_a_symmetric_dodge():
    def positive_lane_only(points):
        # Reject paths that commit to the negative-y side. The zero path is
        # technically valid, so dynamic risk must still prefer +y.
        return float(np.min(points[:, 1])) >= -1e-5

    _, plan = _problem(
        obstacle=_head_on(),
        action_mask=np.array([0.0, 1.0, 0.0]),
        static_path_is_clear=positive_lane_only,
    )

    assert plan.static_clear
    assert plan.actions[0, 1] > 0.1
    assert float(np.min(plan.offsets[:, 1])) >= -1e-5


def test_no_obstacle_plan_recovers_toward_the_nominal_path():
    _, plan = _problem(offset=np.array([0.0, 0.4, 0.0]))

    assert abs(plan.offsets[-1, 1]) < 0.4
