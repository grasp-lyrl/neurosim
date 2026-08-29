import numpy as np

from neurosim.rl.trajectory_dodge_expert import (
    LocalTrajectoryExpert,
    QuinticAvoidancePlan,
    TrajectoryExpertConfig,
)


def test_quintic_bump_has_zero_velocity_and_acceleration_at_boundaries():
    plan = QuinticAvoidancePlan(7, 1.0, 2.0, 3.0, np.array([0.0, 0.6, 0.0]), 0.2)
    for time, expected in [(1.0, np.zeros(3)), (2.0, plan.peak_offset), (3.0, np.zeros(3))]:
        position, velocity, acceleration = plan.evaluate(time)
        np.testing.assert_allclose(position, expected, atol=1e-12)
        np.testing.assert_allclose(velocity, 0.0, atol=1e-12)
        np.testing.assert_allclose(acceleration, 0.0, atol=1e-12)


def test_planner_selects_collision_free_side_and_returns_to_nominal():
    expert = LocalTrajectoryExpert(
        TrajectoryExpertConfig(candidate_offsets_m=(0.4, 0.7), safety_margin_m=0.1)
    )
    plan = expert.make_plan(
        object_id=3,
        now=0.0,
        time_to_closest_approach=2.0,
        obstacle_position=np.array([2.0, 0.0, 0.0]),
        obstacle_velocity=np.array([-1.0, 0.0, 0.0]),
        combined_radius=0.25,
        nominal_position=lambda t: np.zeros(3),
        nominal_velocity=np.array([1.0, 0.0, 0.0]),
    )
    assert plan is not None
    assert plan.predicted_min_clearance >= 0.1
    np.testing.assert_allclose(plan.evaluate(plan.end_time)[0], 0.0)


def test_static_validator_can_reject_one_side():
    expert = LocalTrajectoryExpert(
        TrajectoryExpertConfig(
            candidate_offsets_m=(0.7,),
            safety_margin_m=0.05,
            max_horizontal_speed_mps=1.0,
        )
    )

    def only_negative_y(points):
        return float(np.max(points[:, 1])) <= 1e-9

    plan = expert.make_plan(
        object_id=4,
        now=0.0,
        time_to_closest_approach=2.0,
        obstacle_position=np.array([2.0, 0.0, 0.0]),
        obstacle_velocity=np.array([-1.0, 0.0, 0.0]),
        combined_radius=0.2,
        nominal_position=lambda t: np.zeros(3),
        nominal_velocity=np.array([1.0, 0.0, 0.0]),
        static_path_is_clear=only_negative_y,
    )
    assert plan is not None
    assert plan.peak_offset[1] < 0.0


def test_controller_residual_compensates_outer_loop():
    expert = LocalTrajectoryExpert()
    expert.plan = QuinticAvoidancePlan(1, 0.0, 1.0, 2.0, np.array([0.0, 0.5, 0.0]), 0.2)
    displacement, velocity, _ = expert.plan.evaluate(0.5)
    gain = np.array([1.5, 1.5, 2.5])
    residual = expert.world_velocity_residual(0.5, gain, gate=0.25)
    np.testing.assert_allclose(residual - 0.75 * gain * displacement, velocity)


def test_planner_rejects_untrackable_fast_bump():
    expert = LocalTrajectoryExpert(
        TrajectoryExpertConfig(
            candidate_offsets_m=(0.7,),
            minimum_rise_time_s=0.5,
            return_time_s=0.5,
            max_horizontal_speed_mps=0.5,
        )
    )
    plan = expert.make_plan(
        object_id=9,
        now=0.0,
        time_to_closest_approach=0.5,
        obstacle_position=np.array([10.0, 0.0, 0.0]),
        obstacle_velocity=np.zeros(3),
        combined_radius=0.1,
        nominal_position=lambda t: np.zeros(3),
        nominal_velocity=np.array([1.0, 0.0, 0.0]),
    )
    assert plan is None


def test_obstacle_relative_candidates_are_perpendicular_to_incoming_motion():
    nominal = np.array([1.0, 0.0, 0.0])
    obstacle = np.array([-1.0, 1.0, 0.5])
    relative = obstacle - nominal
    directions = LocalTrajectoryExpert.candidate_directions(nominal, obstacle)[4:]
    assert len(directions) == 8
    for direction in directions:
        assert abs(float(np.dot(direction, relative))) < 1e-12


def test_planner_checks_additional_moving_obstacles_and_selects_clear_side():
    expert = LocalTrajectoryExpert(
        TrajectoryExpertConfig(
            candidate_offsets_m=(0.7,),
            safety_margin_m=0.05,
            max_horizontal_speed_mps=1.0,
        )
    )
    plan = expert.make_plan(
        object_id=1,
        now=0.0,
        time_to_closest_approach=2.0,
        obstacle_position=np.array([2.0, 0.0, 0.0]),
        obstacle_velocity=np.array([-1.0, 0.0, 0.0]),
        combined_radius=0.2,
        nominal_position=lambda t: np.zeros(3),
        nominal_velocity=np.array([1.0, 0.0, 0.0]),
        directions=(np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0])),
        # A stationary sphere blocks the positive-y apex; negative-y remains
        # clear. This obstacle is not the primary trigger.
        additional_obstacles=((np.array([0.0, 0.7, 0.0]), np.zeros(3), 0.2),),
    )
    assert plan is not None
    assert plan.peak_offset[1] < 0.0


def test_replanned_quintic_is_continuous_from_current_offset_and_velocity():
    start_offset = np.array([0.1, -0.2, 0.05])
    start_velocity = np.array([0.0, 0.15, -0.02])
    plan = QuinticAvoidancePlan(
        2,
        3.0,
        4.5,
        6.0,
        np.array([0.0, 0.5, 0.0]),
        0.2,
        start_offset=start_offset,
        start_velocity=start_velocity,
        validated_object_ids=(2, 3),
    )
    position, velocity, acceleration = plan.evaluate(3.0)
    np.testing.assert_allclose(position, start_offset)
    np.testing.assert_allclose(velocity, start_velocity)
    np.testing.assert_allclose(acceleration, 0.0)
    peak_position, peak_velocity, peak_acceleration = plan.evaluate(4.5)
    np.testing.assert_allclose(peak_position, plan.peak_offset, atol=1e-12)
    np.testing.assert_allclose(peak_velocity, 0.0, atol=1e-12)
    np.testing.assert_allclose(peak_acceleration, 0.0, atol=1e-12)


def test_open_gate_removes_nominal_restoring_compensation():
    expert = LocalTrajectoryExpert()
    expert.plan = QuinticAvoidancePlan(1, 0.0, 1.0, 2.0, np.array([0.0, 0.5, 0.0]), 0.2)
    _, velocity, _ = expert.plan.evaluate(0.5)
    residual = expert.world_velocity_residual(0.5, np.ones(3), gate=1.0)
    np.testing.assert_allclose(residual, velocity)


def test_open_gate_corrects_toward_local_plan_on_dagger_state():
    expert = LocalTrajectoryExpert()
    expert.plan = QuinticAvoidancePlan(
        1, 0.0, 1.0, 2.0, np.array([0.0, 0.5, 0.0]), 0.2
    )
    desired, velocity, _ = expert.plan.evaluate(0.5)
    actual = desired + np.array([0.1, -0.2, 0.05])
    gain = np.array([1.5, 2.0, 2.5])
    residual = expert.world_velocity_residual(
        0.5, gain, gate=1.0, actual_displacement=actual
    )

    np.testing.assert_allclose(residual, velocity + gain * (desired - actual))
