"""The expert must plan only in directions the action space can express."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "applications" / "rl"))

pytest.importorskip("wandb")

from evaluate_velocity_dodge_oracle import (  # noqa: E402
    expert_directions,
    expert_for_task,
)
from neurosim.rl.trajectory_dodge_expert import (  # noqa: E402
    TrajectoryExpertConfig,
)


class _Task:
    def __init__(self, **flags):
        self.lateral_axis_only = flags.get("lateral_axis_only", False)
        self.lateral_only = flags.get("lateral_only", False)


class _Env:
    def __init__(self, velocity, **flags):
        self._task = _Task(**flags)
        self._velocity = np.asarray(velocity, dtype=np.float64)

    def _nominal_flat(self):
        return {"x_dot": self._velocity}


def test_lateral_axis_only_plans_purely_sideways():
    """A vertical plan is deleted in full by the task's axis zeroing.

    split_action zeroes body x and z *after* the expert has chosen, so a
    vertical dodge reaches the controller as nothing at all while the expert
    believes it is avoiding. Measured 0/10 successes that way against 55%
    with the vertical axis available.
    """
    directions = expert_directions(_Env([5.0, 0.0, 0.0], lateral_axis_only=True))

    assert len(directions) == 2
    for direction in directions:
        assert direction[0] == pytest.approx(0.0, abs=1e-9)
        assert direction[2] == pytest.approx(0.0, abs=1e-9)
        assert abs(direction[1]) == pytest.approx(1.0)
    # Both signs, so the expert can still choose which way to go.
    assert np.dot(directions[0], directions[1]) == pytest.approx(-1.0)


def test_lateral_is_perpendicular_to_travel_whatever_the_heading():
    """The restriction follows the flight direction, not the world axes."""
    for velocity in ([5.0, 0.0, 0.0], [0.0, 4.0, 0.0], [3.0, 3.0, 0.0]):
        directions = expert_directions(_Env(velocity, lateral_axis_only=True))
        horizontal = np.array([velocity[0], velocity[1], 0.0])
        horizontal /= np.linalg.norm(horizontal)
        for direction in directions:
            assert np.dot(direction, horizontal) == pytest.approx(0.0, abs=1e-9)
            assert direction[2] == pytest.approx(0.0, abs=1e-9)


def test_climb_rate_does_not_tilt_the_lateral_axis():
    """A climbing nominal must still dodge in the horizontal plane."""
    directions = expert_directions(_Env([5.0, 0.0, 2.0], lateral_axis_only=True))
    for direction in directions:
        assert direction[2] == pytest.approx(0.0, abs=1e-9)


def test_degenerate_hover_still_yields_a_usable_axis():
    """A near-zero horizontal velocity must not divide by zero."""
    directions = expert_directions(_Env([0.0, 0.0, 1.0], lateral_axis_only=True))
    assert len(directions) == 2
    for direction in directions:
        assert np.isfinite(direction).all()
        assert np.linalg.norm(direction) == pytest.approx(1.0)


def test_unrestricted_tasks_keep_the_planners_own_directions():
    """make_plan treats a falsy directions argument as 'choose for me'.

    lateral_only leaves the vertical axis available, and the gate measured
    the oracle at 55% there, so that case must not be narrowed.
    """
    assert expert_directions(_Env([5.0, 0.0, 0.0])) is None
    assert expert_directions(_Env([5.0, 0.0, 0.0], lateral_only=True)) is None


def _offset_task(**flags):
    task = _Task(**flags)
    task.residual_control_mode = flags.get(
        "residual_control_mode", "desired_body_offset"
    )
    task.offset_max_m = np.array([0.56, 0.56, 0.36])
    return task


def test_candidates_are_capped_to_what_the_action_space_can_reach():
    """A plan the vehicle cannot execute is validated dishonestly.

    Under desired_body_offset the action is normalised by offset_max_m and
    clipped to +-1, capping the lateral axis at 0.56 m. The planner's default
    candidates run to 1.00 m, so it certified swept clearance for a
    displacement the vehicle stops short of -- and the "safe" plan collided.
    """
    expert = expert_for_task(_offset_task(lateral_axis_only=True))
    candidates = expert.config.candidate_offsets_m

    assert max(candidates) == pytest.approx(0.56)
    assert all(m <= 0.56 + 1e-9 for m in candidates)
    # The cap itself must be offered, or the expert gives up authority it has.
    assert any(m == pytest.approx(0.56) for m in candidates)
    # And the reachable defaults survive, so the search keeps its range.
    assert 0.35 in candidates


def test_unrestricted_offset_tasks_keep_the_full_candidate_set():
    """With the vertical axis in play the reachable cap is direction-dependent.

    make_plan takes one candidate list for every direction, so a single
    conservative number would collapse the set to (0.35, 0.36) and weaken an
    oracle that measures 55% as it stands.
    """
    default = TrajectoryExpertConfig().candidate_offsets_m
    assert expert_for_task(_offset_task()).config.candidate_offsets_m == default
    assert (
        expert_for_task(_offset_task(lateral_only=True)).config.candidate_offsets_m
        == default
    )


def test_velocity_modes_are_left_alone():
    """candidate_offsets_m is a displacement; those modes normalise by speed."""
    task = _offset_task(
        lateral_axis_only=True, residual_control_mode="velocity_delta_integrated"
    )
    assert (
        expert_for_task(task).config.candidate_offsets_m
        == TrajectoryExpertConfig().candidate_offsets_m
    )


def test_plan_clears_keeps_a_still_safe_plan():
    """A newly actionable obstacle must not by itself discard a good plan.

    With static obstacles nothing about the world changes as the drone
    advances -- only which obstacles cross a proximity threshold. Replanning
    on that signal made the vehicle reverse direction mid-dodge, visible on
    video as going one way and then abruptly the other.
    """
    from neurosim.rl.trajectory_dodge_expert import (  # noqa: PLC0415
        LocalTrajectoryExpert,
        QuinticAvoidancePlan,
    )

    expert = LocalTrajectoryExpert()
    plan = QuinticAvoidancePlan(
        object_id=1,
        start_time=0.0,
        peak_time=0.5,
        end_time=1.5,
        peak_offset=np.array([0.0, 0.8, 0.0]),
        predicted_min_clearance=1.0,
        start_offset=np.zeros(3),
        start_velocity=np.zeros(3),
        start_acceleration=np.zeros(3),
        validated_object_ids=(1,),
    )

    def nominal(t):
        return np.array([float(t), 0.0, 0.0])

    # An obstacle far off the path leaves the plan valid.
    far = [(np.array([0.0, -6.0, 0.0]), np.zeros(3), 0.3)]
    assert expert.plan_clears(plan, 0.0, nominal, far)

    # One sitting on the planned bump does not.
    near = [(np.array([0.5, 0.8, 0.0]), np.zeros(3), 0.3)]
    assert not expert.plan_clears(plan, 0.0, nominal, near)


def test_plan_clears_rejects_an_expired_plan():
    from neurosim.rl.trajectory_dodge_expert import (  # noqa: PLC0415
        LocalTrajectoryExpert,
        QuinticAvoidancePlan,
    )

    expert = LocalTrajectoryExpert()
    plan = QuinticAvoidancePlan(
        object_id=1, start_time=0.0, peak_time=0.5, end_time=1.0,
        peak_offset=np.array([0.0, 0.5, 0.0]), predicted_min_clearance=1.0,
        start_offset=np.zeros(3), start_velocity=np.zeros(3),
        start_acceleration=np.zeros(3), validated_object_ids=(1,),
    )
    assert not expert.plan_clears(
        plan, 2.0, lambda t: np.array([float(t), 0.0, 0.0]), []
    )
