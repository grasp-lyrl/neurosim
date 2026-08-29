"""The planner must propagate obstacles the way the simulator integrates them.

``DynamicObstacleManager._update_kinematic`` moves ``kinematic_parabola``
obstacles as ``p0 + v0*t - 0.5*g*t**2`` along Habitat -Y, but the expert
propagated every obstacle in a straight line from a *launch* velocity that
never updated. Combined error is ``g*t*T + 0.5*g*T**2`` -- roughly 7 m at 1 s
of flight over a 1.4 s planning horizon, against a 0.30 m combined hit
radius. The privileged oracle certified dodges at a median 0.216 m predicted
clearance that actually grazed at 0.007 m, and scored 7.5% against a 12.5%
do-nothing control.
"""

from types import SimpleNamespace

import numpy as np

from neurosim.rl.env_reactive_dodge import obstacle_acceleration, obstacle_velocity
from neurosim.rl.trajectory_dodge_expert import (
    LocalTrajectoryExpert,
    QuinticAvoidancePlan,
    TrajectoryExpertConfig,
)

GRAVITY = 3.0


def _item(motion_mode):
    return SimpleNamespace(
        motion_mode=motion_mode,
        velocity=np.array([5.0, 0.0, 0.0]),
        born_time=2.0,
        gravity_mps2=GRAVITY,
        obj=SimpleNamespace(linear_velocity=None),
    )


def test_parabola_velocity_tracks_gravity_while_line_stays_constant():
    parabola, line = _item("kinematic_parabola"), _item("kinematic_line")
    # At launch both report the launch velocity.
    np.testing.assert_allclose(obstacle_velocity(parabola, 2.0), [5.0, 0.0, 0.0])
    # One second later the parabola has fallen; the straight line has not.
    np.testing.assert_allclose(obstacle_velocity(parabola, 3.0), [5.0, -GRAVITY, 0.0])
    np.testing.assert_allclose(obstacle_velocity(line, 3.0), [5.0, 0.0, 0.0])
    # Omitting the clock preserves the launch value for callers with no time.
    np.testing.assert_allclose(obstacle_velocity(parabola), [5.0, 0.0, 0.0])


def test_acceleration_is_gravity_for_parabola_and_zero_otherwise():
    np.testing.assert_allclose(
        obstacle_acceleration(_item("kinematic_parabola")), [0.0, -GRAVITY, 0.0]
    )
    np.testing.assert_allclose(obstacle_acceleration(_item("kinematic_line")), np.zeros(3))


def test_plan_clears_accounts_for_acceleration():
    """An obstacle that falls into the path must invalidate the plan.

    Straight-line propagation keeps it clear overhead; the gravity term drops
    it onto the vehicle. Same geometry, opposite verdict.
    """
    expert = LocalTrajectoryExpert(TrajectoryExpertConfig(safety_margin_m=0.1))
    plan = QuinticAvoidancePlan(1, 0.0, 1.0, 2.0, np.zeros(3), 1.0)
    # Parked 3 m overhead, drifting sideways: on a straight line it stays there.
    position = np.array([0.0, 0.0, 3.0])
    velocity = np.zeros(3)
    falling = np.array([0.0, 0.0, -2.0 * 3.0 / (1.5**2)])  # reaches the path at t=1.5

    def nominal(_t):
        return np.zeros(3)

    assert expert.plan_clears(plan, 0.0, nominal, [(position, velocity, 0.25)])
    assert not expert.plan_clears(
        plan, 0.0, nominal, [(position, velocity, falling, 0.25)]
    )


def test_make_plan_rejects_a_dodge_the_falling_obstacle_would_intercept():
    """make_plan must not certify a plan into an accelerating obstacle."""
    expert = LocalTrajectoryExpert(
        TrajectoryExpertConfig(candidate_offsets_m=(0.6,), safety_margin_m=0.15)
    )
    common = dict(
        object_id=1,
        now=0.0,
        # Peak the bump at arrival, as test_planner_selects_collision_free_side
        # does -- otherwise the offset has already returned to zero when the
        # threat lands and no candidate is feasible for reasons unrelated to
        # the blocker under test.
        time_to_closest_approach=2.0,
        obstacle_position=np.array([2.0, 0.0, 0.0]),
        obstacle_velocity=np.array([-1.0, 0.0, 0.0]),
        combined_radius=0.25,
        nominal_position=lambda t: np.zeros(3),
        nominal_velocity=np.array([1.0, 0.0, 0.0]),
    )
    # A second obstacle sitting off to +y, stationary on a straight line, but
    # accelerating hard along -y straight through the +y dodge corridor.
    blocker = np.array([0.0, 2.2, 0.0])
    sweeping = np.array([0.0, -8.0, 0.0])
    straight = expert.make_plan(
        **common, additional_obstacles=[(2, blocker, np.zeros(3), 0.25)]
    )
    accelerating = expert.make_plan(
        **common,
        additional_obstacles=[(2, blocker, np.zeros(3), sweeping, 0.25)],
    )
    assert straight is not None, "sanity: the +y dodge is clear on a straight line"
    if accelerating is not None:
        # If it still finds a plan it must have gone the other way.
        assert accelerating.peak_offset[1] < 0.0
