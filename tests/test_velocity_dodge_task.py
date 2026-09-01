"""Unit tests for the velocity_dodge task and its residual-control mode.

These are deliberately habitat-free: they pin the numeric contracts of the
leaky-integrator reference shift, the body-frame observation layout, and
the reward structure without paying for a scene load.
"""

import numpy as np
import pytest

from neurosim.core.coord_trans import rotate_vector_by_quat
from neurosim.rl.env_reactive_dodge import (
    acceleration_limited_command,
    obstacle_threat_priority,
)
from neurosim.rl.tasks import build_task
from neurosim.rl.tasks.base import TaskStep

# rotorpy SE3Control stock gains, used to check the displacement ceiling.
KP_POS = np.array([6.5, 6.5, 15.0])
KD_POS = np.array([4.0, 4.0, 9.0])


def make_task(**overrides):
    config = {
        "w_pos": 0.5,
        "w_pos_deadband_m": 0.3,
        "w_vel": 0.25,
        "w_clearance": 1.0,
        "clearance_threshold_m": 1.5,
        "gamma_shaping": 0.994,
        "residual_control": {
            "mode": "velocity_delta_integrated",
            "delta_velocity_limits_mps": [0.6, 0.6, 0.4],
            "offset_tau_s": 1.5,
            "offset_max_m": [1.2, 1.2, 0.8],
        },
    }
    config.update(overrides)
    return build_task(task_name="velocity_dodge", **config)


def make_state(**overrides):
    state = {
        "x": np.zeros(3, dtype=np.float32),
        "v": np.zeros(3, dtype=np.float32),
        "q": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "w": np.zeros(3, dtype=np.float32),
    }
    state.update(overrides)
    return state


def make_step(task, state, action, prev_action=None):
    return TaskStep(
        state=state,
        base_state=np.zeros(13, dtype=np.float32),
        action=np.asarray(action, dtype=np.float32),
        prev_action=prev_action,
        sim_time=0.0,
        dt=0.02,
        event_manager=None,
        obs_mode="combined",
    )


def integrate_offset(task, delta_world, steps, dt=0.02):
    """Replicate the env's leaky integrator, returning (offset, applied_vel)."""
    offset = np.zeros(3)
    applied = np.zeros(3)
    for _ in range(steps):
        previous = offset.copy()
        offset = np.clip(
            offset + dt * (delta_world - offset / task.offset_tau_s),
            -task.offset_max_m,
            task.offset_max_m,
        )
        applied = (offset - previous) / dt
    return offset, applied


def test_obstacle_priority_prefers_incoming_threat_over_nearby_receding() -> None:
    receding = {
        "predicted_clearance": 0.50,
        "time_to_closest_approach": 0.0,
        "clearance": 0.50,
    }
    incoming = {
        "predicted_clearance": 0.15,
        "time_to_closest_approach": 0.8,
        "clearance": 4.0,
    }

    assert min([receding, incoming], key=obstacle_threat_priority) is incoming


# ---- Residual control -----------------------------------------------------


@pytest.mark.parametrize("dv", [0.2, 0.4, 0.6])
def test_held_correction_settles_at_tau_times_dv(dv):
    """The whole point of integrating: deviation is tau*dv, not (Kd/Kp)*dv."""
    task = make_task()
    offset, _ = integrate_offset(task, np.array([0.0, dv, 0.0]), steps=1000)

    assert offset[1] == pytest.approx(task.offset_tau_s * dv, rel=0.02)
    # ...which is far beyond what a reference-velocity-only correction can do.
    velocity_only_ceiling = (KD_POS[1] / KP_POS[1]) * dv
    assert offset[1] > 2.0 * velocity_only_ceiling


def test_offset_decays_back_to_nominal_when_correction_stops():
    """Returning to the path is structural, not something reward must force."""
    task = make_task()
    offset, _ = integrate_offset(task, np.array([0.0, 0.6, 0.0]), steps=1000)
    assert offset[1] > 0.5

    dt = 0.02
    for _ in range(int(5.0 * task.offset_tau_s / dt)):
        offset = offset + dt * (-offset / task.offset_tau_s)
    assert abs(offset[1]) < 0.05


def test_offset_clip_is_a_hard_bound_and_kills_velocity_feedforward():
    """At saturation the applied reference velocity must go to zero.

    The analytic ``dv - e/tau`` form leaves a residual feedforward there,
    which SE3 turns into extra (Kd/Kp)*v_residual of deviation -- so the
    clip would not actually bound how far the drone leaves the path.
    """
    task = make_task()
    offset, applied = integrate_offset(task, np.array([0.0, 5.0, 0.0]), steps=2000)

    assert offset[1] == pytest.approx(task.offset_max_m[1], abs=1e-9)
    assert applied[1] == pytest.approx(0.0, abs=1e-9)

    analytic = 5.0 - offset[1] / task.offset_tau_s
    assert analytic > 0.4  # the form we deliberately avoid


def test_action_is_three_dimensional():
    assert make_task().action_dim == 3


def test_rejects_non_velocity_residual_mode():
    with pytest.raises(ValueError, match="velocity_delta_integrated"):
        make_task(residual_control={"mode": "ctbr_delta"})


def test_rejects_non_positive_tau():
    with pytest.raises(ValueError, match="offset_tau_s"):
        make_task(
            residual_control={
                "mode": "velocity_delta_integrated",
                "offset_tau_s": 0.0,
            }
        )


def test_accepts_desired_body_offset_mode():
    task = make_task(
        residual_control={
            "mode": "desired_body_offset",
            "offset_max_m": [0.5, 0.5, 0.3],
            "offset_command_tau_s": 0.2,
            "offset_rate_limits_mps": [0.8, 0.8, 0.5],
        }
    )
    assert task.residual_control_mode == "desired_body_offset"
    np.testing.assert_allclose(task.offset_rate_limits_mps, [0.8, 0.8, 0.5])


def test_desired_offset_rejects_non_positive_rate_limit():
    with pytest.raises(ValueError, match="offset_rate_limits_mps"):
        make_task(
            residual_control={
                "mode": "desired_body_offset",
                "offset_rate_limits_mps": [0.8, 0.0, 0.5],
            }
        )


def test_accepts_cascaded_velocity_mode():
    task = make_task(
        residual_control={
            "mode": "cascaded_velocity",
            "delta_velocity_limits_mps": [0.7, 0.7, 0.4],
            "action_filter_tau_s": 0.08,
            "outer_position_gain_hz": [1.5, 1.5, 2.0],
            "max_outer_return_velocity_mps": [0.8, 0.8, 0.5],
        }
    )
    assert task.residual_control_mode == "cascaded_velocity"
    np.testing.assert_allclose(task.outer_position_gain_hz, [1.5, 1.5, 2.0])
    np.testing.assert_allclose(
        task.max_outer_return_velocity_mps, [0.8, 0.8, 0.5]
    )


def test_velocity_command_acceleration_limit_is_isotropic():
    previous = np.array([0.2, -0.1, 0.0])
    command = np.array([1.2, 1.9, 0.0])
    limited = acceleration_limited_command(command, previous, 2.0, 0.05)

    assert np.linalg.norm(limited - previous) == pytest.approx(0.1)
    np.testing.assert_allclose(
        (limited - previous) / np.linalg.norm(limited - previous),
        (command - previous) / np.linalg.norm(command - previous),
    )


def test_velocity_command_accepts_shared_acceleration_limit():
    task = make_task(
        residual_control={
            "mode": "velocity_command",
            "velocity_command_accel_limit_mps2": 2.0,
        }
    )
    assert task.velocity_command_accel_limit_mps2 == pytest.approx(2.0)


def test_velocity_command_rejects_negative_acceleration_limit():
    with pytest.raises(ValueError, match="velocity_command_accel_limit_mps2"):
        make_task(
            residual_control={
                "mode": "velocity_command",
                "velocity_command_accel_limit_mps2": -0.1,
            }
        )


def test_accepts_gated_cascaded_velocity_mode():
    task = make_task(
        residual_control={
            "mode": "gated_cascaded_velocity",
            "delta_velocity_limits_mps": [0.7, 0.7, 0.4],
            "action_filter_tau_s": 0.08,
            "outer_position_gain_hz": [1.5, 1.5, 2.0],
            "max_outer_return_velocity_mps": [0.8, 0.8, 0.5],
        }
    )
    assert task.action_dim == 4
    gate, delta = task.split_action(np.array([0.25, 0.0, 1.0, -0.5]))
    assert gate == pytest.approx(0.25)
    np.testing.assert_allclose(delta, [0.0, 0.25, -0.125])


def test_cascaded_velocity_rejects_non_positive_outer_gain():
    with pytest.raises(ValueError, match="outer_position_gain_hz"):
        make_task(
            residual_control={
                "mode": "cascaded_velocity",
                "outer_position_gain_hz": [1.5, 0.0, 2.0],
            }
        )


# ---- Observation ----------------------------------------------------------


def test_state_observation_is_body_frame_and_yaw_invariant():
    """Identical geometry at a different heading must give identical obs.

    This is the reason for body-frame: otherwise the policy has to relearn
    "dodge left" for every yaw.
    """
    task = make_task()

    def observe(yaw):
        quat = np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])
        world_v = rotate_vector_by_quat(np.array([1.0, 0.2, 0.0]), quat)
        world_ref = rotate_vector_by_quat(np.array([1.0, 0.0, 0.0]), quat)
        task.set_context(
            {
                "flat": {"x": np.zeros(3), "x_dot": world_ref},
                "previous_action": np.zeros(3, dtype=np.float32),
                "offset_cmd": np.zeros(3),
            }
        )
        return task.make_state_observation(
            state=make_state(v=world_v, q=quat), base_state=np.zeros(13)
        )

    np.testing.assert_allclose(observe(0.0), observe(1.1), atol=1e-5)
    np.testing.assert_allclose(observe(0.0)[:3], [1.0, 0.2, 0.0], atol=1e-5)
    # The actor needs both its measured body velocity and the reference body
    # velocity; a velocity error alone cannot distinguish a fast trajectory
    # from hovering with the same error. Keep their layout explicit because
    # imitation datasets and deployed checkpoints depend on it.
    np.testing.assert_allclose(observe(0.0)[3:6], [1.0, 0.0, 0.0], atol=1e-5)


def test_state_observation_dim_matches_declared():
    task = make_task()
    task.set_context({"flat": {"x": np.zeros(3), "x_dot": np.zeros(3)}})
    obs = task.make_state_observation(state=make_state(), base_state=np.zeros(13))
    assert obs.shape == (task.state_observation_dim,)
    assert obs.shape == (18,)


def test_observation_carries_the_action_just_applied():
    """The previous correction in the obs must not be a step stale.

    The env populates context before computing reward, when "previous
    action" still means a_{t-1}; the observation from that step drives the
    next decision, where it must mean a_t. Without the refresh the policy
    steers against a correction two decisions old.
    """
    task = make_task()
    task.set_context(
        {
            "flat": {"x": np.zeros(3), "x_dot": np.zeros(3)},
            "previous_action": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        }
    )
    applied = np.array([-0.7, 0.4, 0.0], dtype=np.float32)
    task.set_previous_action(applied)

    obs = task.make_state_observation(state=make_state(), base_state=np.zeros(13))
    np.testing.assert_allclose(obs[-3:], applied, atol=1e-6)


def test_privileged_observation_is_zero_without_obstacles():
    task = make_task()
    task.set_context({"obstacle_relative_states": []})
    privileged = task.make_privileged_observation(state=make_state())

    assert privileged.shape == (task.privileged_observation_dim,)
    assert not privileged.any()


def test_privileged_observation_reports_nearest_obstacle_in_body_frame():
    task = make_task()
    yaw = np.pi / 2
    quat = np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])
    task.set_context(
        {
            "obstacle_relative_states": [
                {
                    "rel_pos": np.array([0.0, 2.0, 0.0]),
                    "rel_vel": np.array([0.0, -1.0, 0.0]),
                    "clearance": 1.8,
                    "time_to_closest_approach": 0.4,
                }
            ]
        }
    )
    privileged = task.make_privileged_observation(state=make_state(q=quat))

    assert privileged[0] == 1.0
    # +y in world is straight ahead when yawed 90 deg.
    np.testing.assert_allclose(privileged[1:4], [2.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(privileged[4:7], [-1.0, 0.0, 0.0], atol=1e-5)
    assert privileged[7] == pytest.approx(1.8)


def test_privileged_channel_can_be_disabled():
    assert make_task(privileged_critic=False).privileged_observation_dim == 0


# ---- Reward ---------------------------------------------------------------


def test_tracking_penalty_is_zero_inside_the_deadband():
    task = make_task()
    task.set_context({"flat": {"x": np.array([0.2, 0.0, 0.0]), "x_dot": np.zeros(3)}})
    outcome = task.compute_reward(make_step(task, make_state(), np.zeros(3)))

    assert outcome.terms["pos_error"] == pytest.approx(0.2)
    assert outcome.terms["pos_excess"] == 0.0
    assert outcome.terms["r_track"] == pytest.approx(0.0)


def test_tracking_penalty_grows_outside_the_deadband():
    task = make_task()
    task.set_context({"flat": {"x": np.array([1.3, 0.0, 0.0]), "x_dot": np.zeros(3)}})
    outcome = task.compute_reward(make_step(task, make_state(), np.zeros(3)))

    assert outcome.terms["pos_excess"] == pytest.approx(1.0)
    assert outcome.terms["r_track"] == pytest.approx(-0.5)


def test_clearance_shaping_is_potential_based_and_telescopes():
    """Approach then retreat must net to ~0, so shaping cannot be farmed."""
    task = make_task()
    distances = [3.0, 1.2, 0.8, 1.2, 3.0]
    total = 0.0
    for distance in distances:
        task.set_context(
            {
                "flat": {"x": np.zeros(3), "x_dot": np.zeros(3)},
                "predicted_closest_distance": distance,
            }
        )
        outcome = task.compute_reward(make_step(task, make_state(), np.zeros(3)))
        total += outcome.terms["r_clearance_shaping"]

    assert abs(total) < 0.05


def test_closing_on_an_obstacle_is_penalised_by_shaping():
    task = make_task()
    for distance, expect_negative in [(3.0, False), (0.6, True)]:
        task.set_context(
            {
                "flat": {"x": np.zeros(3), "x_dot": np.zeros(3)},
                "predicted_closest_distance": distance,
            }
        )
        outcome = task.compute_reward(make_step(task, make_state(), np.zeros(3)))
        if expect_negative:
            assert outcome.terms["r_clearance_shaping"] < 0.0


def test_encounter_clear_bonus_is_awarded_once_when_obstacle_recedes():
    task = make_task(w_encounter_clear=4.0, dodge_clearance_m=0.25)
    base = {
        "flat": {"x": np.zeros(3), "x_dot": np.zeros(3)},
        "min_obstacle_distance": 0.4,
        "predicted_closest_distance": 0.4,
    }
    approaching = {
        "object_id": 7,
        "rel_pos": np.array([1.0, 0.0, 0.0]),
        "rel_vel": np.array([-1.0, 0.0, 0.0]),
        "clearance": 0.4,
    }
    receding = {
        **approaching,
        "rel_pos": np.array([0.7, 0.0, 0.0]),
        "rel_vel": np.array([1.0, 0.0, 0.0]),
    }

    task.set_context({**base, "obstacle_relative_states": [approaching]})
    first = task.compute_reward(make_step(task, make_state(), np.zeros(3)))
    task.set_context({**base, "obstacle_relative_states": [receding]})
    cleared = task.compute_reward(make_step(task, make_state(), np.zeros(3)))
    task.set_context({**base, "obstacle_relative_states": [receding]})
    repeated = task.compute_reward(make_step(task, make_state(), np.zeros(3)))

    assert first.terms["r_encounter_clear"] == 0.0
    assert cleared.terms["r_encounter_clear"] == pytest.approx(4.0)
    assert repeated.terms["r_encounter_clear"] == 0.0


def test_along_track_offset_is_penalised_more_than_cross_track():
    """Braking must cost strictly more than an equal-size lateral dodge.

    A throw solved for a fixed arrival time is defeated by simply being
    late, and that shortcut needs no perception at all -- so it has to be
    priced, or the policy collapses to "slow down for everything".
    """
    task = make_task()
    v_ref = np.array([1.0, 0.0, 0.0])  # travelling along +x
    magnitude = 0.4

    def reward_for(offset):
        task.on_reset()
        task.set_context(
            {
                "flat": {"x": np.zeros(3), "x_dot": v_ref},
                "offset_cmd": np.asarray(offset, dtype=np.float64),
            }
        )
        return task.compute_reward(make_step(task, make_state(), np.zeros(3)))

    braking = reward_for([-magnitude, 0.0, 0.0])  # lag along the path
    lateral = reward_for([0.0, magnitude, 0.0])  # step sideways

    assert braking.terms["offset_along_track"] == pytest.approx(-magnitude)
    assert braking.terms["offset_cross_track"] == pytest.approx(0.0, abs=1e-9)
    assert lateral.terms["offset_along_track"] == pytest.approx(0.0, abs=1e-9)
    assert lateral.terms["offset_cross_track"] == pytest.approx(magnitude)
    assert braking.reward < lateral.reward


def test_speeding_up_is_penalised_like_braking():
    """Running ahead of schedule evades the same way; price it symmetrically."""
    task = make_task()
    v_ref = np.array([1.0, 0.0, 0.0])

    def along_term(sign):
        task.on_reset()
        task.set_context(
            {
                "flat": {"x": np.zeros(3), "x_dot": v_ref},
                "offset_cmd": np.array([sign * 0.4, 0.0, 0.0]),
            }
        )
        return task.compute_reward(make_step(task, make_state(), np.zeros(3))).terms[
            "r_along_track"
        ]

    assert along_term(-1.0) == pytest.approx(along_term(+1.0))


def test_success_requires_an_actual_encounter():
    task = make_task()
    task.set_context(
        {"flat": {"x": np.zeros(3), "x_dot": np.zeros(3)}, "min_obstacle_distance": 9.0}
    )
    task.compute_reward(make_step(task, make_state(), np.zeros(3)))

    assert task.check_success(state=make_state()) is False


def test_success_after_a_close_pass_that_kept_clearance():
    task = make_task()
    for distance in [2.0, 0.9, 2.0]:
        task.set_context(
            {
                "flat": {"x": np.zeros(3), "x_dot": np.zeros(3)},
                "min_obstacle_distance": distance,
                "obstacle_threat": distance < 1.5,
            }
        )
        task.compute_reward(make_step(task, make_state(), np.zeros(3)))

    assert task.check_success(state=make_state()) is True


def test_no_success_when_clearance_dropped_below_threshold():
    task = make_task()
    for distance in [2.0, 0.1, 2.0]:
        task.set_context(
            {
                "flat": {"x": np.zeros(3), "x_dot": np.zeros(3)},
                "min_obstacle_distance": distance,
                "obstacle_threat": distance < 1.5,
            }
        )
        task.compute_reward(make_step(task, make_state(), np.zeros(3)))

    assert task.check_success(state=make_state()) is False


def test_per_encounter_metric_tracks_each_obstacle_separately():
    """Dense progress signal: episode success is conjunctive and stays 0.

    With several throws per episode, "survived every encounter" reads zero
    long after the policy starts clearing most of them, so the per-encounter
    rate is what actually shows learning.
    """
    task = make_task(clearance_threshold_m=1.5, dodge_clearance_m=0.1)

    def step_with(rows):
        task.set_context(
            {
                "flat": {"x": np.zeros(3), "x_dot": np.zeros(3)},
                "obstacle_relative_states": rows,
                "min_obstacle_distance": min(
                    (r["clearance"] for r in rows), default=np.inf
                ),
            }
        )
        return task.compute_reward(make_step(task, make_state(), np.zeros(3)))

    step_with([{"object_id": 1, "clearance": 0.8}])  # cleared
    step_with([{"object_id": 2, "clearance": 0.05}])  # hit
    outcome = step_with([{"object_id": 3, "clearance": 5.0}])  # never close

    assert outcome.terms["encounters_total"] == 2  # obstacle 3 was never near
    assert outcome.terms["encounters_cleared"] == 1
    assert outcome.terms["encounter_clear_rate"] == pytest.approx(0.5)


def test_encounter_metric_uses_closest_approach_not_last_seen():
    """A near miss must stay counted after the obstacle recedes."""
    task = make_task(clearance_threshold_m=1.5, dodge_clearance_m=0.1)
    for clearance in [1.2, 0.05, 1.2, 3.0]:
        task.set_context(
            {
                "flat": {"x": np.zeros(3), "x_dot": np.zeros(3)},
                "obstacle_relative_states": [{"object_id": 7, "clearance": clearance}],
                "min_obstacle_distance": clearance,
            }
        )
        outcome = task.compute_reward(make_step(task, make_state(), np.zeros(3)))

    assert outcome.terms["encounters_total"] == 1
    assert outcome.terms["encounters_cleared"] == 0


def test_encounter_metric_distinguishes_recycled_simulator_object_ids():
    """Sequential throws remain distinct when Habitat recycles object IDs."""
    task = make_task(clearance_threshold_m=1.5, dodge_clearance_m=0.1)

    for encounter_id, clearance in [(1, 0.8), (2, 0.05)]:
        task.set_context(
            {
                "flat": {"x": np.zeros(3), "x_dot": np.zeros(3)},
                "obstacle_relative_states": [
                    {
                        "object_id": 7,
                        "encounter_id": encounter_id,
                        "clearance": clearance,
                    }
                ],
                "min_obstacle_distance": clearance,
            }
        )
        outcome = task.compute_reward(make_step(task, make_state(), np.zeros(3)))

    assert outcome.terms["encounters_total"] == 2
    assert outcome.terms["encounters_cleared"] == 1
    assert outcome.terms["encounter_clear_rate"] == pytest.approx(0.5)


def test_crash_penalty_grows_with_the_episode_a_crash_forfeits():
    """Dying early must not out-earn flying on.

    A flat terminal penalty only deters crashing while the per-step reward
    stays above ``-penalty / remaining_steps``. Two training runs fell
    through that floor and learned to crash on purpose, so the penalty now
    scales with the time forfeited.
    """
    task = make_task(
        crash_penalty_obstacle=150.0,
        crash_penalty_per_remaining_step=0.6,
    )
    task.set_termination_reason("obstacle_collision")

    task.set_context({"remaining_steps": 0.0})
    assert task.crash_penalty == pytest.approx(150.0)

    task.set_context({"remaining_steps": 600.0})
    assert task.crash_penalty == pytest.approx(150.0 + 0.6 * 600.0)

    # The property is what decides whether crashing beats surviving, so
    # check that comparison directly at a per-step rate that beat the old
    # flat-150 scheme (v9 measured -0.4 against a -0.246 break-even).
    rate, episode_steps, crash_at = -0.4, 750.0, 150.0
    task.set_context({"remaining_steps": episode_steps - crash_at})
    survive = rate * episode_steps
    die = rate * crash_at - task.crash_penalty
    assert die < survive

    # Default off, so other configs keep the plain flat penalty.
    default_task = make_task(crash_penalty_obstacle=150.0)
    default_task.set_termination_reason("obstacle_collision")
    default_task.set_context({"remaining_steps": 600.0})
    assert default_task.crash_penalty == pytest.approx(150.0)


def test_no_threat_offset_penalty_only_bites_when_nothing_is_inbound():
    """Deviation must be free under threat and charged once the sky is clear.

    The uniform tracking penalty cannot express this distinction: it prices
    the excursion a genuine dodge requires exactly as hard as idle
    loitering off the path, so raising it to discourage loitering also
    taxes dodging. Every run instead converged on a permanent ~0.6 m offset
    -- which is close to correct when the drone is flagged in-threat on
    96% of steps.
    """
    task = make_task(
        w_no_threat_offset=0.5,
        w_pos=0.0,
        w_vel=0.0,
        w_clearance=0.0,
        w_pos_deadband_m=0.0,
    )
    away = make_state(x=np.array([0.0, 0.6, 0.0], dtype=np.float32))

    task.on_reset()
    task.set_context(
        {"obstacle_threat": True, "predicted_closest_distance": 0.0, "flat": {"x": np.zeros(3), "x_dot": np.zeros(3), "yaw": 0.0}}
    )
    under_threat = task.compute_reward(make_step(task, away, np.zeros(3)))

    task.on_reset()
    task.set_context({"obstacle_threat": False, "flat": {"x": np.zeros(3), "x_dot": np.zeros(3), "yaw": 0.0}})
    when_clear = task.compute_reward(make_step(task, away, np.zeros(3)))

    # Charge now scales with how close the nearest predicted approach is,
    # rather than switching on a binary flag -- at the current obstacle
    # density that flag is true on ~80% of steps, so an on/off term is
    # inactive four steps in five and cannot do its job.
    assert under_threat.terms["r_no_threat_offset"] == pytest.approx(0.0)
    assert when_clear.terms["r_no_threat_offset"] == pytest.approx(-0.5 * 0.6)
    assert when_clear.reward < under_threat.reward

    # Sitting on the nominal path costs nothing either way.
    task.on_reset()
    task.set_context({"obstacle_threat": False, "flat": {"x": np.zeros(3), "x_dot": np.zeros(3), "yaw": 0.0}})
    on_path = task.compute_reward(make_step(task, make_state(), np.zeros(3)))
    assert on_path.terms["r_no_threat_offset"] == pytest.approx(0.0)

    # Off by default, so other configs are unaffected.
    plain = make_task(w_pos=0.0, w_vel=0.0, w_clearance=0.0, w_pos_deadband_m=0.0)
    plain.on_reset()
    plain.set_context({"obstacle_threat": False, "flat": {"x": np.zeros(3), "x_dot": np.zeros(3), "yaw": 0.0}})
    assert plain.compute_reward(
        make_step(plain, away, np.zeros(3))
    ).terms["r_no_threat_offset"] == pytest.approx(0.0)


def test_no_threat_penalty_prices_deviation_not_the_commanded_offset():
    """The exploit lives in the deviation the command does not account for.

    Measured on v24 over 20 stochastic episodes, no-threat steps ran
    pos_error 1.633 m against an offset_cmd of only 0.280 m. Charging the
    command left 83% of the excursion free -- and since obstacles are aimed
    once at spawn with no re-solve, that unpriced drift is exactly what
    makes them miss (threat fraction 51.5% deterministic vs 20.8%
    stochastic). A drone far off the path must be charged even when it is
    commanding nothing at all.
    """
    task = make_task(
        w_no_threat_offset=0.5,
        w_pos=0.0,
        w_vel=0.0,
        w_clearance=0.0,
        w_pos_deadband_m=0.0,
    )
    drifted = make_state(x=np.array([0.0, 1.6, 0.0], dtype=np.float32))

    task.on_reset()
    task.set_context(
        {"obstacle_threat": False, "offset_cmd": np.zeros(3), "flat": {"x": np.zeros(3), "x_dot": np.zeros(3), "yaw": 0.0}}
    )
    outcome = task.compute_reward(make_step(task, drifted, np.zeros(3)))
    assert outcome.terms["r_no_threat_offset"] == pytest.approx(-0.5 * 1.6)

    # Symmetrically, commanding an offset the vehicle has not taken yet is
    # not charged -- the term follows where the drone is, not what it asked
    # for, so a dodge is never billed before it has moved.
    task.on_reset()
    task.set_context(
        {
            "obstacle_threat": False,
            "offset_cmd": np.array([0.0, 1.6, 0.0]),
            "flat": {"x": np.zeros(3), "x_dot": np.zeros(3), "yaw": 0.0},
        }
    )
    commanded_only = task.compute_reward(
        make_step(task, make_state(), np.zeros(3))
    )
    assert commanded_only.terms["r_no_threat_offset"] == pytest.approx(0.0)


def test_no_threat_penalty_shares_the_tracking_error_cap():
    """Capped like r_track, so one bad excursion cannot dominate a return."""
    task = make_task(
        w_no_threat_offset=0.5,
        w_pos=0.0,
        w_vel=0.0,
        w_clearance=0.0,
        w_pos_deadband_m=0.0,
        track_pos_error_cap_m=1.0,
    )
    task.on_reset()
    task.set_context({"obstacle_threat": False, "flat": {"x": np.zeros(3), "x_dot": np.zeros(3), "yaw": 0.0}})
    far = make_state(x=np.array([0.0, 2.5, 0.0], dtype=np.float32))
    outcome = task.compute_reward(make_step(task, far, np.zeros(3)))
    assert outcome.terms["r_no_threat_offset"] == pytest.approx(-0.5 * 1.0)


def test_gated_velocity_delta_closes_the_correction_without_changing_control():
    """A closed gate must zero the correction, leaving the same controller.

    The earlier gated experiment (``gated_cascaded_velocity``) changed the
    control law itself -- the gate released an outer restoring loop, which
    removed the displacement bound and cost a third of the achievable
    ceiling (privileged oracle 36.7% -> 23.3%). This variant gates only the
    policy output: identical integrator, identical clip, so ``gate=1``
    reproduces the ungated mode exactly and ``gate=0`` hands the controller
    a zero delta, letting the leaky integrator decay back to nominal.
    """
    gated = make_task(
        residual_control={
            "mode": "gated_velocity_delta_integrated",
            "delta_velocity_limits_mps": [0.6, 0.6, 0.4],
            "offset_tau_s": 1.5,
            "offset_max_m": [1.2, 1.2, 0.8],
        }
    )
    plain = make_task()
    assert gated.action_dim == 4
    assert plain.action_dim == 3

    direction = np.array([1.0, -0.5, 0.25], dtype=np.float32)

    # Fully open reproduces the ungated split exactly.
    _, delta_open = gated.split_action(np.concatenate(([1.0], direction)))
    _, delta_plain = plain.split_action(direction)
    np.testing.assert_allclose(delta_open, delta_plain, atol=1e-7)

    # Closed produces exactly zero, whatever the direction head says.
    gate_closed, delta_closed = gated.split_action(
        np.array([0.0, 1.0, -1.0, 1.0], dtype=np.float32)
    )
    assert gate_closed == pytest.approx(0.0)
    np.testing.assert_allclose(delta_closed, 0.0, atol=1e-7)

    # And it scales in between, so the gate is a usable continuous control.
    gate_half, delta_half = gated.split_action(np.concatenate(([0.5], direction)))
    assert gate_half == pytest.approx(0.5)
    np.testing.assert_allclose(delta_half, 0.5 * direction, atol=1e-7)

    # Out-of-range gate values are clipped rather than inverting the dodge.
    gate_neg, delta_neg = gated.split_action(np.concatenate(([-3.0], direction)))
    assert gate_neg == pytest.approx(0.0)
    np.testing.assert_allclose(delta_neg, 0.0, atol=1e-7)


def test_gated_velocity_delta_offset_decays_when_the_gate_closes():
    """Closing the gate must let the integrator return to the nominal path."""
    task = make_task(
        residual_control={
            "mode": "gated_velocity_delta_integrated",
            "delta_velocity_limits_mps": [0.6, 0.6, 0.4],
            "offset_tau_s": 1.5,
            "offset_max_m": [1.2, 1.2, 0.8],
        }
    )
    # Hold the gate open to build a real displacement...
    _, delta = task.split_action(np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32))
    offset, _ = integrate_offset(task, delta * task.delta_velocity_limits_mps, steps=200)
    assert np.linalg.norm(offset) > 0.5

    # ...then close it: the delta is zero, so the leak drains the offset.
    decayed = offset.copy()
    dt = 0.02
    for _ in range(200):
        decayed = decayed + dt * (np.zeros(3) - decayed / task.offset_tau_s)
    assert np.linalg.norm(decayed) < 0.1 * np.linalg.norm(offset)


def test_threat_gate_silences_the_policy_until_something_is_inbound():
    """With nothing inbound the policy commands nothing, mechanically.

    Reward shaping could not achieve this. Measured on v24, no-threat steps
    ran 1.633 m off the nominal path while the *commanded* offset was only
    0.280 m -- most of the excursion was the controller chasing a reference
    that sampled actions were shaking, which no penalty on the command can
    reach.
    """
    task = make_task(threat_gated_actions=True)
    action = np.ones(3, dtype=np.float32)

    task.on_reset()
    task.set_context({"obstacle_threat": False})
    _, quiet = task.split_action(action)
    assert not np.any(quiet)

    task.set_context({"obstacle_threat": True})
    _, inbound = task.split_action(action)
    assert np.any(inbound)


def test_threat_gate_is_off_by_default():
    """Every existing config must be unaffected."""
    task = make_task()
    task.on_reset()
    task.set_context({"obstacle_threat": False})
    _, delta = task.split_action(np.ones(3, dtype=np.float32))
    assert np.any(delta)


def test_threat_gate_closes_the_gate_channel_for_gated_modes():
    """Gated modes derive their gate from action[0]; it must clip to zero."""
    task = make_task(
        threat_gated_actions=True,
        residual_control={
            "mode": "gated_velocity_delta_integrated",
            "delta_velocity_limits_mps": [0.6, 0.6, 0.4],
            "offset_tau_s": 1.5,
            "offset_max_m": [1.2, 1.2, 0.8],
        },
    )
    task.on_reset()
    task.set_context({"obstacle_threat": False})
    gate, delta = task.split_action(np.ones(4, dtype=np.float32))
    assert gate == pytest.approx(0.0)
    assert not np.any(delta)


def test_no_threat_charge_ramps_with_threat_distance():
    """Coverage must not collapse when something is almost always inbound.

    At the scattered density the threat flag is true on ~80% of steps
    (measured, zero-action), so a binary on/off term is inactive four steps
    in five and cannot charge for holding an offset. Scaling by predicted
    clearance restores partial credit on those steps without re-taxing a
    dodge that is genuinely close to something.
    """
    task = make_task(
        w_no_threat_offset=0.5,
        w_pos=0.0,
        w_vel=0.0,
        w_clearance=0.0,
        w_pos_deadband_m=0.0,
        threat_distance_m=1.5,
    )
    away = make_state(x=np.array([0.0, 1.0, 0.0], dtype=np.float32))
    flat = {"x": np.zeros(3), "x_dot": np.zeros(3), "yaw": 0.0}

    def charge(predicted):
        task.on_reset()
        task.set_context(
            {
                "obstacle_threat": True,
                "predicted_closest_distance": predicted,
                "flat": flat,
            }
        )
        return task.compute_reward(
            make_step(task, away, np.zeros(3))
        ).terms["r_no_threat_offset"]

    # Touching: a dodge in progress pays nothing.
    assert charge(0.0) == pytest.approx(0.0)
    # At twice the threat distance and beyond: full price.
    assert charge(3.0) == pytest.approx(-0.5 * 1.0)
    assert charge(9.0) == pytest.approx(-0.5 * 1.0)
    # In between it ramps, monotonically.
    mid = [charge(d) for d in (0.5, 1.0, 1.5, 2.0, 2.5)]
    assert all(b <= a for a, b in zip(mid, mid[1:]))
    assert charge(1.5) == pytest.approx(-0.5 * 0.5)


def test_no_tracked_obstacle_reads_as_fully_clear():
    """An empty sky must charge full price, not be treated as threatened."""
    task = make_task(
        w_no_threat_offset=0.5, w_pos=0.0, w_vel=0.0, w_clearance=0.0,
        w_pos_deadband_m=0.0,
    )
    task.on_reset()
    task.set_context(
        {
            "obstacle_threat": False,
            "predicted_closest_distance": float("inf"),
            "flat": {"x": np.zeros(3), "x_dot": np.zeros(3), "yaw": 0.0},
        }
    )
    outcome = task.compute_reward(
        make_step(task, make_state(x=np.array([0.0, 1.0, 0.0], dtype=np.float32)),
                  np.zeros(3))
    )
    assert outcome.terms["r_no_threat_offset"] == pytest.approx(-0.5)
