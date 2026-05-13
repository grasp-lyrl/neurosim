"""Unit tests for the reactive_dodge task and residual CTBR control."""

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("magnum", reason="Habitat runtime dependencies not available")

from neurosim.rl.tasks import ReactiveDodgeTask, TaskStep, build_task
from neurosim.rl.vehicles.ctbr_rotorpy import RateLimits, RotorpyCtbrVehicle


def _step(
    *,
    state: dict[str, np.ndarray],
    action: np.ndarray,
    prev_action: np.ndarray | None = None,
    sim_time: float = 0.0,
) -> TaskStep:
    base = np.concatenate(
        [state["x"], state["v"], state["q"], state["w"]], dtype=np.float32
    )
    return TaskStep(
        state=state,
        base_state=base,
        action=action,
        prev_action=prev_action,
        sim_time=sim_time,
        dt=0.01,
        event_manager=_EventManager(),
        obs_mode="state",
    )


def _state() -> dict[str, np.ndarray]:
    return {
        "x": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "v": np.array([0.1, 0.0, -0.1], dtype=np.float32),
        "q": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "w": np.array([0.01, 0.02, 0.03], dtype=np.float32),
    }


def _flat() -> dict[str, np.ndarray | float]:
    return {
        "x": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "x_dot": np.array([0.1, 0.0, -0.1], dtype=np.float32),
        "yaw": 0.0,
    }


class _EventManager:
    step_event_count = 0
    raw_height = 10
    raw_width = 10


def test_build_task_registers_reactive_dodge():
    task = build_task("reactive_dodge")
    assert isinstance(task, ReactiveDodgeTask)
    assert task.uses_nominal_controller
    assert task.action_dim == 4


def test_reactive_dodge_state_observation_includes_tracking_context():
    task = ReactiveDodgeTask(lookahead_seconds=[0.25, 0.5])
    state = _state()
    base = np.arange(13, dtype=np.float32)
    task.set_context(
        {
            "flat": _flat(),
            "lookahead_errors": [
                np.array([0.1, 0.2, 0.3], dtype=np.float32),
                np.array([0.4, 0.5, 0.6], dtype=np.float32),
            ],
            "nominal_control_normalized": np.array([0.0, 0.1, 0.2, 0.3]),
            "previous_action": np.array([0.0, 0.0, 0.0, 0.0]),
        }
    )

    obs = task.make_state_observation(state=state, base_state=base)

    assert obs.shape == (task.state_observation_dim,)
    assert np.allclose(obs[:13], base)
    assert np.isfinite(obs).all()


def test_reactive_dodge_penalizes_unneeded_correction_more_than_threat_correction():
    state = _state()
    action = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)
    task = ReactiveDodgeTask(w_no_threat_correction=1.0)

    task.set_context({"flat": _flat(), "obstacle_threat": False})
    no_threat = task.compute_reward(_step(state=state, action=action))

    task.set_context({"flat": _flat(), "obstacle_threat": True})
    threat = task.compute_reward(_step(state=state, action=action))

    assert no_threat.reward < threat.reward


def test_gated_reactive_dodge_action_uses_five_dimensions():
    task = ReactiveDodgeTask(residual_control={"mode": "gated_ctbr_delta"})
    assert task.action_dim == 5

    state = _state()
    base = np.arange(13, dtype=np.float32)
    task.set_context({"flat": _flat()})
    obs = task.make_state_observation(state=state, base_state=base)
    assert obs.shape == (task.state_observation_dim,)


def test_reactive_dodge_success_requires_meaningful_encounter_and_recovery():
    task = ReactiveDodgeTask(
        require_obstacle_encounter=True,
        recovery_pos_error_m=0.25,
        recovery_window_s=1.0,
    )
    state = _state()

    task.set_context(
        {
            "flat": _flat(),
            "sim_time": 0.0,
            "obstacle_threat": True,
            "threat_ids": (7,),
        }
    )
    task.compute_reward(
        _step(state=state, action=np.zeros(4, dtype=np.float32), sim_time=0.0)
    )
    assert not task.check_success(state=state)

    task.set_context(
        {
            "flat": _flat(),
            "sim_time": 0.5,
            "obstacle_threat": False,
            "near_miss_ids": (7,),
        }
    )
    outcome = task.compute_reward(
        _step(state=state, action=np.zeros(4, dtype=np.float32), sim_time=0.5)
    )

    assert outcome.terms["meaningful_encounter_count"] == pytest.approx(1.0)
    assert outcome.terms["dodge_success_count"] == pytest.approx(1.0)
    assert task.check_success(state=state)


def test_reactive_dodge_tracking_failure_terminates_after_configured_steps():
    task = ReactiveDodgeTask(
        tracking_failure_pos_error_m=0.5,
        tracking_failure_steps=2,
    )
    state = _state()
    task.set_context(
        {
            "flat": {
                "x": np.zeros(3, dtype=np.float32),
                "x_dot": state["v"],
                "yaw": 0.0,
            }
        }
    )

    for _ in range(2):
        task.compute_reward(_step(state=state, action=np.zeros(4, dtype=np.float32)))

    terminated, reason = task.check_terminated(state=state)
    assert terminated
    assert reason == "tracking_failure"
    assert task.crash_penalty == pytest.approx(task.tracking_failure_penalty)


def test_ctbr_delta_control_clips_to_vehicle_bounds():
    fake_multirotor = SimpleNamespace(
        mass=0.03,
        g=9.81,
        num_rotors=4,
        k_eta=2.3e-8,
        k_m=7.8e-10,
        rotor_speed_min=0.0,
        rotor_speed_max=2500.0,
    )
    vehicle = RotorpyCtbrVehicle(
        dynamics=SimpleNamespace(_multirotor=fake_multirotor),
        vehicle="crazyflie",
        rate_limits=RateLimits(roll=1.0, pitch=1.0, yaw=0.5),
    )
    nominal = {
        "cmd_thrust": vehicle.hover_thrust,
        "cmd_w": np.array([0.9, -0.9, 0.4], dtype=np.float64),
    }

    merged = vehicle.apply_ctbr_delta(
        nominal,
        np.array([1.0, 1.0, -1.0, 1.0], dtype=np.float32),
        delta_thrust_fraction=10.0,
        delta_rate_limits=np.array([2.0, 2.0, 2.0]),
    )

    low, high = vehicle.control_bounds
    cmd = np.array([merged["cmd_thrust"], *merged["cmd_w"]], dtype=np.float64)
    assert np.all(cmd <= high)
    assert np.all(cmd >= low)
    assert cmd[1] == pytest.approx(high[1])
    assert cmd[2] == pytest.approx(low[2])
    assert cmd[3] == pytest.approx(high[3])


def test_gated_ctbr_delta_returns_nominal_when_gate_is_zero():
    fake_multirotor = SimpleNamespace(
        mass=0.03,
        g=9.81,
        num_rotors=4,
        k_eta=2.3e-8,
        k_m=7.8e-10,
        rotor_speed_min=0.0,
        rotor_speed_max=2500.0,
    )
    vehicle = RotorpyCtbrVehicle(
        dynamics=SimpleNamespace(_multirotor=fake_multirotor),
        vehicle="crazyflie",
        rate_limits=RateLimits(roll=1.0, pitch=1.0, yaw=0.5),
    )
    nominal = {
        "cmd_thrust": vehicle.hover_thrust,
        "cmd_w": np.array([0.2, -0.1, 0.05], dtype=np.float64),
    }

    merged = vehicle.apply_gated_ctbr_delta(
        nominal,
        0.0,
        np.array([1.0, 1.0, -1.0, 1.0], dtype=np.float32),
        delta_thrust_fraction=10.0,
        delta_rate_limits=np.array([2.0, 2.0, 2.0]),
    )

    assert merged["cmd_thrust"] == pytest.approx(nominal["cmd_thrust"])
    assert np.allclose(merged["cmd_w"], nominal["cmd_w"])


def test_constant_velocity_closest_approach_detects_approach_vs_recede():
    from neurosim.rl.env_reactive_dodge import constant_velocity_closest_approach

    approaching_distance, approaching_tca = constant_velocity_closest_approach(
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        2.0,
    )
    receding_distance, receding_tca = constant_velocity_closest_approach(
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        2.0,
    )

    assert approaching_distance == pytest.approx(0.0)
    assert approaching_tca == pytest.approx(1.0)
    assert receding_distance == pytest.approx(1.0)
    assert receding_tca == pytest.approx(0.0)


def test_reactive_dodge_experiment_config_has_curriculum_settings():
    cfg_path = (
        Path(__file__).resolve().parents[1]
        / "applications/rl/configs/reactive_dodge_sb3_combined_experiment.yaml"
    )
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    task = cfg["env"]["task"]
    obstacles = cfg["env"]["visual_backend"]["dynamic_obstacles"]

    assert task["name"] == "reactive_dodge"
    assert task["config"]["controller"]["model"] == "rotorpy_se3"
    assert task["config"]["trajectory"]["model"] == "habitat_random_minsnap"
    assert task["config"]["residual_control"]["mode"] == "gated_ctbr_delta"
    assert obstacles["enabled"] is True
    assert obstacles["max_concurrent"] == 1
    assert obstacles["throw_speed_range_mps"][1] <= 3.0
