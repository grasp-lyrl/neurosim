"""Unit tests for the NavMesh trajectory helpers. Pure-Python: no Habitat / GPU."""

import numpy as np
import pytest

from neurosim.core.trajectory.habitat_trajs import (
    YAW_RATE_MARGIN,
    YAW_RATE_MAX,
    rate_limit_yaw,
)


def realised_rates(yaw: np.ndarray, segment_times: np.ndarray) -> np.ndarray:
    return np.abs(np.diff(yaw)) / segment_times


def test_rate_limit_yaw_respects_budget():
    # A half-turn demanded over 0.2 s is 15.7 rad/s, well past YAW_RATE_MAX.
    yaw = np.array([0.0, np.pi, np.pi + 0.1, -2 * np.pi])
    segment_times = np.array([0.2, 1.5, 0.3])

    limited = rate_limit_yaw(yaw, segment_times)

    budget = YAW_RATE_MARGIN * YAW_RATE_MAX
    assert np.all(realised_rates(limited, segment_times) <= budget + 1e-9), (
        "rate_limit_yaw left a step above the yaw rate budget"
    )


def test_rate_limit_yaw_preserves_feasible_turns():
    yaw = np.array([0.0, 0.5, 0.9, 1.0])
    segment_times = np.array([1.0, 1.0, 1.0])

    limited = rate_limit_yaw(yaw, segment_times)

    assert np.allclose(limited, yaw), "a feasible yaw profile was modified"


def test_rate_limit_yaw_keeps_turn_direction():
    yaw = np.array([0.0, -np.pi, -2 * np.pi])
    segment_times = np.array([0.1, 0.1])

    limited = rate_limit_yaw(yaw, segment_times)

    assert np.all(np.diff(limited) < 0), "clamping reversed the direction of the turn"


def test_rate_limit_yaw_anchors_first_waypoint():
    yaw = np.array([0.7, 5.0, -3.0])
    segment_times = np.array([0.1, 0.1])

    limited = rate_limit_yaw(yaw, segment_times)

    assert limited[0] == pytest.approx(yaw[0])


def test_rate_limit_yaw_single_waypoint():
    limited = rate_limit_yaw(np.array([1.23]), np.array([]))

    assert limited == pytest.approx(np.array([1.23]))
