"""Unit tests for the NavMesh trajectory helpers. Pure-Python: no Habitat / GPU."""

import numpy as np
import pytest

from neurosim.core.trajectory.habitat_trajs import (
    YAW_RATE_MARGIN,
    YAW_RATE_MAX,
    HoverMinSnap,
    build_minsnap,
    rate_limit_yaw,
    split_indices,
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


# --------------------------------------------------------------------------- #
# HoverMinSnap
# --------------------------------------------------------------------------- #
def straight_path(n: int = 21, length: float = 20.0) -> np.ndarray:
    """A dense straight path, the shape densify_path hands to MinSnap."""
    t = np.linspace(0, 1, n)[:, None]
    return np.hstack([t * length, np.zeros_like(t), np.ones_like(t)])


def hover_traj(chunks: int = 3, hover_s: float = 2.5, n: int = 21) -> HoverMinSnap:
    path = straight_path(n)
    yaw = np.linspace(0, 0.6, len(path))
    bounds = split_indices(path, chunks, np.random.default_rng(0))
    segments = [
        build_minsnap(path[a : b + 1], yaw[a : b + 1], v_avg=1.0)
        for a, b in zip(bounds[:-1], bounds[1:])
    ]
    return HoverMinSnap(segments, hover_s)


DERIVATIVES = ("x_dot", "x_ddot", "x_dddot")


def test_minsnap_endpoint_is_at_rest():
    """HoverMinSnap holds a finished segment, so MinSnap's endpoint must be a real hover."""
    path = straight_path()
    traj = build_minsnap(path, np.linspace(0, 0.6, len(path)), v_avg=1.0)

    flat = traj.update(traj.t_keyframes[-1])

    for key in DERIVATIVES:
        assert np.allclose(flat[key], 0.0, atol=1e-6), (
            f"MinSnap leaves {key}={flat[key]} at its endpoint, so a held segment drifts"
        )
    assert np.isclose(flat["yaw_dot"], 0.0, atol=1e-6)


def test_hover_phase_is_motionless():
    traj = hover_traj(chunks=3, hover_s=2.5)
    end = float(traj.segments[0].t_keyframes[-1])

    frozen = [traj.update(end + dt) for dt in (0.0, 0.5, 1.5, 2.49)]

    for flat in frozen[1:]:
        assert np.allclose(flat["x"], frozen[0]["x"]), "position moved during a hover"
        assert np.isclose(flat["yaw"], frozen[0]["yaw"]), "yaw moved during a hover"
        for key in DERIVATIVES:
            assert np.allclose(flat[key], 0.0, atol=1e-6), (
                f"{key} non-zero during a hover"
            )


def test_hover_joins_are_continuous():
    traj = hover_traj(chunks=3, hover_s=2.5)

    for start in traj.starts[1:]:
        before, after = traj.update(start - 1e-6), traj.update(start + 1e-6)
        assert np.allclose(before["x"], after["x"], atol=1e-4), (
            "position jumps where a hover hands over to the next flight segment"
        )
        assert np.isclose(before["yaw"], after["yaw"], atol=1e-4), "yaw jumps at a join"


def test_hover_duration_is_flight_plus_holds():
    hover_s, chunks = 2.5, 3
    traj = hover_traj(chunks=chunks, hover_s=hover_s)

    flight = sum(float(s.t_keyframes[-1]) for s in traj.segments)

    assert np.isclose(traj.duration, flight + (chunks - 1) * hover_s)
    assert np.isclose(traj.t_keyframes[-1], traj.duration)


def test_hover_traj_covers_every_segment():
    traj = hover_traj(chunks=3, hover_s=2.5)

    reached = {
        int(np.searchsorted(traj.starts, t, side="right") - 1)
        for t in np.linspace(0, traj.duration, 500)
    }

    assert reached == set(range(len(traj.segments))), (
        "some flight segment is never flown"
    )


def test_split_indices_tile_the_path_and_share_boundaries():
    path = straight_path(n=41)

    bounds = split_indices(path, 4, np.random.default_rng(3))

    assert bounds[0] == 0 and bounds[-1] == len(path) - 1
    assert np.all(np.diff(bounds) >= 1), "a chunk has fewer than two waypoints"
    for a, b in zip(bounds[:-1], bounds[1:]):
        assert len(path[a : b + 1]) >= 2


def test_split_indices_are_deterministic():
    path = straight_path(n=41)

    assert split_indices(path, 3, np.random.default_rng(7)) == split_indices(
        path, 3, np.random.default_rng(7)
    )


def test_split_indices_rejects_a_path_too_short_to_chunk():
    with pytest.raises(AssertionError, match="waypoints spread wider"):
        split_indices(straight_path(n=3), 4, np.random.default_rng(0))
