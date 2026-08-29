import numpy as np

from neurosim.core.visual_backend.dynamic_obstacles import solve_spawn_retarget


def _entry():
    return {
        "speed_mps": 4.0,
        "lead_distance_m": 8.0,
        "approach": np.array([1.0, 0.0, 0.0]),
        "aim_offset": np.zeros(3),
        "target_velocity": np.array([0.0, 0.0, 1.0]),
    }


def test_spawn_retarget_intersects_current_constant_velocity_prediction():
    entry = _entry()
    drone_position = np.array([2.0, 3.0, 4.0])
    drone_velocity = np.array([0.0, 0.0, 1.5])

    spawn, projectile_velocity = solve_spawn_retarget(
        entry, drone_position, drone_velocity
    )
    flight_time = entry["lead_distance_m"] / entry["speed_mps"]

    np.testing.assert_allclose(
        spawn + projectile_velocity * flight_time,
        drone_position + drone_velocity * flight_time,
    )


def test_spawn_retarget_follows_preexisting_position_offset():
    entry = _entry()
    nominal_position = np.array([0.0, 0.0, 0.0])
    offset_position = np.array([0.0, 0.45, 0.0])

    nominal_spawn, _ = solve_spawn_retarget(entry, nominal_position, None)
    offset_spawn, _ = solve_spawn_retarget(entry, offset_position, None)

    np.testing.assert_allclose(offset_spawn - nominal_spawn, offset_position)


def test_spawn_retarget_uses_scheduled_velocity_when_history_unavailable():
    entry = _entry()
    spawn, projectile_velocity = solve_spawn_retarget(entry, np.zeros(3), None)
    flight_time = entry["lead_distance_m"] / entry["speed_mps"]

    np.testing.assert_allclose(
        spawn + projectile_velocity * flight_time,
        entry["target_velocity"] * flight_time,
    )
