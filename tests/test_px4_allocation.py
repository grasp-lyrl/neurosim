import numpy as np

from neurosim.core.dynamics.rotorpy_wrapper import (
    get_multirotor_model,
    get_vehicle_params,
)
from neurosim.core.dynamics.types import DynamicsType


def px4_quad(px4_allocation: bool = True):
    params = {
        **get_vehicle_params("px4_sihsim_quadx"),
        "px4_allocation": px4_allocation,
    }
    return get_multirotor_model(
        DynamicsType.ROTORPY_MULTIROTOR_EULER, params, control_abstraction="cmd_ctbm"
    )


def achieved(quad, thrust: float, moment: list[float]) -> np.ndarray:
    """Collective thrust and body moments the rotors produce once RotorPy clips their speeds."""
    control = {"cmd_thrust": thrust, "cmd_moment": np.array(moment)}
    speeds = np.clip(
        quad.get_cmd_motor_speeds({}, control),
        quad.rotor_speed_min,
        quad.rotor_speed_max,
    )
    return quad.f_to_TM @ (quad.k_eta * speeds**2)


def test_px4_allocation_matches_rotorpy_when_unsaturated():
    thrust, moment = 9.81, [0.02, -0.01, 0.05]
    np.testing.assert_allclose(
        achieved(px4_quad(), thrust, moment),
        achieved(px4_quad(False), thrust, moment),
        err_msg="unsaturated, the allocation must not change anything",
    )


def test_px4_allocation_gives_up_yaw_not_thrust_or_tilt():
    thrust, moment = 9.81, [0.05, -0.03, 1.0]
    result = achieved(px4_quad(), thrust, moment)
    np.testing.assert_allclose(
        result[:3],
        [thrust, *moment[:2]],
        rtol=1e-6,
        err_msg="thrust and roll/pitch must hold",
    )
    assert 0.0 < result[3] < moment[2], "yaw must shrink to what fits, keeping its sign"


def test_px4_allocation_never_raises_thrust_to_fit_roll():
    thrust, moment = 2.0, [1.0, 0.0, 0.0]
    result = achieved(px4_quad(), thrust, moment)
    assert result[0] <= thrust + 1e-9, (
        "airmode off: thrust is never raised to unsaturate"
    )
    assert 0.0 < result[1] < moment[0], (
        "roll must shrink to what fits, keeping its sign"
    )
