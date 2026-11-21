import numpy as np
from rotorpy.vehicles.multirotor import Multirotor

PX4_MINIMUM_YAW_MARGIN = 0.15


class MultirotorEuler(Multirotor):
    def __init__(self, quad_params, *args, **kwargs):
        super().__init__(quad_params, *args, **kwargs)
        self._px4_allocation = quad_params.get("px4_allocation", False)

    def get_cmd_motor_speeds(self, state, control):
        """RotorPy's motor speeds, allocated as PX4 does for a vehicle that asks for it."""
        speeds = super().get_cmd_motor_speeds(state, control)
        if not self._px4_allocation or self.control_abstraction in (
            "cmd_motor_speeds",
            "cmd_motor_thrusts",
        ):
            return speeds
        thrust_moments = self.f_to_TM @ (self.k_eta * speeds * np.abs(speeds))
        forces = self._allocate_px4(thrust_moments)
        return np.sign(forces) * np.sqrt(np.abs(forces) / self.k_eta)

    def _allocate_px4(self, thrust_moments):
        """Rotor forces with airmode off: thrust, roll and pitch first, then what yaw still fits."""
        # after PX4 ControlAllocationSequentialDesaturation::mixAirmodeDisabled (974446c0e8)
        thrust, roll, pitch, yaw = self.TM_to_f.T
        lo = self.k_eta * self.rotor_speed_min**2
        hi = self.k_eta * self.rotor_speed_max**2
        forces = self.TM_to_f[:, :3] @ thrust_moments[:3]
        forces = _desaturate(forces, thrust, lo, hi, reduce_only=True)
        forces = _desaturate(forces, roll, lo, hi)
        forces = _desaturate(forces, pitch, lo, hi)
        forces = forces + yaw * thrust_moments[3]
        forces = _desaturate(forces, yaw, lo, hi + PX4_MINIMUM_YAW_MARGIN * (hi - lo))
        return _desaturate(forces, thrust, lo, hi, reduce_only=True)

    def step(self, state, control, t_step):
        """
        Integrate dynamics forward from state given constant control for time t_step.
        """
        cmd_rotor_speeds = self.get_cmd_motor_speeds(state, control)

        # The true motor speeds can not fall below min and max speeds.
        cmd_rotor_speeds = np.clip(
            cmd_rotor_speeds, self.rotor_speed_min, self.rotor_speed_max
        )

        s = Multirotor._pack_state(state)

        # Form autonomous ODE for constant inputs and integrate one time step.
        s += (
            self._s_dot_fn(0, s, cmd_rotor_speeds) * t_step
        )  # Simple Euler step, RK45 too slow

        # Unpack the state vector.
        state = Multirotor._unpack_state(s)

        # Re-normalize unit quaternion.
        state["q"] = state["q"] / np.linalg.norm(state["q"])

        # Ground constraints, except while lifting off: RotorPy clamps z to 0 below 1 mm, which
        # would swallow a slow liftoff's sub-millimetre steps forever
        if self._enable_ground and self._on_ground(state) and state["v"][2] <= 0.0:
            state = self._handle_vehicle_on_ground(state)

        # Add noise to the motor speed measurement
        state["rotor_speeds"] += np.random.normal(
            scale=np.abs(self.motor_noise), size=(self.num_rotors,)
        )
        state["rotor_speeds"] = np.clip(
            state["rotor_speeds"], self.rotor_speed_min, self.rotor_speed_max
        )

        return state


def _desaturate(forces, direction, lo, hi, reduce_only=False):
    """PX4's desaturateActuators: move forces along direction into [lo, hi], then half the rest.

    reduce_only never moves along +direction (PX4's increase_only on its negative thrust axis).
    """
    direction = direction / np.abs(direction).max()
    gain = _desaturation_gain(forces, direction, lo, hi)
    if reduce_only and gain > 0.0:
        return forces
    forces = forces + gain * direction
    return forces + 0.5 * _desaturation_gain(forces, direction, lo, hi) * direction


def _desaturation_gain(forces, direction, lo, hi):
    """PX4's computeDesaturationGain, skipping rotors with under 0.2 of the strongest effect."""
    strong = np.abs(direction) >= 0.2
    k = (np.clip(forces, lo, hi) - forces)[strong] / direction[strong]
    return min(k.min(initial=0.0), 0.0) + max(k.max(initial=0.0), 0.0)
