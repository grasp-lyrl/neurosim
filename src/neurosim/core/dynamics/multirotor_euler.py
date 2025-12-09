import numpy as np
from rotorpy.vehicles.multirotor import Multirotor


class MultirotorEuler(Multirotor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        # Apply ground constraints (unified across vehicles)
        if self._enable_ground and self._on_ground(state):
            state = self._handle_vehicle_on_ground(state)

        # Add noise to the motor speed measurement
        state["rotor_speeds"] += np.random.normal(
            scale=np.abs(self.motor_noise), size=(self.num_rotors,)
        )
        state["rotor_speeds"] = np.clip(
            state["rotor_speeds"], self.rotor_speed_min, self.rotor_speed_max
        )

        return state
