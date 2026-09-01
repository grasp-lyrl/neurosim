import numpy as np
from rotorpy.vehicles.multirotor import Multirotor
from scipy.spatial.transform import Rotation


class YawAwareVelocityMixin:
    """Add an explicit heading reference to RotorPy's ``cmd_vel`` mode.

    Upstream RotorPy fixes the desired horizontal body axis to world +X for
    every velocity command.  That silently drives yaw to zero regardless of
    the trajectory reference.  Neurosim accepts an optional ``cmd_yaw`` and
    uses it only to choose the otherwise-free heading about the desired
    thrust axis; translational velocity control is unchanged.
    """

    def get_cmd_motor_speeds(self, state, control):
        if self.control_abstraction != "cmd_vel" or "cmd_yaw" not in control:
            return super().get_cmd_motor_speeds(state, control)

        velocity_error = np.asarray(state["v"]) - np.asarray(control["cmd_v"])
        desired_acceleration = -self.k_v * velocity_error
        desired_force = self.mass * (
            desired_acceleration + np.array([0.0, 0.0, self.g])
        )

        rotation = Rotation.from_quat(state["q"]).as_matrix()
        body_z = rotation[:, 2]
        cmd_thrust = float(np.dot(desired_force, body_z))

        force_norm = float(np.linalg.norm(desired_force))
        if force_norm <= 1e-12:
            # This is unreachable for the normal gravity-compensated command,
            # but fail safely instead of constructing a NaN attitude.
            desired_body_z = np.array([0.0, 0.0, 1.0])
        else:
            desired_body_z = desired_force / force_norm

        yaw = float(control["cmd_yaw"])
        desired_heading = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        desired_body_y = np.cross(desired_body_z, desired_heading)
        body_y_norm = float(np.linalg.norm(desired_body_y))
        if body_y_norm <= 1e-12:
            # Only possible for a horizontal thrust vector exactly parallel
            # to the requested heading. Choose the orthogonal horizontal axis
            # deterministically so the controller remains finite.
            fallback = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
            desired_body_y = np.cross(desired_body_z, fallback)
            body_y_norm = float(np.linalg.norm(desired_body_y))
        desired_body_y /= max(body_y_norm, 1e-12)
        desired_body_x = np.cross(desired_body_y, desired_body_z)
        desired_rotation = np.stack(
            [desired_body_x, desired_body_y, desired_body_z], axis=1
        )

        attitude_error_matrix = 0.5 * (
            desired_rotation.T @ rotation - rotation.T @ desired_rotation
        )
        attitude_error = np.array(
            [
                -attitude_error_matrix[1, 2],
                attitude_error_matrix[0, 2],
                -attitude_error_matrix[0, 1],
            ]
        )
        angular_velocity = np.asarray(state["w"])
        cmd_moment = self.inertia @ (
            -self.kp_att * attitude_error - self.kd_att * angular_velocity
        ) + np.cross(angular_velocity, self.inertia @ angular_velocity)

        thrust_moment = np.concatenate(([cmd_thrust], cmd_moment))
        cmd_motor_forces = self.TM_to_f @ thrust_moment
        squared_speeds = cmd_motor_forces / self.k_eta
        return np.sign(squared_speeds) * np.sqrt(np.abs(squared_speeds))


class MultirotorEuler(YawAwareVelocityMixin, Multirotor):
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
