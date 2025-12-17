"""
Taken from an older version of RotorPy.

https://github.com/spencerfolk/rotorpy
"""

import numpy as np


class Polynomial(object):
    """ """

    def __init__(self, points, v_avg=1.2, yaw_angles=None):
        """
        Inputs:
            points, (N, 3) array of N waypoint coordinates in 3D
            v_avg, the average speed between segments
            yaw_angles, (N,) array of yaw angles at each waypoint (optional)
        """

        def get_poly(xi, xf, T):
            """
            Return fully constrained polynomial coefficients from xi to xf in
            time interval [0,T]. Low derivatives are all zero at the endpoints.
            """
            A = np.array(
                [
                    [0, 0, 0, 0, 0, 1],
                    [T**5, T**4, T**3, T**2, T, 1],
                    [0, 0, 0, 0, 1, 0],
                    [5 * T**4, 4 * T**3, 3 * T**2, 2 * T, 1, 0],
                    [0, 0, 0, 2, 0, 0],
                    [20 * T**3, 12 * T**2, 6 * T, 2, 0, 0],
                ]
            )
            b = np.array([xi, xf, 0, 0, 0, 0])
            poly = np.linalg.solve(A, b)
            return poly

        self.v_avg = v_avg

        # Remove any sequential duplicate points; always keep the first point.
        seg_dist = np.linalg.norm(np.diff(points, axis=0), axis=1)
        seg_mask = np.append(True, seg_dist > 1e-3)
        points = points[seg_mask, :]

        # If at least two waypoints remain, calculate segment polynomials.
        if points.shape[0] >= 2:
            N = points.shape[0] - 1
            self.T = seg_dist / self.v_avg  # segment duration
            self.x_poly = np.zeros((N, 3, 6))
            for i in range(N):
                for j in range(3):
                    self.x_poly[i, j, :] = get_poly(
                        points[i, j], points[i + 1, j], self.T[i]
                    )
        # Otherwise, hard code constant polynomial at initial waypoint position.
        else:
            N = 1
            self.T = np.zeros((N,))
            self.x_poly = np.zeros((N, 3, 6))
            self.x_poly[0, :, -1] = points[0, :]

        # Calculate global start time of each segment.
        self.t_start = np.concatenate(([0], np.cumsum(self.T[:-1])))

        # Calculate derivative polynomials.
        self.x_dot_poly = np.zeros((N, 3, 5))
        self.x_ddot_poly = np.zeros((N, 3, 4))
        self.x_dddot_poly = np.zeros((N, 3, 3))
        self.x_ddddot_poly = np.zeros((N, 3, 2))
        for i in range(N):
            for j in range(3):
                self.x_dot_poly[i, j, :] = np.polyder(self.x_poly[i, j, :], m=1)
                self.x_ddot_poly[i, j, :] = np.polyder(self.x_poly[i, j, :], m=2)
                self.x_dddot_poly[i, j, :] = np.polyder(self.x_poly[i, j, :], m=3)
                self.x_ddddot_poly[i, j, :] = np.polyder(self.x_poly[i, j, :], m=4)

        # Handle yaw angles if provided.
        if yaw_angles is not None:
            yaw_angles = np.asarray(yaw_angles)
            # Ensure yaw_angles matches the filtered points
            if yaw_angles.shape[0] == points.shape[0]:
                yaw_angles = yaw_angles[seg_mask]

            if yaw_angles.shape[0] >= 2:
                # Unwrap to prevent discontinuities
                yaw_angles = np.unwrap(yaw_angles)
                self.yaw_poly = np.zeros((N, 6))
                for i in range(N):
                    self.yaw_poly[i, :] = get_poly(
                        yaw_angles[i], yaw_angles[i + 1], self.T[i]
                    )

                # Calculate yaw derivatives
                self.yaw_dot_poly = np.zeros((N, 5))
                self.yaw_ddot_poly = np.zeros((N, 4))
                for i in range(N):
                    self.yaw_dot_poly[i, :] = np.polyder(self.yaw_poly[i, :], m=1)
                    self.yaw_ddot_poly[i, :] = np.polyder(self.yaw_poly[i, :], m=2)
            else:
                # Single point: constant yaw
                self.yaw_poly = np.zeros((N, 6))
                self.yaw_poly[0, -1] = yaw_angles[0] if yaw_angles.size > 0 else 0.0
                self.yaw_dot_poly = np.zeros((N, 5))
                self.yaw_ddot_poly = np.zeros((N, 4))
        else:
            # No yaw specified: default to zero
            self.yaw_poly = np.zeros((N, 6))
            self.yaw_dot_poly = np.zeros((N, 5))
            self.yaw_ddot_poly = np.zeros((N, 4))

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x = np.zeros((3,))
        x_dot = np.zeros((3,))
        x_ddot = np.zeros((3,))
        x_dddot = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0
        yaw_ddot = 0

        # Find interval index i and time within interval t.
        t = np.clip(t, self.t_start[0], self.t_start[-1] + self.T[-1])
        for i in range(self.t_start.size):
            if self.t_start[i] + self.T[i] >= t:
                break
        t = t - self.t_start[i]

        # Evaluate polynomial.
        for j in range(3):
            x[j] = np.polyval(self.x_poly[i, j, :], t)
            x_dot[j] = np.polyval(self.x_dot_poly[i, j, :], t)
            x_ddot[j] = np.polyval(self.x_ddot_poly[i, j, :], t)
            x_dddot[j] = np.polyval(self.x_dddot_poly[i, j, :], t)
            x_ddddot[j] = np.polyval(self.x_ddddot_poly[i, j, :], t)

        # Evaluate yaw polynomial
        yaw = np.polyval(self.yaw_poly[i, :], t)
        yaw_dot = np.polyval(self.yaw_dot_poly[i, :], t)
        yaw_ddot = np.polyval(self.yaw_ddot_poly[i, :], t)

        flat_output = {
            "x": x,
            "x_dot": x_dot,
            "x_ddot": x_ddot,
            "x_dddot": x_dddot,
            "x_ddddot": x_ddddot,
            "yaw": yaw,
            "yaw_dot": yaw_dot,
            "yaw_ddot": yaw_ddot,
        }
        return flat_output
