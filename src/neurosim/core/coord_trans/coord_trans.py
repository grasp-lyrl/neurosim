import numpy as np
from scipy.spatial.transform import Rotation


class CoordinateTransform:
    def __init__(
        self,
        pos_transform: np.ndarray | str = np.eye(3),
        quat_transform: np.quaternion = np.quaternion(1, 0, 0, 0),
    ):
        """
        Initialize the coordinate transformation matrices.
        Args:
            pos_transform: 3x3 matrix to transform position vectors, or string like "rotorpy_to_hm3d"
            quat_transform: quaternion to transform incoming quaternions
        """
        if isinstance(pos_transform, str):
            if pos_transform == "rotorpy_to_hm3d":
                self.pos_transform = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
                self.quat_transform = np.array(
                    [[0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0]]
                )
            elif pos_transform == "rotorpy_to_replica":
                self.pos_transform = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
                self.quat_transform = np.array(
                    [
                        [0.0, 0.7071067811865475, -0.7071067811865475, 0.0],
                        [0.0, -0.7071067811865475, -0.7071067811865475, 0.0],
                        [0.7071067811865475, 0.0, 0.0, 0.7071067811865475],
                        [-0.7071067811865475, 0.0, 0.0, 0.7071067811865475],
                    ]
                )
            else:
                raise ValueError(f"Unknown coordinate transform: {pos_transform}")
        else:
            self.pos_transform = pos_transform
            self.quat_transform = quat_transform

        # Precompute inverse transforms
        self.pos_transform_inv = np.linalg.inv(self.pos_transform)
        self.quat_transform_inv = np.linalg.inv(self.quat_transform)

    def transform(
        self, position: np.ndarray, quaternion: np.ndarray
    ) -> tuple[np.ndarray, np.quaternion]:
        """
        Convert from one coordinate system to another.

        Args:
            position: 3D position vector
            quaternion: Quaternion representing orientation

        Returns:
            Transformed position and quaternion
        """
        position = self.pos_transform @ position
        quaternion = np.quaternion(*(self.quat_transform @ quaternion))
        return position, quaternion

    def transform_batch(self, positions: np.ndarray) -> np.ndarray:
        """
        Batch transform positions.

        Args:
            positions: Array of shape (N, 3) containing position vectors

        Returns:
            Transformed positions array of shape (N, 3)
        """
        return (self.pos_transform @ positions.T).T

    def inverse_transform_batch(self, positions: np.ndarray) -> np.ndarray:
        """
        Batch inverse transform positions.

        Args:
            positions: Array of shape (N, 3) containing position vectors

        Returns:
            Inverse transformed positions array of shape (N, 3)
        """
        return (self.pos_transform_inv @ positions.T).T

    @staticmethod
    def euler_to_quat_and_body_rates(
        roll: float,
        pitch: float,
        yaw: float,
        roll_dot: float = 0.0,
        pitch_dot: float = 0.0,
        yaw_dot: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert roll, pitch, yaw and their rates to quaternion [x, y, z, w] and body rates."""
        # Convert Euler angles (ZYX convention) to quaternion using scipy
        r = Rotation.from_euler("ZYX", [yaw, pitch, roll])
        q = r.as_quat()  # Returns [x, y, z, w]

        # Body rates from Euler angle rates using ZYX convention
        # w = [roll_dot, pitch_dot, yaw_dot] in body frame (for small angles approximation)
        w = np.array([roll_dot, pitch_dot, yaw_dot], dtype=float)

        return q, w
