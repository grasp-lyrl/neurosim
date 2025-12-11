import numpy as np


class CoordinateTransform:
    def __init__(
        self,
        pos_transform: np.ndarray = np.eye(3),
        quat_transform: np.ndarray = np.eye(4),
    ):
        """
        Initialize the coordinate transformation matrices.
        Args:
            pos_transform: 3x3 matrix to transform position vectors
            quat_transform: 4x4 matrix to transform quaternion vectors
        """

        # Position transformation matrix:
        self.pos_transform = pos_transform
        # Quaternion transformation matrix:
        self.quat_transform = quat_transform

    def transform(self, state: dict) -> tuple[np.ndarray, np.quaternion]:
        """
        Convert from Rotorpy coordinate system to Habitat coordinate system.

        Args:
            state: State dictionary with 'x' (position) and 'q' (quaternion) keys

        Returns:
            Tuple of (position, rotation) in Habitat coordinate system
        """
        position = self.pos_transform @ state["x"]
        rotation = np.quaternion(*(self.quat_transform @ state["q"]))
        return position, rotation
