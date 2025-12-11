import numpy as np


class CoordinateTransform:
    """Handles coordinate system transformations between dynamics and visual backend."""

    @staticmethod
    def rotorpy_to_habitat(state: dict) -> tuple[np.ndarray, np.quaternion]:
        """
        Convert from Rotorpy coordinate system to Habitat coordinate system.

        Args:
            state: State dictionary with 'x' (position) and 'q' (quaternion) keys

        Returns:
            Tuple of (position, rotation) in Habitat coordinate system
        """
        # Position: [x, y, z] -> [x, z, -y]
        position = np.array([state["x"][0], state["x"][2], -state["x"][1]])

        # Quaternion: [qx, qy, qz, qw] -> [qw, qx, qz, -qy]
        rotation = np.quaternion(
            state["q"][3],  # w
            state["q"][0],  # x
            state["q"][2],  # z
            -state["q"][1],  # -y
        )

        return position, rotation
