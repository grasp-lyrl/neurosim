"""
Base Visual Backend Protocol for Neurosim.

This module defines the protocol (interface) that all visual backends
must implement to work with the Neurosim simulator.
"""

from typing import Any, Protocol

import torch
import numpy as np


class VisualBackendProtocol(Protocol):
    """Protocol defining the interface for visual rendering backends.

    All visual backends (Habitat, CARLA, etc.) must implement this interface
    to be compatible with the Neurosim simulator.
    """

    def update_agent_state(self, position: np.ndarray, quaternion: np.ndarray) -> None:
        """Update the agent's pose in the simulation.

        Args:
            position: 3D position [x, y, z] in world coordinates.
            quaternion: Rotation quaternion [w, x, y, z] or numpy quaternion.
        """
        ...

    def render_color(self, uuid: str) -> torch.Tensor | np.ndarray:
        """Render RGB image from a color sensor.

        Args:
            uuid: Unique identifier for the sensor.

        Returns:
            RGB image tensor of shape (H, W, 3).
        """
        ...

    def render_depth(self, uuid: str) -> torch.Tensor | np.ndarray:
        """Render depth image from a depth sensor.

        Args:
            uuid: Unique identifier for the sensor.

        Returns:
            Depth image tensor of shape (H, W).
        """
        ...

    def render_events(
        self, uuid: str, time: int, to_numpy: bool = False
    ) -> tuple[Any, ...] | None:
        """Render events from an event camera sensor.

        Args:
            uuid: Unique identifier for the sensor.
            time: Current timestamp in microseconds.
            to_numpy: Whether to convert events to numpy arrays.

        Returns:
            Tuple of (x, y, t, p) event arrays, or None if no events.
        """
        ...

    def close(self) -> None:
        """Clean up and close the backend."""
        ...
