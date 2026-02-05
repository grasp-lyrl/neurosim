"""
Visual Backend Factory for Neurosim.

This module provides a factory function to create visual rendering backends
based on configuration. Supports multiple backends:

- Habitat: For indoor scenes with glb/gltf files (default)
- CARLA: For outdoor driving/flying scenes with Unreal Engine rendering
"""

import logging
from typing import Any

from neurosim.core.visual_backend.base import VisualBackendProtocol

logger = logging.getLogger(__name__)


def create_visual_backend(settings: dict[str, Any]) -> VisualBackendProtocol:
    """Factory function to create a visual backend based on settings.

    Args:
        settings: Configuration dictionary for the visual backend.
                  Must include 'backend_type' key to specify which backend to use.
                  If 'backend_type' is not specified, defaults to 'habitat'.

    Returns:
        A visual backend instance implementing VisualBackendProtocol.

    """
    backend_type = settings.get("backend_type", "habitat").lower()

    if backend_type == "habitat":
        from .habitat_wrapper import HabitatWrapper

        logger.info("Creating Habitat visual backend")
        return HabitatWrapper(settings)

    elif backend_type == "carla":
        from .carla_wrapper import CarlaWrapper

        logger.info("Creating CARLA visual backend")
        return CarlaWrapper(settings)

    else:
        raise ValueError(f"Unknown visual backend type: '{backend_type}'. ")
