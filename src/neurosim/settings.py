"""
Neurosim Settings Module.

This module contains all configuration settings for the Neurosim simulator:
- Default simulation settings for Habitat
- Event camera simulator configuration
- Example/preset configurations
"""

from typing import Any, Dict

import magnum as mn
from habitat_sim.bindings import built_with_bullet


# ============================================================================
# Color Constants
# ============================================================================
BLACK = mn.Color4.from_linear_rgb_int(0)


# ============================================================================
# Default Habitat Simulation Settings
# ============================================================================
default_sim_settings: Dict[str, Any] = {
    # -------------------------------------------------------------------------
    # Scene Configuration
    # -------------------------------------------------------------------------
    # Path to .scene_dataset.json file
    "scene_dataset_config_file": "default",
    # Name of an existing scene in the dataset, a scene, stage, or asset filepath,
    # or "NONE" for an empty scene
    "scene": "habitat-sim/data/scene_datasets/habitat-test-scenes/skokloster-castle-rotated.glb",
    # -------------------------------------------------------------------------
    # Camera Sensor Parameters
    # -------------------------------------------------------------------------
    "width": 640,
    "height": 480,
    # Horizontal field of view in degrees
    "hfov": 90,
    # Far clipping plane
    "zfar": 1000.0,
    # Optional background color override for RGB sensors
    "clear_color": BLACK,
    # Vertical offset of the camera from the agent's root position (e.g., height of eyes)
    "sensor_height": 0.05,
    # -------------------------------------------------------------------------
    # Agent Configuration
    # -------------------------------------------------------------------------
    # Default agent index
    "default_agent": 0,
    # Radius of the agent cylinder approximation for navmesh
    "agent_radius": 0.1,
    # Starting position of the agent
    "start_position": [-3, -14, 2.0],
    # -------------------------------------------------------------------------
    # Sensor Selection
    # -------------------------------------------------------------------------
    "color_sensor": True,
    "event_camera": True,
    "semantic_sensor": False,
    "depth_sensor": False,
    "ortho_rgba_sensor": False,
    "ortho_depth_sensor": False,
    "ortho_semantic_sensor": False,
    "fisheye_rgba_sensor": False,
    "fisheye_depth_sensor": False,
    "fisheye_semantic_sensor": False,
    "equirect_rgba_sensor": False,
    "equirect_depth_sensor": False,
    "equirect_semantic_sensor": False,
    # -------------------------------------------------------------------------
    # Event Camera Configuration
    # -------------------------------------------------------------------------
    # Backend to use: "cuda", "torch", "airsim", or "auto"
    "event_camera_backend": "auto",
    # Contrast thresholds
    "event_contrast_threshold_pos": 0.35,
    "event_contrast_threshold_neg": 0.35,
    # -------------------------------------------------------------------------
    # Simulation Settings
    # -------------------------------------------------------------------------
    # Random seed
    "seed": 1,
    # Path to .physics_config.json file
    "physics_config_file": "data/default.physics_config.json",
    # Use bullet physics for dynamics or not
    "enable_physics": built_with_bullet,
    # Ensure or create compatible navmesh for agent parameters
    "default_agent_navmesh": True,
    # If configuring a navmesh, should STATIC MotionType objects be included
    "navmesh_include_static_objects": False,
    # -------------------------------------------------------------------------
    # Rendering Settings
    # -------------------------------------------------------------------------
    # Enable horizon-based ambient occlusion (soft shadows in corners/crevices)
    "enable_hbao": False,
    # Frustum culling (skip rendering objects outside the camera's view)
    "frustum_culling": True,
}
