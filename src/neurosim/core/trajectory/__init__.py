import logging
import numpy as np

from .types import TrajectoryProtocol, TrajectoryType
from neurosim.core.utils import format_dict

logger = logging.getLogger(__name__)


def create_trajectory(model: TrajectoryType | str, **kwargs) -> TrajectoryProtocol:
    """
    Factory function to create trajectory instances.

    Inputs:
        model, the trajectory model to create
        **kwargs, keyword arguments specific to the trajectory model

    Returns:
        An instance of a trajectory class implementing TrajectoryProtocol
    """

    if not isinstance(model, TrajectoryType):
        try:
            model = TrajectoryType(model.lower())
        except ValueError:
            raise ValueError(f"Unknown trajectory type: {model}")

    if model == TrajectoryType.CONSTANT_SPEED:
        from rotorpy.trajectories.speed_traj import ConstantSpeed

        traj = ConstantSpeed(
            init_pos=kwargs.get("init_pos"),
            dist=kwargs.get("dist", 5),
            speed=kwargs.get("speed", 1),
            axis=kwargs.get("axis", 0),
        )
    elif model == TrajectoryType.MINSNAP:
        from rotorpy.trajectories.minsnap import MinSnap

        points = np.array(kwargs.get("points", []))
        yaw_angles = kwargs.get("yaw_angles", None)
        if yaw_angles is not None:
            yaw_angles = np.array(yaw_angles)

        traj = MinSnap(
            points=points,
            yaw_angles=yaw_angles,
            yaw_rate_max=kwargs.get("yaw_rate_max", 2 * np.pi),
            poly_degree=kwargs.get("poly_degree", 7),
            yaw_poly_degree=kwargs.get("yaw_poly_degree", 7),
            v_max=kwargs.get("v_max", 3.0),
            v_avg=kwargs.get("v_avg", 1.0),
            v_start=kwargs.get("v_start", [0.0, 0.0, 0.0]),
            v_end=kwargs.get("v_end", [0.0, 0.0, 0.0]),
        )
    elif model == TrajectoryType.POLYNOMIAL:
        from .polynomial_traj import Polynomial

        points = np.array(kwargs.get("points", []))
        yaw_angles = kwargs.get("yaw_angles", None)
        if yaw_angles is not None:
            yaw_angles = np.array(yaw_angles)

        traj = Polynomial(
            points=points,
            yaw_angles=yaw_angles,
            v_avg=kwargs.get("v_avg", 1.2),
        )
    elif model == TrajectoryType.HABITAT_RANDOM_MINSNAP:
        from .habitat_trajs import generate_interesting_traj

        traj = generate_interesting_traj(
            pathfinder=kwargs.get("pathfinder"),
            seed=kwargs.get("seed", 324),
            target_length=kwargs.get("target_length", 20.0),
            min_waypoint_distance=kwargs.get("min_waypoint_distance", 2.0),
            max_waypoints=kwargs.get("max_waypoints", 100),
            v_avg=kwargs.get("v_avg", 1.0),
            start=kwargs.get("start", None),
            max_tries_per_waypoint=kwargs.get("max_tries_per_waypoint", 100),
            max_path_attempts=kwargs.get("max_path_attempts", 6),
            coord_transform=kwargs.get("coord_transform", None),
            min_altitude_m=kwargs.get("min_altitude_m", 0.0),
            max_altitude_m=kwargs.get("max_altitude_m", None),
            ceiling_margin_m=kwargs.get("ceiling_margin_m", 0.3),
            lateral_margin_m=kwargs.get("lateral_margin_m", 0.0),
            bounds_margin_m=kwargs.get("bounds_margin_m", 0.0),
            collision_sim=kwargs.get("collision_sim", None),
            trajectory_to_habitat=kwargs.get("trajectory_to_habitat", None),
            static_clearance_m=kwargs.get("static_clearance_m", 0.0),
            collision_sample_spacing_m=kwargs.get(
                "collision_sample_spacing_m", 0.05
            ),
        )
    elif model == TrajectoryType.FREESPACE_MINSNAP:
        from .habitat_trajs import generate_freespace_traj

        traj = generate_freespace_traj(
            seed=kwargs.get("seed", 324),
            box_lo=kwargs.get("box_lo"),
            box_hi=kwargs.get("box_hi"),
            target_length=kwargs.get("target_length", 20.0),
            min_waypoint_distance=kwargs.get("min_waypoint_distance", 2.0),
            max_waypoints=kwargs.get("max_waypoints", 100),
            v_avg=kwargs.get("v_avg", 1.0),
            start=kwargs.get("start", None),
            coord_transform=kwargs.get("coord_transform", None),
            max_path_attempts=kwargs.get("max_path_attempts", 6),
            collision_sim=kwargs.get("collision_sim", None),
            trajectory_to_habitat=kwargs.get("trajectory_to_habitat", None),
            static_clearance_m=kwargs.get("static_clearance_m", 0.0),
            collision_sample_spacing_m=kwargs.get(
                "collision_sample_spacing_m", 0.05
            ),
        )
    else:
        raise ValueError(f"Unsupported trajectory type: {model}")

    logger.info("═══════════════════════════════════════════════════════")
    logger.info(f"✅ Trajectory initialized: {traj.__class__.__name__}")
    logger.info(format_dict(kwargs))
    logger.info("═══════════════════════════════════════════════════════")

    return traj


__all__ = ["TrajectoryType", "TrajectoryProtocol", "create_trajectory"]
