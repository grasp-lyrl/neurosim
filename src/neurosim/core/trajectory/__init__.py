from .types import TrajectoryProtocol, TrajectoryType
import numpy as np


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

        return ConstantSpeed(
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

        return MinSnap(
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

        return Polynomial(
            points=np.array(kwargs.get("points", [])),
            v_avg=kwargs.get("v_avg", 1.2),
        )
    else:
        raise ValueError(f"Unsupported trajectory type: {model}")


__all__ = ["TrajectoryType", "TrajectoryProtocol", "create_trajectory"]
