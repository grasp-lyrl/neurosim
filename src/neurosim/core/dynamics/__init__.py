from .types import DynamicsProtocol, DynamicsType


def create_dynamics(
    model: DynamicsType | str,
    **kwargs,
) -> DynamicsProtocol:
    """
    Factory function to create dynamics model instances based on settings.

    Args:
        dynamics_settings (Dict[str, Any]): Settings dictionary containing the dynamics model type and parameters.
    Returns:
        DynamicsProtocol: An instance of a dynamics model conforming to DynamicsProtocol.
    """
    if not isinstance(model, DynamicsType):
        try:
            dynamics_type = DynamicsType(model.lower())
        except ValueError:
            raise ValueError(f"Unknown dynamics model type: {model}")
    else:
        dynamics_type = model

    if dynamics_type in [
        DynamicsType.ROTORPY_MULTIROTOR,
        DynamicsType.ROTORPY_MULTIROTOR_EULER,
        DynamicsType.ROTORPY_PX4_MULTIROTOR,
        DynamicsType.ROTORPY_ARDUPILOT_MULTIROTOR,
    ]:
        from .rotorpy_wrapper import RotorpyDynamics

        return RotorpyDynamics(
            vehicle=kwargs["vehicle"],
            dynamics_type=dynamics_type,
            initial_state=kwargs.get("initial_state", {}),
        )
    else:
        raise ValueError(f"Unsupported dynamics type: {dynamics_type}")


__all__ = ["DynamicsType", "DynamicsProtocol", "create_dynamics"]
