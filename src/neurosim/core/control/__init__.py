from .types import ControllerProtocol, ControllerType


def create_controller(
    model: ControllerType | str,
    **kwargs,
) -> ControllerProtocol:
    """
    Factory function to create a controller based on the provided settings.

    Args:
        model (str): The model name of the controller to create.
    Returns:
        ControllerProtocol: An instance of a controller conforming to ControllerProtocol.
    """
    if not isinstance(model, ControllerType):
        try:
            model = ControllerType(model.lower())
        except ValueError:
            raise ValueError(f"Unsupported controller model: {model}")

    if model == ControllerType.ROTORPY_SE3:
        from .rotorpy_wrapper import RotorpySE3Controller

        return RotorpySE3Controller(
            vehicle=kwargs["vehicle"],
            controller_type=model,
        )
    else:
        raise ValueError(f"Unsupported controller type: {model}")


__all__ = ["ControllerType", "ControllerProtocol", "create_controller"]
