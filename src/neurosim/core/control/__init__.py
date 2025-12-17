import logging

from .types import ControllerProtocol, ControllerType
from neurosim.core.utils.logging_utils import format_dict

logger = logging.getLogger(__name__)


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

        controller = RotorpySE3Controller(
            vehicle=kwargs["vehicle"], controller_type=model
        )
    else:
        raise ValueError(f"Unsupported controller type: {model}")

    logger.info("═══════════════════════════════════════════════════════")
    logger.info(f"✅ Controller initialized: {controller.__class__.__name__}")
    logger.info(format_dict(kwargs))
    logger.info("═══════════════════════════════════════════════════════")

    return controller


__all__ = ["ControllerType", "ControllerProtocol", "create_controller"]
