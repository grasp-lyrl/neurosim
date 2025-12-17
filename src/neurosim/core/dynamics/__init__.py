import logging

from .types import DynamicsProtocol, DynamicsType
from neurosim.core.utils.logging_utils import format_dict

logger = logging.getLogger(__name__)


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

        dynamics = RotorpyDynamics(
            vehicle=kwargs["vehicle"],
            dynamics_type=dynamics_type,
            initial_state=kwargs.get("initial_state", {}),
        )
    else:
        raise ValueError(f"Unsupported dynamics type: {dynamics_type}")

    logger.info("═══════════════════════════════════════════════════════")
    logger.info(f"✅ Dynamics initialized: {dynamics.__class__.__name__}")
    logger.info(format_dict(kwargs))
    logger.info("═══════════════════════════════════════════════════════")

    return dynamics


__all__ = ["DynamicsType", "DynamicsProtocol", "create_dynamics"]
