import logging

from .types import IMUSensorProtocol, IMUSensorType
from neurosim.core.utils import format_dict

logger = logging.getLogger(__name__)


def create_imu_sensor(model: IMUSensorType | str, **kwargs) -> IMUSensorProtocol:
    if not isinstance(model, IMUSensorType):
        try:
            model = IMUSensorType(model.lower())
        except ValueError:
            raise ValueError(f"Unknown sensor type: {model}")

    if model == IMUSensorType.ROTORPY:
        from .rotorpy_wrapper import RotorpyImuSensor

        sensor = RotorpyImuSensor(sampling_rate=kwargs["sampling_rate"])
    else:
        raise ValueError(f"Unsupported sensor type: {model}")

    logger.info("═══════════════════════════════════════════════════════")
    logger.info(f"✅ IMU sensor initialized: {sensor.__class__.__name__} @")
    logger.info(format_dict(kwargs))
    logger.info("═══════════════════════════════════════════════════════")

    return sensor


__all__ = ["SensorType", "SensorProtocol", "create_sensor"]
