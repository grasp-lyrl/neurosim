from .types import IMUSensorProtocol, IMUSensorType


def create_imu_sensor(model: IMUSensorType | str, **kwargs) -> IMUSensorProtocol:
    if not isinstance(model, IMUSensorType):
        try:
            model = IMUSensorType(model.lower())
        except ValueError:
            raise ValueError(f"Unknown sensor type: {model}")

    if model == IMUSensorType.ROTORPY_IMU:
        from .rotorpy_wrapper import RotorpyImuSensor

        return RotorpyImuSensor(sampling_rate=kwargs.get("sampling_rate", 100))
    else:
        raise ValueError(f"Unsupported sensor type: {model}")


__all__ = ["SensorType", "SensorProtocol", "create_sensor"]
