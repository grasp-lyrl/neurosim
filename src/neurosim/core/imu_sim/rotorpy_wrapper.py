from typing import Dict, Any
import numpy as np
from rotorpy.sensors.imu import Imu
from .types import IMUSensorProtocol


class RotorpyImuSensor(IMUSensorProtocol):
    def __init__(self, sampling_rate=100):
        self._imu = Imu(p_BS=np.zeros(3), R_BS=np.eye(3), sampling_rate=sampling_rate)

    def measurement(
        self, state: Dict[str, Any], statedot: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self._imu.measurement(state, statedot, with_noise=False)
