"""Camera and IMU calibration of a neurosim settings file: OpenCV XML, Kalibr YAML, CameraInfo, H5.

    python -m neurosim.core.coord_trans.calibration --settings configs/apartment_1-settings.yaml \
        --config src/neurosim/comms_ros2/src/neurosim_ros2_bridge/config/apartment_1.yaml \
        --bag outputs/bags/flight_2026-09-29-16-13-04
"""

import argparse
import time
from pathlib import Path
from typing import NamedTuple

import cv2
import h5py
import numpy as np
import quaternion
import yaml
from scipy.spatial.transform import Rotation

from neurosim.core.coord_trans import CoordinateTransform

COORD_TRANSFORM = "rotorpy_to_hm3d"  # SimulationConfig's default
GL_FROM_CV = np.diag([1.0, -1.0, -1.0, 1.0])
CAMERA_PAYLOADS = ("color_image", "depth_image", "events")


class CameraCalibration(NamedTuple):
    K: np.ndarray  # [3, 3] px, pixel centres at integer coordinates
    dist: np.ndarray  # [N] OpenCV k1, k2, p1, p2[, k3]
    resolution: tuple[int, int]  # width, height
    T_cam_imu: np.ndarray  # [4, 4] IMU coordinates to OpenCV camera coordinates


def camera_matrix(cfg: dict) -> np.ndarray:
    """Pixel intrinsics of a camera sensor config."""
    w, h = cfg["width"], cfg["height"]
    f = w / 2 / np.tan(np.radians(cfg["hfov"]) / 2)
    cx, cy = cfg.get("principal_point", ((w - 1) / 2, (h - 1) / 2))
    return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])


def distortion_coefficients(cfg: dict) -> np.ndarray:
    """OpenCV radial-tangential coefficients of a camera sensor config; zero for a pinhole."""
    scale = cfg.get("distortion_scale", 1.0)
    return np.asarray(cfg.get("distortion", np.zeros(4)), float) * scale


def sensor_pose(cfg: dict) -> np.ndarray:
    """A camera's Habitat sensor node in its agent node, in Habitat's y-up, -z-forward axes."""
    pose = np.eye(4)
    # esp/sensor/Sensor.cpp rotates about the parent's x, then y, then z, and its scene
    # nodes keep the translation apart
    pose[:3, :3] = Rotation.from_euler("xyz", cfg["orientation"]).as_matrix()
    pose[:3, 3] = cfg["position"]
    return pose


def agent_from_body(transform: CoordinateTransform) -> np.ndarray:
    """Rotation from rotorpy body axes to the axes of the Habitat agent the body drives."""
    _, q = transform.transform(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))
    return quaternion.as_rotation_matrix(q).T @ transform.pos_transform


def camera_from_imu(settings: dict, uuid: str) -> np.ndarray:
    """T_cam_imu of a camera sensor."""
    vb = settings["visual_backend"]
    name = settings.get("simulator", {}).get("coord_transform", COORD_TRANSFORM)
    agent_from_imu = np.eye(4)
    # RotorpyImuSensor mounts rotorpy's Imu at the body origin, unrotated. HabitatWrapper
    # flies the agent agent_height straight above it, which changes no IMU reading, so
    # the IMU sits at the agent origin.
    agent_from_imu[:3, :3] = agent_from_body(CoordinateTransform(name))
    agent_from_camera = sensor_pose(vb["sensors"][uuid]) @ GL_FROM_CV
    return np.linalg.inv(agent_from_camera) @ agent_from_imu


def habitat_from_world(settings: dict) -> np.ndarray:
    """Rotorpy's world in the Habitat scene, where HabitatWrapper lifts the drone."""
    name = settings.get("simulator", {}).get("coord_transform", COORD_TRANSFORM)
    pose = np.eye(4)
    pose[:3, :3] = CoordinateTransform(name).pos_transform
    pose[1, 3] = settings["visual_backend"]["agent_height"]
    return pose


def camera_calibration(settings: dict, uuid: str) -> CameraCalibration:
    """Intrinsics and T_cam_imu of one camera sensor of a settings file."""
    cfg = settings["visual_backend"]["sensors"][uuid]
    return CameraCalibration(
        K=camera_matrix(cfg),
        dist=distortion_coefficients(cfg),
        resolution=(cfg["width"], cfg["height"]),
        T_cam_imu=camera_from_imu(settings, uuid),
    )


def camera_info(camera: CameraCalibration) -> dict:
    """sensor_msgs/CameraInfo fields of a camera: plumb_bob, unrectified."""
    return {
        "width": camera.resolution[0],
        "height": camera.resolution[1],
        "distortion_model": "plumb_bob",
        "d": np.pad(camera.dist, (0, 5 - len(camera.dist))).tolist(),
        "k": camera.K.ravel().tolist(),
        "r": np.eye(3).ravel().tolist(),
        "p": np.hstack([camera.K, np.zeros((3, 1))]).ravel().tolist(),
    }


def kalibr_imu(settings: dict, uuid: str) -> dict:
    """An IMU in Kalibr's imu.yaml fields; RotorpyImuSensor reads without noise or bias."""
    return {
        "update_rate": float(settings["simulator"]["sensor_rates"][uuid]),
        "accelerometer_noise_density": 0.0,
        "accelerometer_random_walk": 0.0,
        "gyroscope_noise_density": 0.0,
        "gyroscope_random_walk": 0.0,
    }


def bridged_topics(config: dict, payloads: tuple[str, ...]) -> dict[str, str]:
    """ROS topic of each sensor a bridge config publishes with one of these payloads."""
    return {
        entry["cortex_topic"].rpartition("/")[2]: entry["ros2_topic"]
        for entry in config["cortex_to_ros2"]
        if entry["payload"] in payloads
    }


def write_opencv_intrinsics(path: Path, camera: CameraCalibration) -> None:
    """Write intrinsics the way OpenCV's camera calibration sample saves them."""
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    storage.write("image_width", camera.resolution[0])
    storage.write("image_height", camera.resolution[1])
    storage.write("camera_matrix", camera.K)
    storage.write("distortion_coefficients", camera.dist.reshape(-1, 1))
    storage.release()


def kalibr_camera(camera: CameraCalibration) -> dict:
    """A camera in Kalibr's camchain fields: pinhole-radtan intrinsics and T_cam_imu."""
    if camera.dist[4:].any():
        raise ValueError(f"Kalibr's radtan has no k3, got {camera.dist}")
    return {
        "camera_model": "pinhole",
        "intrinsics": camera.K[[0, 1, 0, 1], [0, 1, 2, 2]].tolist(),
        "distortion_model": "radtan",
        "distortion_coeffs": camera.dist[:4].tolist(),
        "resolution": list(camera.resolution),
        "T_cam_imu": camera.T_cam_imu.tolist(),
    }


def write_kalibr_camchain(
    path: Path, cameras: dict[str, CameraCalibration], topics: dict[str, str]
) -> None:
    """Write cameras as Kalibr's camchain-imucam.yaml: pinhole-radtan and T_cam_imu."""
    uuids = list(cameras)
    chain = {}
    for n, uuid in enumerate(uuids):
        camera = cameras[uuid]
        chain[f"cam{n}"] = {
            **kalibr_camera(camera),
            "rostopic": topics[uuid],
            "timeshift_cam_imu": 0.0,
        }
        if n:
            previous = cameras[uuids[n - 1]].T_cam_imu
            chain[f"cam{n}"]["T_cn_cnm1"] = (
                camera.T_cam_imu @ np.linalg.inv(previous)
            ).tolist()
    path.write_text(yaml.safe_dump(chain, sort_keys=False, default_flow_style=None))


def write_calibration(folder: Path, settings: dict, config: dict) -> list[str]:
    """Each bridged camera's OpenCV XML, their Kalibr camchain, each bridged IMU's Kalibr yaml."""
    topics = bridged_topics(config, CAMERA_PAYLOADS)
    cameras = {uuid: camera_calibration(settings, uuid) for uuid in topics}
    for uuid, camera in cameras.items():
        write_opencv_intrinsics(folder / f"{uuid}.xml", camera)
    write_kalibr_camchain(folder / "camchain-imucam.yaml", cameras, topics)
    imus = bridged_topics(config, ("sensor_imu",))
    for uuid, topic in imus.items():
        imu = {"rostopic": topic, **kalibr_imu(settings, uuid)}
        (folder / f"{uuid}.yaml").write_text(yaml.safe_dump(imu, sort_keys=False))
    return [*cameras, *imus]


def write_h5_calibration(file: h5py.File, settings: dict) -> None:
    """Kalibr fields of every camera and IMU under /<uuid>/calib, T_habitat_world in /state."""
    for uuid, cfg in settings["visual_backend"]["sensors"].items():
        if "hfov" in cfg:
            calib = file.require_group(f"{uuid}/calib")
            for name, value in kalibr_camera(
                camera_calibration(settings, uuid)
            ).items():
                calib[name] = value
    for uuid, cfg in settings["simulator"].get("additional_sensors", {}).items():
        if cfg["type"] == "imu":
            calib = file.require_group(f"{uuid}/calib")
            for name, value in kalibr_imu(settings, uuid).items():
                calib[name] = value
    file.require_group("state/calib")["T_habitat_world"] = habitat_from_world(settings)


def wait_for_bag(bag: Path, timeout_s: float) -> None:
    """Block until the recorder has created the bag directory."""
    deadline = time.monotonic() + timeout_s
    while not bag.is_dir():
        if time.monotonic() > deadline:
            raise TimeoutError(f"no bag at {bag} after {timeout_s:.0f} s")
        time.sleep(0.1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--settings", type=Path, required=True, help="simulator settings of the flight"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="bridge config: which cameras and IMUs reach ROS, on which topics",
    )
    parser.add_argument(
        "--bag",
        type=Path,
        required=True,
        help="rosbag2 directory, created by the recorder",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="seconds to wait for the bag directory",
    )
    args = parser.parse_args()

    settings = yaml.safe_load(args.settings.read_text())
    config = yaml.safe_load(args.config.read_text())
    wait_for_bag(args.bag, args.timeout)
    sensors = write_calibration(args.bag, settings, config)
    print(f"wrote the calibration of {', '.join(sensors)} to {args.bag}")


if __name__ == "__main__":
    main()
