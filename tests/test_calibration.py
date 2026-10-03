"""Camera intrinsics and T_cam_imu against Habitat's own cameras and rotorpy's IMU."""

import copy
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import quaternion
import yaml
from scipy.spatial.transform import Rotation

from neurosim.core.coord_trans.calibration import (
    GL_FROM_CV,
    CameraCalibration,
    agent_from_body,
    camera_calibration,
    camera_info,
    kalibr_camera,
    range_calibration,
    ros_range,
    write_kalibr_camchain,
    write_opencv_intrinsics,
)
from neurosim.core.coord_trans import CoordinateTransform

SETTINGS = Path("configs/apartment_1-settings.yaml")
SCENE = Path("data/scene_datasets/habitat-test-scenes/apartment_1.glb")
BRIDGE_CONFIG = Path(
    "src/neurosim/comms_ros2/src/neurosim_ros2_bridge/config/apartment_1.yaml"
)
REFIT = [-0.5721, 0.2233, -0.0025, 0.0066, 0.0]
CAMERAS = {
    "pinhole": {
        "position": [0.1, -0.2, 0.3],
        "orientation": [0.3, -0.5, 0.7],
        "width": 32,
        "height": 24,
        "hfov": 70.0,
    },
    "radtan": {
        "position": [-0.2, 0.1, -0.1],
        "orientation": [-0.2, 0.4, 0.1],
        "width": 40,
        "height": 30,
        "hfov": 49.07,
        "principal_point": [20.7, 13.9],
        "distortion": REFIT,
        "render_scale": 8,
    },
}


def world_from_camera(settings, x, q, T_cam_imu):
    """OpenCV camera pose in Habitat's world, for a rotorpy body at (x, q)."""
    world_from_body = np.eye(4)
    world_from_body[:3, :3] = Rotation.from_quat(q).as_matrix()
    world_from_body[:3, 3] = x
    habitat_from_rotorpy = np.eye(4)
    transform = CoordinateTransform(settings["simulator"]["coord_transform"])
    habitat_from_rotorpy[:3, :3] = transform.pos_transform
    habitat_from_rotorpy[1, 3] = settings["visual_backend"]["agent_height"]
    return habitat_from_rotorpy @ world_from_body @ np.linalg.inv(T_cam_imu)


@pytest.fixture(scope="module")
def habitat():
    """Two depth cameras on a tilted body over a navigable point of apartment_1."""
    pytest.importorskip("habitat_sim")
    if not SCENE.exists():
        pytest.skip("apartment_1 scene not available")
    from neurosim.core.visual_backend.habitat_wrapper import HabitatWrapper

    settings = yaml.safe_load(SETTINGS.read_text())
    settings["visual_backend"]["sensors"] = {
        uuid: {**cfg, "type": "depth", "zfar": 100.0}
        for uuid, cfg in copy.deepcopy(CAMERAS).items()
    }
    backend = HabitatWrapper(settings["visual_backend"])
    transform = CoordinateTransform(settings["simulator"]["coord_transform"])
    point = backend._sim.pathfinder.get_random_navigable_point()
    x = transform.pos_transform_inv @ np.asarray(point, float)
    q = Rotation.from_euler("xyz", [0.2, -0.1, 1.3]).as_quat()
    backend.update_agent_state(*transform.transform(x, q))
    yield settings, backend, x, q
    backend.close()


def test_backend_leaves_the_configured_positions_alone(habitat):
    settings = habitat[0]
    for uuid, cfg in CAMERAS.items():
        position = settings["visual_backend"]["sensors"][uuid]["position"]
        assert position == cfg["position"], (
            f"{uuid}: HabitatWrapper edited its settings"
        )


@pytest.mark.parametrize("uuid", list(CAMERAS))
def test_camera_pose_is_habitats_sensor_node(habitat, uuid):
    settings, backend, x, q = habitat
    node = backend._sim.get_agent(0)._sensors[uuid].node
    expected = np.array(node.absolute_transformation()) @ GL_FROM_CV
    pose = world_from_camera(
        settings, x, q, camera_calibration(settings, uuid).T_cam_imu
    )
    np.testing.assert_allclose(
        pose, expected, atol=1e-4, err_msg=f"{uuid}: T_cam_imu misplaces the camera"
    )


def test_camera_matrix_is_habitats_projection(habitat):
    settings, backend = habitat[:2]
    sensor = backend._sim.get_agent(0)._sensors["pinhole"]
    proj = np.array(sensor.render_camera.projection_matrix)
    w, h = CAMERAS["pinhole"]["width"], CAMERAS["pinhole"]["height"]
    assert proj[0, 2] == proj[1, 2] == 0, "Habitat's pinhole frustum is symmetric"
    expected = [
        [w / 2 * proj[0, 0], 0, (w - 1) / 2],
        [0, h / 2 * proj[1, 1], (h - 1) / 2],
    ]
    K = camera_calibration(settings, "pinhole").K
    np.testing.assert_allclose(K[:2], expected, rtol=1e-6, err_msg="K is not Habitat's")


@pytest.mark.parametrize("uuid", list(CAMERAS))
def test_rendered_depth_lies_along_the_calibrated_ray(habitat, uuid):
    import habitat_sim as hsim
    import magnum as mn

    settings, backend, x, q = habitat
    camera = camera_calibration(settings, uuid)
    w, h = camera.resolution
    pixels = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).astype(float)
    xy = cv2.undistortPoints(pixels.reshape(-1, 1, 2), camera.K, camera.dist)[:, 0]
    pose = world_from_camera(settings, x, q, camera.T_cam_imu)
    rays = np.column_stack([xy, np.ones(len(xy))]) @ pose[:3, :3].T
    origin = mn.Vector3(pose[:3, 3])
    hits = [
        backend._sim.cast_ray(hsim.geo.Ray(origin, mn.Vector3(r))).hits for r in rays
    ]
    along_ray = np.array([h[0].ray_distance if h else np.nan for h in hits])
    depth = backend.render_depth(uuid).cpu().numpy().ravel()
    seen = depth > 0
    err = np.abs(along_ray[seen] - depth[seen]) / depth[seen]
    assert np.nanmedian(err) < 1e-3, (
        f"{uuid}: median depth {np.nanmedian(err):.1%} off its ray"
    )
    assert np.mean(err < 1e-2) > 0.95, (
        f"{uuid}: {np.mean(err >= 1e-2):.1%} of pixels off"
    )


def test_turning_the_body_turns_the_cameras_in_place(habitat):
    settings, backend, x, q = habitat
    transform = CoordinateTransform(settings["simulator"]["coord_transform"])
    sensors = backend._sim.get_agent(0)._sensors

    def cameras(q):
        backend.update_agent_state(*transform.transform(x, q))
        return {u: np.array(sensors[u].node.absolute_translation) for u in CAMERAS}

    level = cameras(np.array([0.0, 0.0, 0.0, 1.0]))
    for turn in Rotation.random(8, random_state=1):
        turned = cameras(turn.as_quat())
        for uuid, cfg in CAMERAS.items():
            mount = np.linalg.norm(cfg["position"])
            moved = np.linalg.norm(turned[uuid] - level[uuid])
            assert moved <= 2 * mount * np.sin(turn.magnitude() / 2) + 1e-5, (
                f"{uuid} moved {moved:.3f} m on a {mount:.3f} m mount"
            )
    backend.update_agent_state(*transform.transform(x, q))


def test_imu_reads_in_the_body_frame_at_the_body_origin():
    from neurosim.core.imu_sim import create_imu_sensor

    imu = create_imu_sensor(model="rotorpy", sampling_rate=100)
    q, w = (
        Rotation.from_euler("xyz", [0.3, -0.4, 1.2]).as_quat(),
        np.array([0.5, -1, 2]),
    )
    state = {"x": np.zeros(3), "v": np.zeros(3), "q": q, "w": w}
    reading = imu.measurement(state, {"vdot": np.zeros(3), "wdot": np.zeros(3)})
    gravity = Rotation.from_quat(q).inv().apply([0.0, 0.0, 9.81])
    np.testing.assert_allclose(
        reading["accel"], gravity, err_msg="accel is not in body axes"
    )
    np.testing.assert_allclose(reading["gyro"], w, err_msg="gyro is not in body axes")


@pytest.mark.parametrize("name", ["rotorpy_to_hm3d", "rotorpy_to_replica"])
def test_agent_turns_rigidly_with_the_body(name):
    transform = CoordinateTransform(name)
    for q in Rotation.random(16, random_state=0).as_quat():
        _, q_agent = transform.transform(np.zeros(3), q)
        agent = quaternion.as_rotation_matrix(q_agent)
        body = Rotation.from_quat(q).as_matrix()
        np.testing.assert_allclose(
            agent.T @ transform.pos_transform @ body,
            agent_from_body(transform),
            atol=1e-9,
            err_msg=f"{name} does not carry the agent rigidly with the body",
        )


@pytest.mark.parametrize(
    "orientation, beam",
    [([-1.5708, 0.0, 0.0], [0, 0, -1]), ([1.5708, 0.0, 0.0], [0, 0, 1])],
)
def test_rangefinder_beam_is_the_body_axis_it_is_turned_to(orientation, beam):
    rangefinder = {
        "type": "range",
        "position": [0.0, 0.0, 0.0],
        "orientation": orientation,
        "hfov": 2.0,
        "min_range": 0.05,
        "max_range": 12.0,
    }
    settings = {"visual_backend": {"sensors": {"range": rangefinder}}}
    T_range_imu = np.array(range_calibration(settings, "range")["T_range_imu"])
    np.testing.assert_allclose(
        T_range_imu[2, :3], beam, atol=1e-5, err_msg="the beam is not that body axis"
    )


def test_ros_range_reads_out_of_span_as_rep_117_infinities():
    info = {"field_of_view": 0.035, "min_range": 0.05, "max_range": 12.0}
    assert ros_range(0.0, info) == np.inf, "no hit must read +inf"
    assert ros_range(12.5, info) == np.inf, "past max_range must read +inf"
    assert ros_range(0.01, info) == -np.inf, "short of min_range must read -inf"
    assert ros_range(1.7, info) == 1.7


def test_opencv_intrinsics_read_back_through_filestorage(tmp_path):
    K = np.array([[400.0, 0.0, 319.5], [0.0, 400.0, 239.5], [0.0, 0.0, 1.0]])
    camera = CameraCalibration(K, np.array(REFIT), (640, 480), np.eye(4))
    write_opencv_intrinsics(tmp_path / "camera.xml", camera)
    storage = cv2.FileStorage(str(tmp_path / "camera.xml"), cv2.FILE_STORAGE_READ)
    size = storage.getNode("image_width").real(), storage.getNode("image_height").real()
    assert size == (640, 480)
    np.testing.assert_array_equal(storage.getNode("camera_matrix").mat(), K)
    dist = storage.getNode("distortion_coefficients").mat().ravel()
    np.testing.assert_array_equal(dist, REFIT)


def test_kalibr_camchain_links_consecutive_cameras(tmp_path):
    settings = {"visual_backend": {"sensors": CAMERAS}}
    cameras = {uuid: camera_calibration(settings, uuid) for uuid in CAMERAS}
    topics = {uuid: f"/{uuid}" for uuid in CAMERAS}
    write_kalibr_camchain(tmp_path / "camchain.yaml", cameras, topics)
    chain = yaml.safe_load((tmp_path / "camchain.yaml").read_text())
    cam0, cam1 = (np.array(chain[cam]["T_cam_imu"]) for cam in ("cam0", "cam1"))
    np.testing.assert_allclose(np.array(chain["cam1"]["T_cn_cnm1"]) @ cam0, cam1)
    K = cameras["radtan"].K
    assert chain["cam1"]["intrinsics"] == [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
    assert chain["cam1"]["distortion_coeffs"] == REFIT[:4]


def test_kalibr_camchain_refuses_a_k3(tmp_path):
    camera = CameraCalibration(
        np.eye(3), np.array([0.1, 0, 0, 0, 0.2]), (4, 3), np.eye(4)
    )
    with pytest.raises(ValueError):
        write_kalibr_camchain(
            tmp_path / "camchain.yaml", {"cam": camera}, {"cam": "/cam"}
        )


def test_camera_info_and_kalibr_describe_one_camera():
    settings = {"visual_backend": {"sensors": CAMERAS}}
    for uuid in CAMERAS:
        camera = camera_calibration(settings, uuid)
        info, kalibr = camera_info(camera), kalibr_camera(camera)
        K = np.reshape(info["k"], (3, 3))
        fx, fy, cx, cy = kalibr["intrinsics"]
        np.testing.assert_array_equal(K, [[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        np.testing.assert_array_equal(
            np.reshape(info["p"], (3, 4)), np.hstack([K, [[0], [0], [0]]])
        )
        np.testing.assert_array_equal(np.reshape(info["r"], (3, 3)), np.eye(3))
        assert info["d"] == [*kalibr["distortion_coeffs"], 0.0], (
            f"{uuid}: plumb_bob has k3"
        )
        assert [info["width"], info["height"]] == kalibr["resolution"]


def test_cli_waits_for_the_recorder_and_writes_beside_the_bag(tmp_path):
    bag = tmp_path / "flight"
    cmd = [
        "-m",
        "neurosim.core.coord_trans.calibration",
        "--settings",
        str(SETTINGS),
        "--config",
        str(BRIDGE_CONFIG),
        "--bag",
        str(bag),
    ]
    process = subprocess.Popen([sys.executable, *cmd])
    time.sleep(2.0)
    bag.mkdir()
    assert process.wait(timeout=60) == 0
    files = sorted(path.name for path in bag.iterdir())
    cameras = ["color_camera_1", "depth_camera_1", "event_camera_1"]
    assert files == [
        "camchain-imucam.yaml",
        *(f"{c}.xml" for c in cameras),
        "imu_1.yaml",
    ]
    chain = yaml.safe_load((bag / "camchain-imucam.yaml").read_text())
    assert [chain[cam]["rostopic"] for cam in chain] == [
        "/neurosim/camera/color/image_raw",
        "/neurosim/camera/depth/image_raw",
        "/neurosim/camera/events/event_camera_1",
    ]
    imu = yaml.safe_load((bag / "imu_1.yaml").read_text())
    assert imu == {
        "rostopic": "/neurosim/imu/data",
        "update_rate": 50.0,
        "accelerometer_noise_density": 0.0,
        "accelerometer_random_walk": 0.0,
        "gyroscope_noise_density": 0.0,
        "gyroscope_random_walk": 0.0,
    }, "the IMU file is not the noise-free 50 Hz IMU of apartment_1"
