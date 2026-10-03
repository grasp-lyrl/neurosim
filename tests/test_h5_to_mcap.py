"""An H5 recording made into a bag: the topics, stamps and values the bridge publishes."""

import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from neurosim.core.coord_trans.calibration import (
    camera_calibration,
    camera_info,
    range_calibration,
    range_info,
    ros_range,
)

rosbag2_py = pytest.importorskip("rosbag2_py")

SETTINGS = Path("configs/apartment_1-settings.yaml")
SCENE = Path("data/scene_datasets/habitat-test-scenes/apartment_1.glb")
TYPES = {
    "/neurosim/state": "neurosim_ros2_bridge/msg/State",
    "/neurosim/odom": "nav_msgs/msg/Odometry",
    "/tf": "tf2_msgs/msg/TFMessage",
    "/neurosim/imu/imu_1": "neurosim_ros2_bridge/msg/Imu",
    "/neurosim/imu/data": "sensor_msgs/msg/Imu",
    "/neurosim/camera/color/image_raw": "sensor_msgs/msg/Image",
    "/neurosim/camera/color/camera_info": "sensor_msgs/msg/CameraInfo",
    "/neurosim/camera/depth/image_raw": "sensor_msgs/msg/Image",
    "/neurosim/camera/depth/camera_info": "sensor_msgs/msg/CameraInfo",
    "/neurosim/camera/events/event_camera_1": "neurosim_ros2_bridge/msg/Events",
    "/neurosim/camera/events/camera_info": "sensor_msgs/msg/CameraInfo",
    "/neurosim/range/range_sensor_1": "sensor_msgs/msg/Range",
}
CAMERAS = {
    "color_camera_1": ("/neurosim/camera/color/image_raw", "color"),
    "depth_camera_1": ("/neurosim/camera/depth/image_raw", "depth"),
    "event_camera_1": ("/neurosim/camera/events/event_camera_1", "events"),
}
RANGE = {
    "type": "range",
    "position": [0.0, 0.0, 0.0],
    "orientation": [-1.5708, 0.0, 0.0],
    "hfov": 2.0,
    "min_range": 0.05,
    "max_range": 12.0,
}


@pytest.fixture(scope="module")
def bag(tmp_path_factory):
    """A 0.3 s apartment_1 flight recorded to H5, converted, and every message read back."""
    pytest.importorskip("habitat_sim")
    if not SCENE.exists():
        pytest.skip("apartment_1 scene not available")
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    settings = yaml.safe_load(SETTINGS.read_text())
    settings["simulator"]["sim_time"] = 0.3
    for cfg in settings["visual_backend"]["sensors"].values():
        cfg.update(width=64, height=48)
    settings["visual_backend"]["sensors"]["range_sensor_1"] = RANGE
    settings["simulator"]["sensor_rates"]["range_sensor_1"] = 100
    out = tmp_path_factory.mktemp("convert")
    (out / "settings.yaml").write_text(yaml.safe_dump(settings))
    record = [sys.executable, "test_sim.py", "--settings", str(out / "settings.yaml")]
    subprocess.run([*record, "--log-h5", str(out / "flight.h5")], check=True)
    script = [sys.executable, "scripts/h5_to_mcap.py", str(out / "flight.h5")]
    subprocess.run([*script, "--out", str(out / "bag")], check=True)

    # other test modules import habitat_sim, whose corrade leaves dlopen at RTLD_GLOBAL;
    # typesupports loaded that way bind sensor_msgs/Imu to neurosim_ros2_bridge/Imu's
    # same-named symbols
    flags = sys.getdlopenflags()
    sys.setdlopenflags(flags & ~os.RTLD_GLOBAL)
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(out / "bag"), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    messages = {name: [] for name in types}
    while reader.has_next():
        topic, data, ns = reader.read_next()
        messages[topic].append(
            (ns, deserialize_message(data, get_message(types[topic])))
        )
    sys.setdlopenflags(flags)
    with h5py.File(out / "flight.h5", "r") as f:
        yield f, types, messages, out / "bag"


def stamp_ns(msg) -> int:
    header = msg.transforms[0].header if hasattr(msg, "transforms") else msg.header
    return header.stamp.sec * 10**9 + header.stamp.nanosec


def logged_ns(group: h5py.Group, every: int) -> list[int]:
    steps, times = group["sim_step"][:], group["sim_time"][:]
    return [round(float(t) * 1e9) for t in times[steps % every == 0]]


def test_bag_has_the_recorded_topics_with_the_bridge_types(bag):
    _, types, messages, _ = bag
    assert types == TYPES
    assert all(messages.values()), "a topic is empty"


def test_bag_time_is_every_header_stamp(bag):
    _, _, messages, _ = bag
    for topic, items in messages.items():
        assert [ns for ns, _ in items] == [stamp_ns(m) for _, m in items], topic
        assert all(np.diff([ns for ns, _ in items]) > 0), f"{topic} repeats a stamp"


def test_state_and_odometry_are_the_logged_state_every_control_step(bag):
    f, _, messages, _ = bag
    rows = np.flatnonzero(f["state/sim_step"][:] % 10 == 0)
    x, q, v, w = (f[f"state/{k}"][:][rows] for k in "xqvw")
    for topic in ("/neurosim/state", "/neurosim/odom", "/tf"):
        assert [ns for ns, _ in messages[topic]] == logged_ns(f["state"], 10), topic
    for i, (_, msg) in enumerate(messages["/neurosim/state"]):
        np.testing.assert_array_equal([msg.x.x, msg.x.y, msg.x.z], x[i])
        np.testing.assert_array_equal([msg.q.x, msg.q.y, msg.q.z, msg.q.w], q[i])
        np.testing.assert_array_equal([msg.v.x, msg.v.y, msg.v.z], v[i])
        np.testing.assert_array_equal([msg.w.x, msg.w.y, msg.w.z], w[i])
    for i, (_, msg) in enumerate(messages["/neurosim/odom"]):
        twist = msg.twist.twist.linear
        body_v = Rotation.from_quat(q[i]).inv().apply(v[i])
        np.testing.assert_allclose([twist.x, twist.y, twist.z], body_v, atol=1e-12)
        assert (msg.header.frame_id, msg.child_frame_id) == ("world", "base_link")
    for i, (_, msg) in enumerate(messages["/tf"]):
        t = msg.transforms[0].transform.translation
        np.testing.assert_array_equal([t.x, t.y, t.z], x[i])


def test_imu_messages_are_the_logged_samples(bag):
    f, _, messages, _ = bag
    accel, gyro = f["imu_1/accel"][:], f["imu_1/gyro"][:]
    custom, standard = messages["/neurosim/imu/imu_1"], messages["/neurosim/imu/data"]
    assert [ns for ns, _ in standard] == logged_ns(f["imu_1"], 20)
    for i, ((_, a), (_, b)) in enumerate(zip(custom, standard)):
        np.testing.assert_array_equal([a.accel.x, a.accel.y, a.accel.z], accel[i])
        np.testing.assert_array_equal([a.gyro.x, a.gyro.y, a.gyro.z], gyro[i])
        lin, ang = b.linear_acceleration, b.angular_velocity
        np.testing.assert_array_equal([lin.x, lin.y, lin.z], accel[i])
        np.testing.assert_array_equal([ang.x, ang.y, ang.z], gyro[i])
        assert b.orientation_covariance[0] == -1.0, "orientation must read as unknown"


@pytest.mark.parametrize("uuid", ["color_camera_1", "depth_camera_1"])
def test_images_are_the_logged_frames(bag, uuid):
    f, _, messages, _ = bag
    topic = CAMERAS[uuid][0]
    frames = f[f"{uuid}/data"][:]
    assert [ns for ns, _ in messages[topic]] == logged_ns(f[uuid], 50)
    for frame, (_, msg) in zip(frames, messages[topic]):
        assert (msg.height, msg.width, msg.header.frame_id) == (48, 64, uuid)
        assert bytes(msg.data) == frame.tobytes(), f"{uuid} frame differs"


def test_events_are_the_logged_events_in_viz_packets(bag):
    f, _, messages, _ = bag
    packets = messages["/neurosim/camera/events/event_camera_1"]
    stamps = [ns for ns, _ in packets]
    assert set(stamps) <= set(logged_ns(f["event_camera_1"], 50))
    previous = -1
    for ns, msg in packets:
        t = np.asarray(msg.t, dtype=np.int64) * 1000
        assert previous < t.min() and t.max() <= ns, "an event left its packet"
        previous = ns
    for k in "xytp":
        sent = np.concatenate([np.asarray(getattr(m, k)) for _, m in packets])
        np.testing.assert_array_equal(sent, f[f"event_camera_1/{k}"][: len(sent)])
    t = f["event_camera_1/t"][:]
    assert (t <= stamps[-1] // 1000).sum() == len(sent), "events dropped"


@pytest.mark.parametrize("uuid", list(CAMERAS))
def test_camera_info_accompanies_each_camera_message(bag, uuid):
    f, _, messages, _ = bag
    topic, name = CAMERAS[uuid]
    infos = messages[f"/neurosim/camera/{name}/camera_info"]
    assert [ns for ns, _ in infos] == [ns for ns, _ in messages[topic]]
    settings = yaml.safe_load(f.attrs["settings"])
    expected = camera_info(camera_calibration(settings, uuid))
    for _, msg in infos:
        assert msg.header.frame_id == uuid
        assert (msg.width, msg.height, msg.distortion_model) == (64, 48, "plumb_bob")
        for key in ("d", "k", "r", "p"):
            np.testing.assert_array_equal(getattr(msg, key), expected[key], key)


def test_range_messages_are_the_logged_readings_as_rep_117_ranges(bag):
    f, _, messages, _ = bag
    readings = messages["/neurosim/range/range_sensor_1"]
    assert [ns for ns, _ in readings] == logged_ns(f["range_sensor_1"], 10)
    info = range_info(RANGE)
    fixed = np.float32([info["field_of_view"], info["min_range"], info["max_range"]])
    for distance, (_, msg) in zip(f["range_sensor_1/data"][:], readings):
        assert (msg.header.frame_id, msg.radiation_type) == (
            "range_sensor_1",
            msg.INFRARED,
        )
        np.testing.assert_array_equal(
            [msg.field_of_view, msg.min_range, msg.max_range], fixed
        )
        assert msg.range == np.float32(ros_range(distance, info))


def test_calibration_files_sit_beside_the_bag(bag):
    f, *_, path = bag
    names = {p.name for p in path.iterdir()}
    beside = {"camchain-imucam.yaml", "imu_1.yaml", "range_sensor_1.yaml"}
    assert beside | {f"{c}.xml" for c in CAMERAS} <= names
    rangefinder = yaml.safe_load((path / "range_sensor_1.yaml").read_text())
    settings = yaml.safe_load(f.attrs["settings"])
    assert rangefinder == {
        "rostopic": "/neurosim/range/range_sensor_1",
        **range_calibration(settings, "range_sensor_1"),
    }
