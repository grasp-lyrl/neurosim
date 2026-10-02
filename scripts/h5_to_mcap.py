"""Write a neurosim H5 recording as the ROS 2 bag that the bridge and recorder would make of it.

    python scripts/h5_to_mcap.py outputs/flight.h5 --out outputs/bags/flight

Topics, types and frames come from the bridge and recorder configs; header stamps and bag times
are the simulation time. Run with ROS 2 and neurosim_ros2_bridge sourced.
"""

import argparse
import array
import heapq
from bisect import bisect_right
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, NamedTuple

import h5py
import numpy as np
import rosbag2_py
import yaml
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point, Quaternion, TransformStamped, Vector3
from nav_msgs.msg import Odometry
from neurosim_ros2_bridge import msg as bridge_msg
from rclpy.serialization import serialize_message
from rosgraph_msgs.msg import Clock
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, Imu
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage

from neurosim.core.coord_trans.calibration import (
    camera_calibration,
    camera_info,
    write_calibration,
)
from neurosim.core.utils import SimulationConfig

BRIDGE_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "src/neurosim/comms_ros2/src/neurosim_ros2_bridge/config"
)
ENCODINGS = {"color_image": "rgb8", "depth_image": "32FC1"}

Messages = Iterator[tuple[int, str, Any]]  # (stamp ns, topic, message), in time order
Schedule = list[tuple[int, Any]]  # (stamp ns, row), or (stamp ns, slice) for events


class Entry(NamedTuple):
    """One cortex_to_ros2 entry of the bridge config, as config.hpp's Entry."""

    payload: str
    uuid: str  # the stream's H5 group: the cortex topic's last part
    ros2_topic: str
    frame_id: str
    child_frame_id: str  # odometry only
    width: int  # events_image only: the sensor
    height: int


class Recording(NamedTuple):
    h5: h5py.File
    settings: dict  # the simulator settings the recording ran with
    schedules: dict[str, Schedule]  # by stream uuid


class Payload(NamedTuple):
    ros_type: str
    messages: Callable[[Entry, Recording], Messages]


def parse_entry(entry: dict) -> Entry:
    return Entry(
        payload=entry["payload"],
        uuid=entry["cortex_topic"].rpartition("/")[2],
        ros2_topic=entry["ros2_topic"],
        frame_id=entry.get("frame_id", ""),
        child_frame_id=entry.get("child_frame_id", ""),
        width=entry.get("width", 0),
        height=entry.get("height", 0),
    )


def stamp(seconds: float) -> int:
    """Nanoseconds of a simulation time, rounded as the bridge rounds its stamps."""
    return int(round(seconds * 1e9))


def ros_time(ns: int) -> Time:
    return Time(sec=ns // 10**9, nanosec=ns % 10**9)


def header(ns: int, frame_id: str) -> Header:
    return Header(stamp=ros_time(ns), frame_id=frame_id)


def xyz(msg_type: type, values: np.ndarray):
    return msg_type(x=float(values[0]), y=float(values[1]), z=float(values[2]))


def xyzw(values: np.ndarray) -> Quaternion:
    x, y, z, w = (float(v) for v in values)
    return Quaternion(x=x, y=y, z=z, w=w)


def schedule(h5: h5py.File, sim_cfg: SimulationConfig, uuid: str) -> Schedule:
    """When the simulator publishes a stream: its control or viz steps."""
    group = h5[uuid]
    if uuid == "state":
        every = sim_cfg.control_steps
    else:
        every = sim_cfg.sensor_manager.sensors[uuid].viz_steps
    times, steps = group["sim_time"][:], group["sim_step"][:]
    rows = [(stamp(times[i]), i) for i in np.flatnonzero(steps % every == 0)]
    if "t" not in group:
        return rows
    # a packet holds the events rendered since the last one; empty ones are not published
    t, packets, start = group["t"], [], 0
    for ns, _ in rows:
        end = bisect_right(t, ns // 1000, lo=start)
        if end > start:
            packets.append((ns, slice(start, end)))
        start = end
    return packets


def state_messages(entry: Entry, rec: Recording) -> Messages:
    state = rec.h5[entry.uuid]
    x, q, v, w, times, steps = (
        state[k][:] for k in ("x", "q", "v", "w", "sim_time", "sim_step")
    )
    for ns, i in rec.schedules[entry.uuid]:
        msg = bridge_msg.State(
            header=header(ns, entry.frame_id),
            timestamp=float(times[i]),
            simsteps=int(steps[i]),
            x=xyz(Vector3, x[i]),
            q=xyzw(q[i]),
            v=xyz(Vector3, v[i]),
            w=xyz(Vector3, w[i]),
        )
        yield ns, entry.ros2_topic, msg


def odometry_messages(entry: Entry, rec: Recording) -> Messages:
    """Odometry with the twist in the body frame, and its TF, as decoders.cpp makes them."""
    state = rec.h5[entry.uuid]
    x, q, v, w = (state[k][:] for k in "xqvw")
    for ns, i in rec.schedules[entry.uuid]:
        odom = Odometry(
            header=header(ns, entry.frame_id), child_frame_id=entry.child_frame_id
        )
        odom.pose.pose.position = xyz(Point, x[i])
        odom.pose.pose.orientation = xyzw(q[i])
        body_v = Rotation.from_quat(q[i]).inv().apply(v[i])
        odom.twist.twist.linear = xyz(Vector3, body_v)
        odom.twist.twist.angular = xyz(Vector3, w[i])
        tf = TransformStamped(header=odom.header, child_frame_id=entry.child_frame_id)
        tf.transform.translation = xyz(Vector3, x[i])
        tf.transform.rotation = odom.pose.pose.orientation
        yield ns, entry.ros2_topic, odom
        yield ns, "/tf", TFMessage(transforms=[tf])


def clock_messages(entry: Entry, rec: Recording) -> Messages:
    for ns, _ in rec.schedules[entry.uuid]:
        yield ns, entry.ros2_topic, Clock(clock=ros_time(ns))


def imu_messages(entry: Entry, rec: Recording) -> Messages:
    imu = rec.h5[entry.uuid]
    accel, gyro, times, steps = (
        imu[k][:] for k in ("accel", "gyro", "sim_time", "sim_step")
    )
    for ns, i in rec.schedules[entry.uuid]:
        msg = bridge_msg.Imu(
            header=header(ns, entry.frame_id),
            uuid=entry.uuid,
            timestamp=float(times[i]),
            simsteps=int(steps[i]),
            accel=xyz(Vector3, accel[i]),
            gyro=xyz(Vector3, gyro[i]),
        )
        yield ns, entry.ros2_topic, msg


def sensor_imu_messages(entry: Entry, rec: Recording) -> Messages:
    """IMU samples as sensor_msgs/Imu, orientation marked unknown."""
    imu = rec.h5[entry.uuid]
    accel, gyro = imu["accel"][:], imu["gyro"][:]
    for ns, i in rec.schedules[entry.uuid]:
        msg = Imu(
            header=header(ns, entry.frame_id),
            linear_acceleration=xyz(Vector3, accel[i]),
            angular_velocity=xyz(Vector3, gyro[i]),
        )
        msg.orientation_covariance[0] = -1.0
        yield ns, entry.ros2_topic, msg


def image_messages(entry: Entry, rec: Recording) -> Messages:
    frames = rec.h5[entry.uuid]["data"]
    for ns, i in rec.schedules[entry.uuid]:
        frame = frames[i]
        msg = Image(
            header=header(ns, entry.frame_id),
            height=frame.shape[0],
            width=frame.shape[1],
            encoding=ENCODINGS[entry.payload],
            step=frame.nbytes // frame.shape[0],
            data=array.array("B", frame.tobytes()),
        )
        yield ns, entry.ros2_topic, msg


def events_messages(entry: Entry, rec: Recording) -> Messages:
    events = rec.h5[entry.uuid]
    for ns, rows in rec.schedules[entry.uuid]:
        msg = bridge_msg.Events(
            header=header(ns, entry.frame_id),
            x=array.array("H", events["x"][rows].tobytes()),
            y=array.array("H", events["y"][rows].tobytes()),
            t=array.array("Q", events["t"][rows].tobytes()),
            p=array.array("B", events["p"][rows].tobytes()),
        )
        yield ns, entry.ros2_topic, msg


def events_image_messages(entry: Entry, rec: Recording) -> Messages:
    """Each events packet drawn on black, ON blue and OFF red, as decoders.cpp draws it."""
    events = rec.h5[entry.uuid]
    for ns, rows in rec.schedules[entry.uuid]:
        image = np.zeros((entry.height, entry.width, 3), np.uint8)
        image[events["y"][rows], events["x"][rows], 2 * events["p"][rows]] = 255
        msg = Image(
            header=header(ns, entry.frame_id),
            height=entry.height,
            width=entry.width,
            encoding="rgb8",
            step=entry.width * 3,
            data=array.array("B", image.tobytes()),
        )
        yield ns, entry.ros2_topic, msg


def camera_info_messages(entry: Entry, rec: Recording) -> Messages:
    """The camera's CameraInfo with each of its images or events packets."""
    info = camera_info(camera_calibration(rec.settings, entry.uuid))
    for ns, _ in rec.schedules[entry.uuid]:
        msg = CameraInfo(header=header(ns, entry.frame_id), **info)
        yield ns, entry.ros2_topic, msg


PAYLOADS = {
    "state": Payload("neurosim_ros2_bridge/msg/State", state_messages),
    "odometry": Payload("nav_msgs/msg/Odometry", odometry_messages),
    "clock": Payload("rosgraph_msgs/msg/Clock", clock_messages),
    "imu": Payload("neurosim_ros2_bridge/msg/Imu", imu_messages),
    "sensor_imu": Payload("sensor_msgs/msg/Imu", sensor_imu_messages),
    "color_image": Payload("sensor_msgs/msg/Image", image_messages),
    "depth_image": Payload("sensor_msgs/msg/Image", image_messages),
    "events": Payload("neurosim_ros2_bridge/msg/Events", events_messages),
    "events_image": Payload("sensor_msgs/msg/Image", events_image_messages),
    "camera_info": Payload("sensor_msgs/msg/CameraInfo", camera_info_messages),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("h5", type=Path, help="neurosim recording")
    parser.add_argument(
        "--out",
        type=Path,
        help="bag directory to create; default: the H5 path sans .h5",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=BRIDGE_CONFIG / "apartment_1.yaml",
        help="bridge config: what each ROS topic carries",
    )
    parser.add_argument(
        "--record-config",
        type=Path,
        default=BRIDGE_CONFIG / "recorder.yaml",
        help="recorder config: which topics go in the bag",
    )
    args = parser.parse_args()
    out = args.out or args.h5.with_suffix("")

    bridge = yaml.safe_load(args.config.read_text())
    recorder = yaml.safe_load(args.record_config.read_text())
    recorded = set(recorder["recorder"]["ros__parameters"]["topics"])
    entries = [
        e
        for e in map(parse_entry, bridge["cortex_to_ros2"])
        if e.ros2_topic in recorded
    ]
    topics = {e.ros2_topic: PAYLOADS[e.payload].ros_type for e in entries}
    if "/tf" in recorded and any(e.payload == "odometry" for e in entries):
        topics["/tf"] = "tf2_msgs/msg/TFMessage"

    with h5py.File(args.h5, "r") as h5:
        settings = yaml.safe_load(h5.attrs["settings"])
        sim_cfg = SimulationConfig(
            **settings["simulator"],
            visual_sensors=settings["visual_backend"]["sensors"],
        )
        uuids = {e.uuid for e in entries}
        schedules = {uuid: schedule(h5, sim_cfg, uuid) for uuid in uuids}
        rec = Recording(h5, settings, schedules)
        writer = rosbag2_py.SequentialWriter()
        writer.open(
            rosbag2_py.StorageOptions(uri=str(out), storage_id="mcap"),
            rosbag2_py.ConverterOptions("cdr", "cdr"),
        )
        for name, ros_type in topics.items():
            writer.create_topic(rosbag2_py.TopicMetadata(name, ros_type, "cdr"))
        streams = [PAYLOADS[e.payload].messages(e, rec) for e in entries]
        written = 0
        for ns, topic, msg in heapq.merge(*streams, key=lambda m: m[0]):
            if topic in topics:
                writer.write(topic, serialize_message(msg), ns)
                written += 1
        writer.close()
    write_calibration(out, settings, bridge)
    print(
        f"wrote {written} messages on {len(topics)} topics and the calibration to {out}"
    )


if __name__ == "__main__":
    main()
