"""The synchronous simulator as a ROS 2 node, driven and read over ROS 2 topics.

Runs neurosim's SynchronousSimulator in this process and talks ROS 2 directly through
rclpy (no Cortex, no bridge). It needs ROS 2 sourced in the neurosim conda env, as in the
neurosim:ros image:

    python -m neurosim.sims.synchronous_simulator.ros2_node --ros-args \
        -p settings:=configs/neurofly-apartment_1-settings.yaml

Publishes, every header stamped with the simulation time the sample was taken at:
    /clock                              rosgraph_msgs/Clock, at control_rate
    <prefix>/odom                       nav_msgs/Odometry world -> base_link, at control_rate
                                        (+ TF); twist in base_link (REP 105), or in world
                                        with odom_twist_frame:=world (mocap drivers do that).
                                        The world frame is the dynamics' translated to the
                                        settings' ros2.world_origin (default the origin)
    <prefix>/imu/data                   sensor_msgs/Imu, per IMU sensor (frame base_link)
    <prefix>/camera/<type>/image_raw    sensor_msgs/Image: color rgb8, depth 32FC1 metres,
                                        events rgb8 (ON blue, OFF red, per viz packet)
    <prefix>/camera/<type>/camera_info  sensor_msgs/CameraInfo with every image
    /tf_static                          base_link -> each camera's optical frame
<type> is the sensor type (color, depth, event); with two of a type the topic uses the
sensor's uuid instead.

Subscribes:
    <prefix>/control                    std_msgs/Float64MultiArray, the dynamics'
                                        control_abstraction's fields flattened in order:
                                        cmd_motor_speeds -> [w1..wn]; cmd_ctatt -> [thrust
                                        (N), qx, qy, qz, qw]; ... The last command holds
                                        until the next; before the first, the dynamics hover.

The simulation steps at world_rate, paced to real_time_factor (1.0 real time, 0 as fast as
possible). Each sensor is sampled at its sensor_rate and published at its viz_rate.
"""

import array
import logging
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import torch
import yaml
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, Imu
from std_msgs.msg import Float64MultiArray, Header
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from neurosim.core.coord_trans.calibration import camera_calibration, camera_info
from neurosim.sims.synchronous_simulator import SynchronousSimulator

logger = logging.getLogger(__name__)

IMAGE_SENSORS = ("color", "depth", "event")


def to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def stamp(t: float) -> Time:
    sec = int(t)
    return Time(sec=sec, nanosec=int(round((t - sec) * 1e9)))


def control_layout(default: dict) -> list[tuple[str, int, bool]]:
    """(key, size, scalar) of each field of a control dict, in order."""
    return [(k, int(np.size(v)), np.isscalar(v)) for k, v in default.items()]


def unflatten_control(data, layout) -> dict:
    """Split a flat command into the control dict the dynamics take."""
    expected = sum(n for _, n, _ in layout)
    if len(data) != expected:
        raise ValueError(
            f"control has {len(data)} values, the dynamics take {expected}: "
            + ", ".join(f"{k}[{n}]" for k, n, _ in layout)
        )
    control, i = {}, 0
    for key, n, scalar in layout:
        chunk = np.asarray(data[i : i + n], dtype=np.float64)
        control[key] = float(chunk[0]) if scalar else chunk
        i += n
    return control


def events_image(events, width: int, height: int) -> np.ndarray:
    """rgb8 image of an events packet: ON blue, OFF red, the last event per pixel."""
    img = torch.zeros((height, width, 3), dtype=torch.uint8, device=events[0].device)
    if events[0].numel():
        x, y, p = events[0].long(), events[1].long(), events[3].bool()
        img[y[p], x[p], 2] = 255
        img[y[~p], x[~p], 0] = 255
    return img.cpu().numpy()


class Ros2SimulatorNode(Node):
    def __init__(self):
        super().__init__("neurosim_simulator")
        p = self.declare_parameter
        settings = p("settings", "").value
        prefix = p("topic_prefix", "/neurosim").value.rstrip("/")
        self.world_frame = p("world_frame", "world").value
        self.body_frame = p("body_frame", "base_link").value
        self.twist_in_world = p("odom_twist_frame", "child").value == "world"
        self.publish_tf = p("publish_tf", True).value
        self.rtf = float(p("real_time_factor", 1.0).value)
        if not settings or not Path(settings).exists():
            raise FileNotFoundError(f"settings:={settings!r} does not exist")
        with open(settings) as f:
            self.settings = yaml.safe_load(f)

        self.world_origin = np.asarray(
            self.settings.get("ros2", {}).get("world_origin", [0.0, 0.0, 0.0]), float
        )
        self.sim = SynchronousSimulator(self.settings, visualizer_disabled=True)
        self.sensors = self.sim.config.sensor_manager
        self.control = self.sim.dynamics._default_control()
        self.layout = control_layout(self.control)
        self.n_controls = 0

        sensor_qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.clock_pub = self.create_publisher(Clock, "/clock", 10)
        self.odom_pub = self.create_publisher(Odometry, f"{prefix}/odom", 10)
        self.tf = TransformBroadcaster(self) if self.publish_tf else None

        # one publisher set per sensor; topic by type unless a type has several sensors
        by_type: dict[str, list[str]] = {}
        for uuid, s in self.sensors.sensors.items():
            by_type.setdefault(s.sensor_type, []).append(uuid)
        self.pubs, self.infos, self.event_packets = {}, {}, {}
        statics = []
        for uuid, s in self.sensors.sensors.items():
            kind = s.sensor_type
            name = kind if len(by_type[kind]) == 1 else uuid
            if kind == "imu":
                self.pubs[uuid] = self.create_publisher(
                    Imu, f"{prefix}/imu/{'data' if name == kind else uuid}", 50
                )
            elif kind in IMAGE_SENSORS:
                cal = camera_calibration(self.settings, uuid)
                # depth is the mapper's input: reliable, so a slow consumer sees every frame
                qos = 5 if kind == "depth" else sensor_qos
                self.pubs[uuid] = self.create_publisher(
                    Image, f"{prefix}/camera/{name}/image_raw", qos
                )
                info_pub = self.create_publisher(
                    CameraInfo, f"{prefix}/camera/{name}/camera_info", 5
                )
                self.infos[uuid] = (info_pub, camera_info(cal))
                statics.append(self._camera_tf(uuid, cal.T_cam_imu))
                if kind == "event":
                    self.event_packets[uuid] = []
            else:
                self.get_logger().warn(f"sensor {uuid} ({kind}) is not published over ROS 2")
        if statics:
            self.static_tf = StaticTransformBroadcaster(self)
            self.static_tf.sendTransform(statics)

        self.create_subscription(
            Float64MultiArray, f"{prefix}/control", self._on_control, 1
        )
        self.latest: dict[str, tuple[float, object]] = {}
        self.get_logger().info(
            f"neurosim ready: {self.sim.config.world_rate:g} Hz world, "
            f"{self.sim.config.control_rate:g} Hz state, control takes "
            + ", ".join(f"{k}[{n}]" for k, n, _ in self.layout)
            + f", real_time_factor {self.rtf:g}"
        )

    def _camera_tf(self, uuid: str, T_cam_imu: np.ndarray) -> TransformStamped:
        T = np.linalg.inv(T_cam_imu)  # camera optical frame in the body (IMU) frame
        t = TransformStamped()
        t.header.frame_id = self.body_frame
        t.child_frame_id = uuid
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (
            float(v) for v in T[:3, 3]
        )
        q = Rotation.from_matrix(T[:3, :3]).as_quat()
        t.transform.rotation.x, t.transform.rotation.y = float(q[0]), float(q[1])
        t.transform.rotation.z, t.transform.rotation.w = float(q[2]), float(q[3])
        return t

    def _on_control(self, msg: Float64MultiArray) -> None:
        try:
            self.control = unflatten_control(msg.data, self.layout)
            self.n_controls += 1
        except ValueError as e:
            self.get_logger().error(str(e), throttle_duration_sec=2.0)

    # ---- simulation loop -----------------------------------------------------------

    def run(self) -> None:
        cfg = self.sim.config
        wall0 = time.perf_counter()
        last_report, steps_at_report = wall0, 0
        while rclpy.ok():
            self.sim.step(self.control)
            step, t = self.sim.simsteps, self.sim.time
            for uuid, m in self.sim._render_sensors().items():
                if uuid in self.event_packets:
                    if m is not None:
                        self.event_packets[uuid].append(m)
                else:
                    self.latest[uuid] = (t, m)
            if step % cfg.control_steps == 0:
                self._publish_state(t)
            for uuid in self.pubs:
                if self.sensors.should_visualize(uuid, step):
                    self._publish_sensor(uuid, t)

            now = time.perf_counter()
            if self.rtf > 0:
                ahead = wall0 + t / self.rtf - now
                if ahead > 1e-3:
                    time.sleep(ahead)
            if now - last_report > 5.0:
                rate = (step - steps_at_report) * cfg.t_step / (now - last_report)
                self.get_logger().info(
                    f"sim time {t:.1f} s, {rate:.2f}x real time, {self.n_controls} commands"
                )
                last_report, steps_at_report = now, step

    def _publish_state(self, t: float) -> None:
        s = self.sim.dynamics.state
        header = Header(stamp=stamp(t), frame_id=self.world_frame)
        self.clock_pub.publish(Clock(clock=header.stamp))
        odom = Odometry(header=header, child_frame_id=self.body_frame)
        pose = odom.pose.pose
        pose.position.x, pose.position.y, pose.position.z = (
            float(v) for v in np.asarray(s["x"], float) - self.world_origin
        )
        q = s["q"]  # [x, y, z, w]
        pose.orientation.x, pose.orientation.y = float(q[0]), float(q[1])
        pose.orientation.z, pose.orientation.w = float(q[2]), float(q[3])
        v = np.asarray(s["v"], float)
        if not self.twist_in_world:
            v = Rotation.from_quat(q).inv().apply(v)
        tw = odom.twist.twist
        tw.linear.x, tw.linear.y, tw.linear.z = (float(c) for c in v)
        tw.angular.x, tw.angular.y, tw.angular.z = (float(c) for c in s["w"])
        self.odom_pub.publish(odom)
        if self.tf is not None:
            tf = TransformStamped(header=header, child_frame_id=self.body_frame)
            tf.transform.translation.x = pose.position.x
            tf.transform.translation.y = pose.position.y
            tf.transform.translation.z = pose.position.z
            tf.transform.rotation = pose.orientation
            self.tf.sendTransform(tf)

    def _publish_sensor(self, uuid: str, t: float) -> None:
        kind = self.sensors.sensors[uuid].sensor_type
        if kind == "event":
            packets = self.event_packets[uuid]
            self.event_packets[uuid] = []
            info = self.infos[uuid][1]
            events = (
                tuple(torch.cat([pk[i] for pk in packets]) for i in range(4))
                if packets
                else tuple(torch.empty(0, dtype=torch.int64) for _ in range(4))
            )
            self._publish_image(uuid, t, events_image(events, info["width"], info["height"]), "rgb8")
            return
        sample = self.latest.pop(uuid, None)
        if sample is None:
            return
        t_sample, m = sample
        if kind == "imu":
            msg = Imu(header=Header(stamp=stamp(t_sample), frame_id=self.body_frame))
            a, g = to_numpy(m["accel"]).ravel(), to_numpy(m["gyro"]).ravel()
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = (
                float(c) for c in a
            )
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = (
                float(c) for c in g
            )
            msg.orientation_covariance[0] = -1.0  # no orientation estimate
            self.pubs[uuid].publish(msg)
        elif kind == "color":
            self._publish_image(uuid, t_sample, to_numpy(m)[..., :3], "rgb8")
        elif kind == "depth":
            self._publish_image(uuid, t_sample, to_numpy(m).astype(np.float32), "32FC1")

    def _publish_image(self, uuid: str, t: float, img: np.ndarray, encoding: str) -> None:
        img = np.ascontiguousarray(img)
        header = Header(stamp=stamp(t), frame_id=uuid)
        msg = Image(
            header=header,
            height=img.shape[0],
            width=img.shape[1],
            encoding=encoding,
            is_bigendian=0,
            step=img.strides[0],
        )
        # array('B') is rclpy's fast path; bytes or a list are checked element by element
        msg.data = array.array("B", img.tobytes())
        self.pubs[uuid].publish(msg)
        info_pub, info = self.infos[uuid]
        ci = CameraInfo(header=header, width=info["width"], height=info["height"])
        ci.distortion_model = info["distortion_model"]
        ci.d, ci.k, ci.r, ci.p = info["d"], info["k"], info["r"], info["p"]
        info_pub.publish(ci)

    def close(self) -> None:
        self.sim.close()


def _spin(executor) -> None:
    try:
        executor.spin()
    except Exception:
        # Ctrl-C: rclpy's signal handler shuts the context down under the executor's wait
        if rclpy.ok():
            raise


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    rclpy.init()
    node = Ros2SimulatorNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    spinner = threading.Thread(target=_spin, args=(executor,), daemon=True)
    spinner.start()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
