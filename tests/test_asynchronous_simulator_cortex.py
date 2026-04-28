"""Integration tests for async simulator nodes with real Habitat + dynamics.

Uses the same ``apartment_1.glb`` scene as :mod:`test_rl_env` (see scene-path
parity test). Heavy tests use a session-scoped Cortex discovery daemon and
module-scoped real nodes; they skip when data assets are missing.
"""

import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from cortex.messages.standard import ArrayMessage, DictMessage, MultiArrayMessage
from cortex.utils.loop import run

from neurosim.rl.env import NeurosimRLEnv
from neurosim.sims.asynchronous_simulator import (
    controller_node,
    simulator_node,
    visualizer_node,
)
from neurosim.sims.asynchronous_simulator.controller_node import ControllerNode
from neurosim.sims.asynchronous_simulator.cortex_io import (
    CONTROL_TOPIC,
    STATE_TOPIC,
    message_type_for_sensor,
    sensor_topic,
    sensor_topics_from_settings,
)
from neurosim.sims.asynchronous_simulator.simulator_node import SimulatorNode

try:
    from test_rl_env import _test_env_config
except ImportError:  # pragma: no cover
    from tests.test_rl_env import _test_env_config  # type: ignore[no-redef]

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCENE_GLB = _REPO_ROOT / "data/scene_datasets/habitat-test-scenes/apartment_1.glb"
_APARTMENT_SETTINGS = _REPO_ROOT / "configs" / "apartment_1-settings.yaml"

skip_missing_async_settings = pytest.mark.skipif(
    not _APARTMENT_SETTINGS.is_file(),
    reason=f"Async reference settings not found: {_APARTMENT_SETTINGS}",
)


@pytest.fixture(scope="session")
def cortex_discovery_daemon():
    """Cortex discovery must be up for publishers/subscribers (see deps/cortex examples)."""
    proc = subprocess.Popen(
        ["cortex-discovery", "--log-level", "ERROR"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.5)
    try:
        yield
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


def _load_apartment_settings() -> dict:
    with open(_APARTMENT_SETTINGS) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def real_simulator_node(cortex_discovery_daemon):
    """Real :class:`SimulatorNode` (Habitat + RotorPy dynamics + sensors)."""
    if not _SCENE_GLB.is_file() or not _APARTMENT_SETTINGS.is_file():
        pytest.skip(
            f"Missing scene or async settings: {_SCENE_GLB} / {_APARTMENT_SETTINGS}"
        )
    node = SimulatorNode(settings=_load_apartment_settings())
    try:
        yield node
    finally:
        run(node.close())


@pytest.fixture(scope="module")
def real_controller_node(cortex_discovery_daemon):
    """Real :class:`ControllerNode` (Habitat pathfinder + trajectory + control)."""
    if not _SCENE_GLB.is_file() or not _APARTMENT_SETTINGS.is_file():
        pytest.skip(
            f"Missing scene or async settings: {_SCENE_GLB} / {_APARTMENT_SETTINGS}"
        )
    node = ControllerNode(settings_path=_APARTMENT_SETTINGS)
    try:
        yield node
    finally:
        run(node.close())


@skip_missing_async_settings
def test_rl_built_settings_use_same_scene_as_apartment_async_reference():
    """Parity with :mod:`test_rl_env`: same scene path as full async YAML."""
    rl_settings = NeurosimRLEnv._build_simulator_settings(
        _test_env_config(obs_mode="state", episode_seconds=0.05)
    )
    apartment = _load_apartment_settings()
    assert (
        rl_settings["visual_backend"]["scene"] == apartment["visual_backend"]["scene"]
    )


def test_sensor_topics_from_apartment_settings(real_simulator_node):
    topics = sensor_topics_from_settings(real_simulator_node.settings)
    assert STATE_TOPIC == "state"
    assert CONTROL_TOPIC == "control"
    assert ("events/event_camera_1", "event_camera_1") in topics["events"]
    assert ("color/color_camera_1", "color_camera_1") in topics["color"]
    assert ("depth/depth_camera_1", "depth_camera_1") in topics["depth"]
    assert ("imu/imu_1", "imu_1") in topics["imu"]
    assert sensor_topic("events", "event_camera_1") == "events/event_camera_1"


def test_sensor_topics_cover_all_sync_visual_sensor_types():
    sensors = {
        "event_1": {"type": "event"},
        "color_1": {"type": "color"},
        "semantic_1": {"type": "semantic"},
        "depth_1": {"type": "depth"},
        "navmesh_1": {"type": "navmesh"},
        "flow_1": {"type": "optical_flow"},
        "corner_1": {"type": "corner"},
        "edge_1": {"type": "edge"},
        "gray_1": {"type": "grayscale"},
    }
    settings = {
        "visual_backend": {"sensors": sensors},
        "simulator": {"additional_sensors": {"imu_1": {"type": "imu"}}},
    }

    topics = sensor_topics_from_settings(settings)
    assert ("events/event_1", "event_1") in topics["events"]
    assert ("color/color_1", "color_1") in topics["color"]
    assert ("semantic/semantic_1", "semantic_1") in topics["semantic"]
    assert ("depth/depth_1", "depth_1") in topics["depth"]
    assert ("navmesh/navmesh_1", "navmesh_1") in topics["navmesh"]
    assert ("optical_flow/flow_1", "flow_1") in topics["optical_flow"]
    assert ("corner/corner_1", "corner_1") in topics["corner"]
    assert ("edge/edge_1", "edge_1") in topics["edge"]
    assert ("grayscale/gray_1", "gray_1") in topics["grayscale"]
    assert ("imu/imu_1", "imu_1") in topics["imu"]

    assert message_type_for_sensor("event") is MultiArrayMessage
    assert message_type_for_sensor("imu") is DictMessage
    assert message_type_for_sensor("corner") is DictMessage
    for sensor_type in (
        "color",
        "semantic",
        "depth",
        "navmesh",
        "optical_flow",
        "edge",
        "grayscale",
    ):
        assert message_type_for_sensor(sensor_type) is ArrayMessage


def test_simulator_node_real_step_advances_time_and_finite_state(real_simulator_node):
    node = real_simulator_node
    t0, s0 = node.time, node.simsteps
    run(node.simulate_step())
    assert node.time > t0
    assert node.simsteps == s0 + 1
    for key in ("x", "q", "v", "w"):
        assert np.isfinite(node.dynamics.state[key]).all()


def test_simulator_step_updates_dynamic_obstacles(real_simulator_node, monkeypatch):
    node = real_simulator_node
    calls = []

    def update_dynamic_obstacles(sim_time, dt):
        calls.append((sim_time, dt))

    monkeypatch.setattr(
        node.visual_backend, "update_dynamic_obstacles", update_dynamic_obstacles
    )
    run(node.simulate_step())

    assert calls
    assert calls[-1][0] == node.time
    assert calls[-1][1] == node.config.t_step


def test_simulator_publish_state_uses_real_dynamics(real_simulator_node):
    node = real_simulator_node
    run(node.simulate_step())
    before = node.state_pub.publish_count
    run(node.publish_state())
    assert node.state_pub.publish_count == before + 1


def test_simulator_receive_control_updates_from_dict_message(real_simulator_node):
    node = real_simulator_node
    n = len(node.control["cmd_motor_speeds"])
    run(
        node.receive_control(
            DictMessage(data={"cmd_motor_speeds": [0.1] * n}),
            None,
        )
    )
    assert np.allclose(node.control["cmd_motor_speeds"], np.array([0.1] * n))


def test_simulator_publish_color_after_real_render(real_simulator_node):
    node = real_simulator_node
    color_uuid = "color_camera_1"
    assert color_uuid in node.sensor_publishers
    sensor_cfg = node.sensor_manager.sensors[color_uuid]
    pub = node.sensor_publishers[color_uuid]
    before = pub.publish_count
    for _ in range(800):
        run(node.simulate_step())
        if color_uuid in node.measurements:
            break
    assert color_uuid in node.measurements, "color camera did not sample in time"
    run(node.publish_color(sensor_cfg))
    assert pub.publish_count == before + 1


def test_simulator_publish_events_when_buffer_non_empty(real_simulator_node):
    node = real_simulator_node
    uuid = "event_camera_1"
    pub = node.sensor_publishers[uuid]
    buf = node.event_buffers[uuid]
    before = pub.publish_count
    for _ in range(1500):
        run(node.simulate_step())
        if buf.size > 0:
            break
    if buf.size == 0:
        pytest.skip("No events accumulated in allotted steps (GPU/scene dependent)")
    run(node.publish_events(node.sensor_manager.sensors[uuid]))
    assert pub.publish_count == before + 1


def test_controller_node_compute_control_after_real_state(real_controller_node):
    """One real control tick using trajectory + controller (Habitat from YAML)."""
    node = real_controller_node
    # Use plausible state dict matching async simulator publish format
    dyn = {
        "x": [0.0, 0.0, 1.0],
        "q": [0.0, 0.0, 0.0, 1.0],
        "v": [0.0, 0.0, 0.0],
        "w": [0.0, 0.0, 0.0],
    }
    run(node.receive_state(DictMessage(data=dyn), None))
    node._cpu_clock_start_time = time.perf_counter()  # noqa: SLF001
    before = node.control_pub.publish_count
    run(node.compute_and_publish_control())
    assert node.control_pub.publish_count == before + 1


def test_async_nodes_follow_deep_cortex_import_and_run_conventions():
    for path in [
        Path(simulator_node.__file__),
        Path(controller_node.__file__),
        Path(visualizer_node.__file__),
    ]:
        source = path.read_text()
        assert "from cortex import ArrayMessage" not in source
        assert "from cortex import DictMessage" not in source
        assert "from cortex import Node" not in source
        assert "asyncio.run" not in source
        assert "cortex.run(main())" in source
