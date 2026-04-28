"""Shared Cortex topic and message helpers for asynchronous simulator nodes."""

import numpy as np

from cortex.discovery.client import DiscoveryClient
from cortex.discovery.daemon import DEFAULT_DISCOVERY_ADDRESS
from cortex.discovery.protocol import (
    DiscoveryCommand,
    DiscoveryRequest,
    DiscoveryStatus,
)
from cortex.messages.standard import ArrayMessage, DictMessage, MultiArrayMessage

STATE_TOPIC = "state"
CONTROL_TOPIC = "control"

SENSOR_TOPIC_TYPES = {
    "event": "events",
    "imu": "imu",
    "color": "color",
    "semantic": "semantic",
    "depth": "depth",
    "navmesh": "navmesh",
    "optical_flow": "optical_flow",
    "corner": "corner",
    "edge": "edge",
    "grayscale": "grayscale",
}

SUBSCRIBE_DEFAULTS = {
    "queue_size": 1000,
    "wait_for_topic": True,
    "topic_timeout": 60.0,
}


def message_type_for_sensor(sensor_type: str):
    """Return the Cortex message type used for a supported sensor stream."""
    if sensor_type == "event":
        return MultiArrayMessage
    if sensor_type in {"imu", "corner"}:
        return DictMessage
    if sensor_type in {
        "color",
        "semantic",
        "depth",
        "navmesh",
        "optical_flow",
        "edge",
        "grayscale",
    }:
        return ArrayMessage
    return None


def sensor_topic(topic_type: str, uuid: str) -> str:
    return f"{topic_type}/{uuid}"


def sensor_frame_id(uuid: str, timestamp: float, simsteps: int) -> str:
    """Pack small sensor metadata into standard Cortex message metadata."""
    return f"{uuid}|{timestamp:.9f}|{simsteps}"


def sensor_metadata_from_frame_id(frame_id: str) -> tuple[str | None, float, int]:
    """Unpack metadata from :func:`sensor_frame_id` with best-effort fallback."""
    parts = frame_id.split("|")
    if len(parts) != 3:
        return None, 0.0, 0
    uuid, timestamp, simsteps = parts
    try:
        return uuid, float(timestamp), int(simsteps)
    except ValueError:
        return uuid, 0.0, 0


def ensure_discovery_daemon(discovery_address: str = DEFAULT_DISCOVERY_ADDRESS) -> None:
    """Fail fast if the standalone Cortex discovery daemon is unavailable."""
    client = DiscoveryClient(
        discovery_address=discovery_address,
        timeout_ms=1000,
        retries=1,
    )
    try:
        response = client._send_request(  # noqa: SLF001 - Cortex exposes no ping API yet.
            DiscoveryRequest(command=DiscoveryCommand.LIST_TOPICS)
        )
    finally:
        client.close()

    if response.status != DiscoveryStatus.OK:
        raise RuntimeError(
            "Cortex discovery daemon returned an error while listing topics: "
            f"{response.message}"
        )


def state_from_message(data: dict) -> dict[str, np.ndarray]:
    """Convert a Cortex DictMessage payload into controller/visualizer state."""
    return {
        "x": np.asarray(data["x"]),
        "q": np.asarray(data["q"]),
        "v": np.asarray(data["v"]),
        "w": np.asarray(data["w"]),
    }


def sensor_topics_from_settings(settings: dict) -> dict[str, list[tuple[str, str]]]:
    """Build explicit visualizer subscription topics from simulator settings."""
    visual_sensors = settings.get("visual_backend", {}).get("sensors", {})
    additional_sensors = settings.get("simulator", {}).get("additional_sensors", {})
    topics: dict[str, list[tuple[str, str]]] = {
        topic_type: [] for topic_type in SENSOR_TOPIC_TYPES.values()
    }
    for uuid, sensor_cfg in {**visual_sensors, **additional_sensors}.items():
        topic_type = SENSOR_TOPIC_TYPES.get(sensor_cfg.get("type"))
        if topic_type is not None:
            topics[topic_type].append((sensor_topic(topic_type, uuid), uuid))
    return topics
