"""Shared Cortex topic and message helpers for asynchronous simulator nodes."""

import numpy as np

from cortex.discovery.client import DiscoveryClient
from cortex.discovery.daemon import DEFAULT_DISCOVERY_ADDRESS
from cortex.discovery.protocol import (
    DiscoveryCommand,
    DiscoveryRequest,
    DiscoveryStatus,
)

STATE_TOPIC = "state"
CONTROL_TOPIC = "control"

SENSOR_TOPIC_TYPES = {
    "event": "events",
    "imu": "imu",
    "color": "color",
    "depth": "depth",
}

SUBSCRIBE_DEFAULTS = {
    "queue_size": 1000,
    "wait_for_topic": True,
    "topic_timeout": 60.0,
}


def sensor_topic(topic_type: str, uuid: str) -> str:
    return f"{topic_type}/{uuid}"


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
        k: [] for k in ("events", "imu", "color", "depth")
    }
    for uuid, sensor_cfg in {**visual_sensors, **additional_sensors}.items():
        topic_type = SENSOR_TOPIC_TYPES.get(sensor_cfg.get("type"))
        if topic_type is not None:
            topics[topic_type].append((sensor_topic(topic_type, uuid), uuid))
    return topics
