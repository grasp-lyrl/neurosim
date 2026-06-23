"""Sensor schema: kinds, roles, and the batching dispatch table.

The pipeline is generic over sensors *by UUID*. No sensor type is special-cased;
instead every requested UUID resolves to:

* a **kind** (:class:`SensorKind`) — derived from the simulator ``sensor_type`` —
  that fully determines how its payload is batched, and
* a **role** (:class:`SensorRole`) — chosen in config — that determines how the
  producer's anchor assembler treats it (boundary / accumulate / hold).

This allows us to support arbitrary sensor combinations and new sensor types, e.g,
events+depth, events+IMU, color+pose, etc., without too much work.
(see :data:`BATCH_STRATEGY_FOR_KIND`), and the worker dispatches on *role*.

A :class:`SampleSchema` is built once at loader init from the requested sensors
and validated against every producer's sensor config (fail fast if a UUID, kind,
shape, or dtype disagrees across producers).
"""

import logging
from enum import Enum
from typing import Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class SensorKind(str, Enum):
    """How a sensor's payload is shaped and therefore batched."""

    FRAME = "frame"  # dense (H, W[, C]) array
    EVENT_STREAM = "event_stream"  # variable-length {x,y,t,p} packet
    VECTOR = "vector"  # fixed (D,) reading
    VECTOR_STREAM = "vector_stream"  # variable (k, D) readings over a window


class SensorRole(str, Enum):
    """How the producer's anchor assembler treats a sensor."""

    ANCHOR = "anchor"  # its tick defines a sample boundary
    STREAM = "stream"  # accumulated into packets between anchor ticks
    LATEST = "latest"  # most-recent value held and attached at the anchor


# Simulator ``sensor_type`` -> default kind. Overridable per UUID at build time.
SENSOR_TYPE_TO_KIND: dict[str, SensorKind] = {
    "color": SensorKind.FRAME,
    "depth": SensorKind.FRAME,
    "semantic": SensorKind.FRAME,
    "grayscale": SensorKind.FRAME,
    "optical_flow": SensorKind.FRAME,
    "edge": SensorKind.FRAME,
    "navmesh": SensorKind.FRAME,
    "event": SensorKind.EVENT_STREAM,
    "imu": SensorKind.VECTOR_STREAM,
}

# Batching strategy per kind. The batcher dispatches on this; the table must stay
# *total* over SensorKind (asserted in tests).
BATCH_STRATEGY_FOR_KIND: dict[SensorKind, str] = {
    SensorKind.FRAME: "stack",  # -> (B, …) / (B, L, …)
    SensorKind.VECTOR: "stack",  # -> (B, D)
    SensorKind.EVENT_STREAM: "concat_counts",  # -> concatenated + counts vector
    SensorKind.VECTOR_STREAM: "concat_counts",  # -> concatenated + counts vector
}

# Kinds that are inherently streaming (cannot be anchors).
_STREAM_KINDS = (SensorKind.EVENT_STREAM, SensorKind.VECTOR_STREAM)

# Frame channel count by sensor_type: None => 2-D (H, W); int => (H, W, C).
_FRAME_CHANNELS: dict[str, int | None] = {
    "color": 3,
    "depth": None,
    "semantic": None,
    "grayscale": None,
    "optical_flow": 2,
    "edge": None,
    "navmesh": None,
}

# Frame dtype by sensor_type (best-effort; used for buffer preallocation later).
_FRAME_DTYPE: dict[str, str] = {
    "color": "uint8",
    "depth": "float32",
    "semantic": "int32",
    "grayscale": "uint8",
    "optical_flow": "float32",
    "edge": "uint8",
    "navmesh": "uint8",
}


def infer_kind(
    sensor_type: str, override: SensorKind | str | None = None
) -> SensorKind:
    """Resolve the :class:`SensorKind` for a simulator ``sensor_type``."""
    if override is not None:
        return SensorKind(override)
    try:
        return SENSOR_TYPE_TO_KIND[sensor_type]
    except KeyError:
        raise ValueError(
            f"Unknown sensor_type {sensor_type!r}; pass a kind override or extend "
            "SENSOR_TYPE_TO_KIND."
        )


@dataclass(slots=True)
class SensorSpec:
    """Resolved description of one requested sensor.

    Attributes:
        uuid: Sensor UUID.
        kind: Batching kind (:class:`SensorKind`).
        role: Assembler role (:class:`SensorRole`).
        sensor_type: Original simulator ``sensor_type`` (provenance).
        shape: Per-sample payload shape when fixed (frames/vectors); ``None`` for
            variable-length streams.
        dtype: Payload dtype string when known.
        extras: Free-form passthrough (event geometry, ring caps, feature dim …).
    """

    uuid: str
    kind: SensorKind
    role: SensorRole
    sensor_type: str = ""
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def batch_strategy(self) -> str:
        return BATCH_STRATEGY_FOR_KIND[self.kind]


def _infer_shape_dtype(
    sensor_type: str, kind: SensorKind, cfg: dict[str, Any]
) -> tuple[tuple[int, ...] | None, str | None]:
    """Best-effort (shape, dtype) for a sensor from its config dict."""
    if kind is SensorKind.FRAME:
        h, w = cfg.get("height"), cfg.get("width")
        if h is None or w is None:
            return None, _FRAME_DTYPE.get(sensor_type)
        channels = _FRAME_CHANNELS.get(sensor_type)
        shape = (int(h), int(w)) if channels is None else (int(h), int(w), channels)
        return shape, _FRAME_DTYPE.get(sensor_type)
    if kind is SensorKind.VECTOR:
        d = cfg.get("dim")
        return ((int(d),) if d is not None else None), cfg.get("dtype")
    # Streams are variable-length: no fixed per-sample shape.
    return None, cfg.get("dtype")


@dataclass(slots=True)
class SampleSchema:
    """Resolved, validated schema for one online-data run.

    Built via :meth:`from_sensor_configs`. Carries one :class:`SensorSpec` per
    delivered UUID (in deliver order) plus the role partitions.
    """

    specs: dict[str, SensorSpec]
    anchor_uuids: tuple[str, ...]
    stream_uuids: tuple[str, ...]
    latest_uuids: tuple[str, ...]

    # ------------------------------------------------------------------ build
    @classmethod
    def from_sensor_configs(
        cls,
        sensor_configs: dict[str, dict[str, Any]],
        *,
        anchor: list[str],
        stream: list[str] | None = None,
        latest: list[str] | None = None,
        deliver: list[str] | None = None,
        kind_overrides: dict[str, SensorKind | str] | None = None,
        extras: dict[str, dict[str, Any]] | None = None,
    ) -> "SampleSchema":
        """Build and validate a schema from simulator sensor configs + roles.

        Args:
            sensor_configs: ``uuid -> config dict`` (must contain ``type`` and,
                for frames, ``height``/``width``). Typically the merged
                ``visual_backend.sensors`` + ``simulator.additional_sensors``.
            anchor / stream / latest: UUID lists per role. ``anchor`` must be
                non-empty and may not contain a streaming-kind sensor.
            deliver: UUIDs the loader yields (default: anchor+stream+latest,
                de-duplicated, role order).
            kind_overrides: Force a kind for specific UUIDs.
            extras: Per-UUID passthrough merged into each :class:`SensorSpec`.
        """
        stream = list(stream or [])
        latest = list(latest or [])
        anchor = list(anchor or [])
        kind_overrides = kind_overrides or {}
        extras = extras or {}

        if not anchor:
            raise ValueError("SampleSchema requires at least one anchor sensor.")

        role_of: dict[str, SensorRole] = {}
        for uuid in anchor:
            role_of[uuid] = SensorRole.ANCHOR
        for uuid in stream:
            if uuid in role_of:
                raise ValueError(f"Sensor {uuid!r} assigned to multiple roles.")
            role_of[uuid] = SensorRole.STREAM
        for uuid in latest:
            if uuid in role_of:
                raise ValueError(f"Sensor {uuid!r} assigned to multiple roles.")
            role_of[uuid] = SensorRole.LATEST

        if deliver is None:
            deliver = list(dict.fromkeys([*anchor, *stream, *latest]))

        specs: dict[str, SensorSpec] = {}
        for uuid in deliver:
            if uuid not in role_of:
                raise ValueError(
                    f"Delivered sensor {uuid!r} has no role (anchor/stream/latest)."
                )
            if uuid not in sensor_configs:
                raise ValueError(
                    f"Sensor {uuid!r} not found in sensor_configs; available: "
                    f"{sorted(sensor_configs)}"
                )
            cfg = sensor_configs[uuid]
            sensor_type = cfg.get("type")
            if not sensor_type:
                raise ValueError(f"Sensor {uuid!r} config missing 'type'.")
            kind = infer_kind(sensor_type, kind_overrides.get(uuid))
            role = role_of[uuid]

            if role is SensorRole.ANCHOR and kind in _STREAM_KINDS:
                raise ValueError(
                    f"Anchor sensor {uuid!r} has streaming kind {kind.value!r}; "
                    "anchors must produce a discrete frame/vector tick."
                )

            shape, dtype = _infer_shape_dtype(sensor_type, kind, cfg)
            spec_extras = dict(extras.get(uuid, {}))
            # Capture event geometry for downstream normalization, if present.
            if kind is SensorKind.EVENT_STREAM:
                for k in ("width", "height"):
                    if k in cfg and k not in spec_extras:
                        spec_extras[k] = int(cfg[k])

            specs[uuid] = SensorSpec(
                uuid=uuid,
                kind=kind,
                role=role,
                sensor_type=sensor_type,
                shape=shape,
                dtype=dtype,
                extras=spec_extras,
            )

        return cls(
            specs=specs,
            anchor_uuids=tuple(u for u in deliver if role_of[u] is SensorRole.ANCHOR),
            stream_uuids=tuple(u for u in deliver if role_of[u] is SensorRole.STREAM),
            latest_uuids=tuple(u for u in deliver if role_of[u] is SensorRole.LATEST),
        )

    # ----------------------------------------------------------- dispatch API
    def deliver_uuids(self) -> list[str]:
        return list(self.specs.keys())

    def kind_of(self, uuid: str) -> SensorKind:
        return self.specs[uuid].kind

    def role_of(self, uuid: str) -> SensorRole:
        return self.specs[uuid].role

    def batch_strategy_of(self, uuid: str) -> str:
        return self.specs[uuid].batch_strategy

    # ------------------------------------------------------ cross-producer check
    def validate_against(
        self, sensor_configs: dict[str, dict[str, Any]], *, producer: str = ""
    ) -> None:
        """Assert another producer's sensor configs match this schema.

        Re-derives kind/shape/dtype for each delivered UUID from
        ``sensor_configs`` and raises on the first disagreement, so heterogeneous
        producers can't silently emit incompatible samples.
        """
        where = f" (producer={producer})" if producer else ""
        for uuid, spec in self.specs.items():
            if uuid not in sensor_configs:
                raise ValueError(f"Sensor {uuid!r} missing from producer{where}.")
            cfg = sensor_configs[uuid]
            sensor_type = cfg.get("type")
            kind = infer_kind(sensor_type)
            if kind != spec.kind:
                raise ValueError(
                    f"Sensor {uuid!r} kind mismatch{where}: schema={spec.kind.value}, "
                    f"producer={kind.value}."
                )
            shape, dtype = _infer_shape_dtype(sensor_type, kind, cfg)
            if spec.shape is not None and shape is not None and shape != spec.shape:
                raise ValueError(
                    f"Sensor {uuid!r} shape mismatch{where}: schema={spec.shape}, "
                    f"producer={shape}."
                )
            if spec.dtype is not None and dtype is not None and dtype != spec.dtype:
                raise ValueError(
                    f"Sensor {uuid!r} dtype mismatch{where}: schema={spec.dtype}, "
                    f"producer={dtype}."
                )
