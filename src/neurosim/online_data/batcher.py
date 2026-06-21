"""Batchers: turn a stream of TimeAlignedSamples into batched tensors.

Dispatch is by sensor *kind* via
:data:`neurosim.online_data.schema.BATCH_STRATEGY_FOR_KIND`:

* ``stack``         — frames/vectors stacked to ``(B, ...)`` (preallocated buffer).
* ``concat_counts`` — event/vector streams concatenated with a ``counts`` vector.
  Events are packed ``[x, y, t_anchor - t, p]`` (raw values, anchor-relative
  time); model-side normalization happens downstream (see :func:`shift_events`).

Unlike the archived loader, a :class:`~neurosim.online_data.sample.TimeAlignedSample`
is an **atomic, already-aligned row**: one sample fills row ``i`` for *every*
delivered sensor at once, so there is no per-sensor "ready" bookkeeping.

``ShuffledBatcher`` builds batches in arrival order — diversity comes from
producers interleaving on the shared bus, not from a shuffle buffer.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from neurosim.online_data.sample import SampleMeta, TimeAlignedSample
from neurosim.online_data.schema import SampleSchema, SensorKind

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Event packing (anchor-relative time; normalization is a downstream concern)
# --------------------------------------------------------------------------- #
def shift_events(packet: dict, t_anchor_us: int) -> np.ndarray:
    """Stack an event packet ``{x,y,t,p}`` into an ``(N, 4)`` float32 array.

    Columns are ``[x, y, t_anchor_us - t, p]``: raw pixel coordinates, raw
    polarity, and time measured **backwards from the anchor** in microseconds
    (newest event -> 0, oldest -> ~``window_us``). Events fall in
    ``(t_prev, t_anchor]`` so the relative time is bounded by the sample window,
    which keeps it ``float32``-exact (absolute µs would overflow the mantissa).

    Spatial/temporal *scaling* (``/W``, ``/H``, ``/window``) is a model concern
    and is applied downstream (e.g. in the training loop, ideally on-GPU), not
    here — different models want different normalizations, and ``width`` /
    ``height`` / ``window_us`` all travel in the schema + ``BatchMeta``.
    """
    n = len(packet.get("x", ())) if packet else 0
    if n == 0:
        return np.zeros((0, 4), dtype=np.float32)
    x, y, t, p = packet["x"], packet["y"], packet["t"], packet["p"]
    t_rel = int(t_anchor_us) - t.astype(
        np.int64
    )  # int64: values << 2**31, no underflow
    return np.column_stack([x, y, t_rel, p]).astype(np.float32)


# --------------------------------------------------------------------------- #
# Buffers + batch containers
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class FrameBuffer:
    """Preallocated ``(B, *shape)`` buffer; row ``i`` set per sample, copied out."""

    batch_size: int
    shape: tuple
    dtype: object
    data: np.ndarray = field(init=False)

    def __post_init__(self):
        self.data = np.zeros((self.batch_size, *self.shape), dtype=self.dtype)

    def set(self, i: int, frame: np.ndarray) -> None:
        self.data[i] = frame

    def stacked_copy(self) -> np.ndarray:
        return self.data.copy()


@dataclass(slots=True)
class BatchMeta:
    """Per-row metadata for a batch (stacked :class:`SampleMeta` fields)."""

    t_us: np.ndarray
    window_us: np.ndarray
    episode_id: np.ndarray
    step_idx: np.ndarray
    is_first: np.ndarray
    is_last: np.ndarray
    sample_uid: np.ndarray
    worker_id: np.ndarray
    spec_id: np.ndarray
    scene: list
    anchor_uuids: tuple

    @classmethod
    def from_metas(cls, metas: list[SampleMeta]) -> "BatchMeta":
        col = lambda f, dt: np.array([getattr(m, f) for m in metas], dtype=dt)  # noqa: E731
        return cls(
            t_us=col("t_us", np.int64),
            window_us=col("window_us", np.int64),
            episode_id=col("episode_id", np.int64),
            step_idx=col("step_idx", np.int64),
            is_first=col("is_first", bool),
            is_last=col("is_last", bool),
            sample_uid=col("sample_uid", np.int64),
            worker_id=col("worker_id", np.int64),
            spec_id=col("spec_id", np.int64),
            scene=[m.scene for m in metas],
            anchor_uuids=metas[0].anchor_uuids if metas else (),
        )


class Batch(dict):
    """A batch: ``dict[uuid -> payload]`` with a parallel :attr:`meta`.

    Payloads: ``frame`` -> ``(B, *shape)`` array; ``event_stream`` -> tuple
    ``(counts (B,), events (N_total, 4))`` (matching the training code's
    ``counts, events = batch[event_sensor]`` contract).
    """

    def __init__(self, data: dict, meta: BatchMeta):
        super().__init__(data)
        self.meta = meta


# --------------------------------------------------------------------------- #
# Batchers
# --------------------------------------------------------------------------- #
class Batcher(ABC):
    """Assembles samples into batches according to a :class:`SampleSchema`."""

    def __init__(self, schema: SampleSchema, batch_size: int):
        self.schema = schema
        self.batch_size = batch_size

    @abstractmethod
    def add(self, sample: TimeAlignedSample) -> Batch | None:
        """Add one sample; return a finished :class:`Batch` or ``None``."""
        raise NotImplementedError


class ShuffledBatcher(Batcher):
    """Arrival-order feed-forward batcher (v1)."""

    def __init__(self, schema: SampleSchema, batch_size: int):
        super().__init__(schema, batch_size)

        self._frame_bufs: dict[str, FrameBuffer] = {}
        self._event_uuids: list[str] = []
        for uuid in schema.deliver_uuids():
            spec = schema.specs[uuid]
            strat = spec.batch_strategy
            if strat == "stack":
                if spec.shape is None:
                    raise ValueError(
                        f"sensor {uuid!r} (kind {spec.kind.value}) needs a fixed "
                        "shape to batch with 'stack'."
                    )
                self._frame_bufs[uuid] = FrameBuffer(
                    batch_size, spec.shape, np.dtype(spec.dtype or "float32")
                )
            elif strat == "concat_counts":
                if spec.kind is not SensorKind.EVENT_STREAM:
                    raise NotImplementedError(
                        f"batching for kind {spec.kind.value!r} (sensor {uuid!r}) "
                        "is not supported; only frame + event_stream are."
                    )
                self._event_uuids.append(uuid)
            else:  # pragma: no cover - dispatch table is total over kinds
                raise ValueError(f"unknown batch strategy {strat!r} for {uuid!r}")

        self._reset_accumulators()

    def _reset_accumulators(self) -> None:
        self._idx = 0
        self._event_rows: dict[str, list[np.ndarray]] = {
            u: [] for u in self._event_uuids
        }
        self._metas: list[SampleMeta] = []

    def add(self, sample: TimeAlignedSample) -> Batch | None:
        i = self._idx
        for uuid, buf in self._frame_bufs.items():
            buf.set(i, sample.sensors[uuid])
        for uuid in self._event_uuids:
            self._event_rows[uuid].append(
                shift_events(sample.sensors[uuid], sample.meta.t_us)
            )
        self._metas.append(sample.meta)
        self._idx += 1

        if self._idx >= self.batch_size:
            return self._emit()
        return None

    def _emit(self) -> Batch:
        data: dict = {}
        for uuid, buf in self._frame_bufs.items():
            data[uuid] = buf.stacked_copy()
        for uuid in self._event_uuids:
            rows = self._event_rows[uuid]
            counts = np.array([r.shape[0] for r in rows], dtype=np.int32)
            events = (
                np.concatenate(rows, axis=0)
                if rows
                else np.zeros((0, 4), dtype=np.float32)
            )
            data[uuid] = (counts, events)
        meta = BatchMeta.from_metas(self._metas)
        self._reset_accumulators()
        return Batch(data, meta)
