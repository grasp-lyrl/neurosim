"""Anchor-driven sample assembly (pure, simulator-agnostic).

This is the heart of a producer (plan §6.1). It consumes per-step sensor
measurements (already converted to **owned host memory** by the worker) and emits
:class:`~neurosim.online_data.sample.TimeAlignedSample`s whose boundaries are
defined by an *anchor* sensor tick:

* ``stream`` sensors  — accumulated into bounded (drop-oldest) packets between
  anchor ticks, flushed at the anchor (window ``(t_prev, t_anchor]``).
* ``anchor``/``latest`` sensors — the most-recent value is held and attached at
  the anchor tick. The *primary* anchor (``schema.anchor_uuids[0]``) firing is
  what triggers a sample; secondary anchors behave as held-latest, so a
  depth+color joint anchor that fires on the same tick gets both fresh.

Design choices (locked while implementing PR2):

* **Always emit.** Every primary-anchor tick (after warm-up) emits a sample, even
  if the stream window is empty — event-count gating is left to training (Q4).
* **Ring cap, drop-oldest.** A stream that overflows its cap keeps the newest
  rows and counts/logs the overflow; no time-based window cap (Q4).
* **Hold-back-by-one.** A sample is buffered and released when the *next* sample
  is assembled (or at :meth:`end_episode`), so ``is_first``/``is_last`` are exact
  in a streaming callback — the basis for the recurrent (TBPTT) batcher later.
* **Ownership backstop.** Inputs must be owned host arrays (the worker's
  ``_to_host`` is the copy boundary); ``strict_owned`` asserts ``OWNDATA`` so a
  reused-buffer aliasing bug is caught immediately (plan §0.10).

This module imports neither torch nor Habitat, so it is exhaustively unit-testable
with synthetic numpy measurements.
"""

import logging
from collections import defaultdict

import numpy as np

from neurosim.online_data.sample import (
    SampleMeta,
    TimeAlignedSample,
    assert_owned_array,
)
from neurosim.online_data.schema import SampleSchema, SensorRole

logger = logging.getLogger(__name__)


class StreamAccumulator:
    """Accumulates equal-leading-dim packets for one stream sensor, ring-capped.

    A *packet* is a ``dict[str, np.ndarray]`` whose arrays share a leading
    dimension (the per-packet count) — e.g. events ``{x, y, t, p}`` each ``(N,)``,
    or IMU ``{data: (k, D), t: (k,)}``. :meth:`flush` concatenates everything
    accumulated since the last flush into fresh (owned) arrays and resets.
    """

    def __init__(self, ring_cap: int | None = None):
        self.ring_cap = ring_cap
        self._packets: list[dict[str, np.ndarray]] = []
        self._count = 0
        self._keys: tuple[str, ...] | None = None
        self._dtypes: dict[str, np.dtype] | None = None

    def __len__(self) -> int:
        return self._count

    @staticmethod
    def _packet_len(packet: dict[str, np.ndarray]) -> int:
        lengths = {k: int(v.shape[0]) for k, v in packet.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(
                f"stream packet arrays must share leading dim, got {lengths}"
            )
        return next(iter(lengths.values())) if lengths else 0

    def append(self, packet: dict[str, np.ndarray]) -> int:
        """Append a packet; return number of (oldest) rows dropped to respect cap."""
        n = self._packet_len(packet)
        if self._keys is None and packet:
            self._keys = tuple(packet.keys())
            self._dtypes = {k: packet[k].dtype for k in packet}
        if n:
            self._packets.append(packet)
            self._count += n
        return self._trim()

    def _trim(self) -> int:
        if self.ring_cap is None:
            return 0
        excess = self._count - self.ring_cap
        if excess <= 0:
            return 0
        dropped = excess
        while excess > 0 and self._packets:
            head = self._packets[0]
            hn = self._packet_len(head)
            if hn <= excess:
                self._packets.pop(0)
                self._count -= hn
                excess -= hn
            else:
                # Drop the oldest `excess` rows of the head packet.
                self._packets[0] = {k: v[excess:] for k, v in head.items()}
                self._count -= excess
                excess = 0
        return dropped

    def flush(self) -> dict[str, np.ndarray]:
        """Concatenate accumulated packets into fresh owned arrays and reset.

        Returns empty (zero-length) arrays for the known keys when nothing was
        accumulated since the last flush, or ``{}`` if no packet was ever seen.
        """
        if not self._packets:
            if self._keys is None:
                return {}
            return {k: np.empty((0,), dtype=self._dtypes[k]) for k in self._keys}
        keys = self._packets[0].keys()
        out = {k: np.concatenate([p[k] for p in self._packets]) for k in keys}
        self._packets = []
        self._count = 0
        return out


class AnchorAssembler:
    """Turns a per-step measurement stream into time-aligned samples.

    Args:
        schema: Resolved :class:`SampleSchema` (roles + delivered UUIDs).
        worker_id: Producer process index (provenance / episode-id high bits).
        spec_id: Producer spec / DR-variant index (diversity provenance).
        ring_caps: Optional ``uuid -> max rows`` cap per stream sensor.
        strict_owned: Assert every ingested array owns its memory (backstop for
            the §0.10 copy rule).
        uid_start: Starting value for the globally-monotonic ``sample_uid``.
    """

    def __init__(
        self,
        schema: SampleSchema,
        *,
        worker_id: int = 0,
        spec_id: int = 0,
        ring_caps: dict[str, int] | None = None,
        strict_owned: bool = True,
        uid_start: int = 0,
    ):
        self.schema = schema
        self.worker_id = worker_id
        self.spec_id = spec_id
        self.strict_owned = strict_owned
        ring_caps = ring_caps or {}

        self._streams = {
            uuid: StreamAccumulator(ring_caps.get(uuid)) for uuid in schema.stream_uuids
        }
        # Anchors + latest sensors are "held": newest snapshot attached at anchor.
        self._held_uuids = (*schema.anchor_uuids, *schema.latest_uuids)
        # All held sensors must be seen once before the first emit (warm-up).
        self._required_held = set(self._held_uuids)
        if not schema.anchor_uuids:
            raise ValueError("AnchorAssembler requires at least one anchor sensor.")
        self._primary_anchor = schema.anchor_uuids[0]

        self._uid = int(uid_start)
        self.stats: dict[str, int] = defaultdict(int)

        # Per-episode state (set in begin_episode).
        self._held: dict[str, np.ndarray] = {}
        self._episode_id = 0
        self._scene = ""
        self._seed = 0
        self._step_idx = 0
        self._t_prev_us = 0
        self._pending: TimeAlignedSample | None = None
        self._in_episode = False

    # ------------------------------------------------------------------ episode
    def begin_episode(
        self,
        *,
        episode_idx: int,
        scene: str = "",
        seed: int = 0,
        t_start_us: int = 0,
    ) -> None:
        """Reset accumulators for a new episode/trajectory."""
        for acc in self._streams.values():
            acc.flush()  # discard any residue
        self._held = {}
        self._episode_id = SampleMeta.make_episode_id(self.worker_id, episode_idx)
        self._scene = scene
        self._seed = seed
        self._step_idx = 0
        self._t_prev_us = int(t_start_us)
        self._pending = None
        self._in_episode = True

    def on_step(
        self, measurements: dict[str, object], t_us: int
    ) -> list[TimeAlignedSample]:
        """Process one simulator step; return any released sample(s).

        ``measurements`` maps UUID -> owned payload for sensors sampled this step
        (frames as ``np.ndarray``; stream packets as ``dict[str, np.ndarray]``).
        Returns at most one released sample under hold-back-by-one (a list for
        API generality / future batched producers).
        """
        if not self._in_episode:
            raise RuntimeError("on_step called before begin_episode().")

        for uuid, payload in measurements.items():
            spec = self.schema.specs.get(uuid)
            if spec is None:
                continue  # sensor not requested for delivery
            if spec.role is SensorRole.STREAM:
                packet = self._ingest_stream(uuid, payload)
                dropped = self._streams[uuid].append(packet)
                if dropped:
                    self.stats[f"overflow_drops:{uuid}"] += dropped
                    logger.warning(
                        "stream %s overflowed ring cap; dropped %d oldest rows",
                        uuid,
                        dropped,
                    )
            else:  # anchor or latest -> held snapshot
                self._held[uuid] = self._ingest_frame(uuid, payload)

        emitted: list[TimeAlignedSample] = []
        if self._primary_anchor in measurements:
            if not self._warmed_up():
                self.stats["warmup_skips"] += 1
                return emitted
            sample = self._assemble(int(t_us))
            if self._pending is not None:
                emitted.append(self._pending)
            self._pending = sample
            self._t_prev_us = int(t_us)
            self._step_idx += 1
        return emitted

    def end_episode(self) -> list[TimeAlignedSample]:
        """Flush the held-back final sample with ``is_last=True``."""
        self._in_episode = False
        if self._pending is None:
            return []
        self._pending.meta.is_last = True
        out = [self._pending]
        self._pending = None
        return out

    # ------------------------------------------------------------------ helpers
    def _warmed_up(self) -> bool:
        return self._required_held <= set(self._held)

    def _ingest_frame(self, uuid: str, payload: object) -> np.ndarray:
        if not isinstance(payload, np.ndarray):
            raise TypeError(
                f"held sensor {uuid!r} payload must be np.ndarray, got {type(payload)!r}"
            )
        if self.strict_owned:
            assert_owned_array(payload, name=uuid)
        return payload

    def _ingest_stream(self, uuid: str, payload: object) -> dict[str, np.ndarray]:
        if not isinstance(payload, dict):
            raise TypeError(
                f"stream sensor {uuid!r} payload must be dict of arrays, "
                f"got {type(payload)!r}"
            )
        if self.strict_owned:
            for k, v in payload.items():
                assert_owned_array(v, name=f"{uuid}.{k}")
        return payload

    def _assemble(self, t_us: int) -> TimeAlignedSample:
        sensors: dict[str, object] = {}
        for uuid in self.schema.deliver_uuids():
            if self.schema.role_of(uuid) is SensorRole.STREAM:
                sensors[uuid] = self._streams[uuid].flush()
            else:
                sensors[uuid] = self._held[uuid]

        meta = SampleMeta(
            worker_id=self.worker_id,
            spec_id=self.spec_id,
            scene=self._scene,
            seed=self._seed,
            t_us=t_us,
            window_us=t_us - self._t_prev_us,
            anchor_uuids=self.schema.anchor_uuids,
            episode_id=self._episode_id,
            step_idx=self._step_idx,
            is_first=(self._step_idx == 0),
            is_last=False,
            sample_uid=self._uid,
        )
        self._uid += 1
        self.stats["emitted"] += 1
        return TimeAlignedSample(meta=meta, sensors=sensors)
