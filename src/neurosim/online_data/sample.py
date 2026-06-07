"""The wire unit of the online data pipeline: a time-aligned multi-sensor sample.

A :class:`TimeAlignedSample` bundles, at a single reference time (the *anchor*
time ``t_us``), one payload per requested sensor UUID:

* ``anchor`` sensors  — the frame(s) whose tick defined this sample boundary.
* ``stream`` sensors  — packets accumulated over the window ``(t_prev, t_us]``.
* ``latest`` sensors  — the most recent held snapshot.
"""

import numpy as np
from typing import Any
from dataclasses import dataclass, field


@dataclass(slots=True)
class SampleMeta:
    """Provenance, time-alignment, and sequence metadata for one sample.

    Attributes:
        worker_id: Producer process index that emitted this sample.
        spec_id: Producer spec / domain-randomization variant index (diversity).
        scene: Scene identifier active for this sample.
        seed: Producer seed (worker seed; trajectory seed lives in the episode).
        t_us: Anchor time in microseconds = the common reference time.
        window_us: Accumulation window length, i.e. ``t_us - t_prev_anchor``.
        anchor_uuids: Sensor UUID(s) whose tick defined this sample boundary.
        episode_id: Globally-unique trajectory id (see :meth:`make_episode_id`).
        step_idx: Monotonic 0-based index of this sample within its episode.
        is_first: True for the first sample of an episode (hidden-state reset).
        is_last: True for the final sample of an episode.
        sample_uid: Globally-monotonic id (dedup / sharding correctness checks).
    """

    worker_id: int
    spec_id: int
    scene: str
    seed: int
    t_us: int
    window_us: int
    anchor_uuids: tuple[str, ...]
    episode_id: int
    step_idx: int
    is_first: bool
    is_last: bool
    sample_uid: int

    @staticmethod
    def make_episode_id(worker_id: int, episode_idx: int) -> int:
        """Compose a process-unique episode id from ``(worker_id, episode_idx)``.

        Packs ``worker_id`` into the high bits so episode ids never collide
        across producer processes (single-node assumption; 32-bit each).
        """
        if not (0 <= worker_id < (1 << 31)):
            raise ValueError(f"worker_id out of range: {worker_id}")
        if not (0 <= episode_idx < (1 << 31)):
            raise ValueError(f"episode_idx out of range: {episode_idx}")
        return (worker_id << 31) | episode_idx


@dataclass(slots=True)
class TimeAlignedSample:
    """One aligned multi-sensor sample: metadata + ``{uuid -> owned payload}``.

    Payload types are sensor-*kind* specific (see :mod:`.schema`):
    ``frame`` -> ``np.ndarray``; ``event_stream`` -> ``dict`` of ``x,y,t,p``
    arrays; ``vector`` -> 1-D ``np.ndarray``; ``vector_stream`` -> 2-D
    ``np.ndarray`` ``(k, D)``.
    """

    meta: SampleMeta
    sensors: dict[str, Any] = field(default_factory=dict)

    def uuids(self) -> list[str]:
        """Sensor UUIDs carried by this sample."""
        return list(self.sensors.keys())


def assert_owned_array(arr: np.ndarray, name: str = "payload") -> np.ndarray:
    """Assert ``arr`` owns its memory (not a view into a reused sim buffer).

    A defensive guard for the §0.10 ownership rule. Returns ``arr`` unchanged so
    it can be used inline. Note: ``OWNDATA`` being False is a *sufficient* signal
    of aliasing; ``True`` does not fully prove independence, but combined with the
    "copy on the way out" discipline in the worker it catches the common bug
    (handing out a slice/view of a buffer that the next step overwrites).
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(arr)!r}")
    if not arr.flags["OWNDATA"]:
        raise ValueError(
            f"{name} does not own its memory (OWNDATA=False); copy it out of the "
            "reused simulator buffer before adding it to a TimeAlignedSample."
        )
    return arr
