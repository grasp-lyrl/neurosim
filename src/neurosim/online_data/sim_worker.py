"""SimulatorWorker: one producer that emits time-aligned samples from a sim.

Wires a ``RandomizedSimulator`` to an :class:`~neurosim.online_data.assembler.AnchorAssembler`:
each episode it (re-)randomizes, flies one trajectory via ``sim.run(callback=...)``,
and on every step converts the raw (GPU) measurements to **owned host memory**
and feeds them to the assembler. Released samples are routed to ``emit_fn``..
"""

import logging
from typing import Any, Callable

import numpy as np

from neurosim.online_data.assembler import AnchorAssembler
from neurosim.online_data.sample import TimeAlignedSample
from neurosim.online_data.schema import SampleSchema, SensorKind

logger = logging.getLogger(__name__)


# cu_esim event field -> host dtype (mirrors HabitatWrapper.render_events to_numpy).
_EVENT_DTYPES = {"x": np.uint16, "y": np.uint16, "t": np.uint64, "p": np.uint8}


def _to_host_array(x: Any) -> np.ndarray:
    """Convert a (possibly GPU) tensor/array to an OWNED, independent host array.

    This is the §0.10 ownership boundary: the returned array never aliases a
    simulator buffer that a later step overwrites. ``copy=True`` guarantees
    independence even when ``tensor.numpy()`` would share memory with a reused
    host tensor.
    """
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    arr = x.numpy() if hasattr(x, "numpy") else np.asarray(x)
    return np.array(arr, copy=True)


def _events_to_host(raw: Any) -> dict[str, np.ndarray]:
    """Convert a cu_esim events struct (``.x/.y/.t/.p``) to an owned packet dict.

    Returns an empty dict when ``raw`` is ``None`` (no events this render).
    """
    if raw is None:
        return {}
    out: dict[str, np.ndarray] = {}
    for field, dtype in _EVENT_DTYPES.items():
        col = getattr(raw, field)
        if hasattr(col, "detach"):
            col = col.detach()
        if hasattr(col, "cpu"):
            col = col.cpu()
        col = col.numpy() if hasattr(col, "numpy") else np.asarray(col)
        out[field] = col.astype(dtype, copy=True)
    return out


class SimulatorWorker:
    """A single data-generation producer.

    Args:
        schema: Resolved delivery schema (roles + UUIDs).
        rsim: An existing ``RandomizedSimulator`` (or mock). If ``None``,
            ``base_settings`` is used to build one (lazy Habitat import).
        base_settings: Settings dict used to build the simulator when ``rsim`` is
            ``None`` (``visual_backend.gpu_id`` is set from ``gpu_id``).
        randomization: Domain-randomization dict (scenes + sensors).
        worker_id / spec_id: Provenance for emitted samples.
        gpu_id: GPU for the simulator's visual backend.
        seed: Worker seed; per-episode RNG is derived from it for determinism.
        ring_caps: ``uuid -> max rows`` per stream sensor.
        emit_fn: Sink for released samples (default: append to ``self.samples``).
        strict_owned: Forwarded to the assembler (ownership backstop).
        validate: Validate ``schema`` against the simulator's sensor configs.
    """

    def __init__(
        self,
        schema: SampleSchema,
        *,
        rsim: Any = None,
        base_settings: dict | None = None,
        randomization: dict | None = None,
        worker_id: int = 0,
        spec_id: int = 0,
        gpu_id: int = 0,
        seed: int = 0,
        ring_caps: dict[str, int] | None = None,
        emit_fn: Callable[[TimeAlignedSample], None] | None = None,
        strict_owned: bool = True,
        validate: bool = True,
    ):
        self.schema = schema
        self.worker_id = worker_id
        self.spec_id = spec_id
        self.gpu_id = gpu_id
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._episode_idx = 0

        self.samples: list[TimeAlignedSample] = []
        self._emit_fn = emit_fn if emit_fn is not None else self.samples.append

        if rsim is None:
            if base_settings is None:
                raise ValueError("Provide either `rsim` or `base_settings`.")
            rsim = self._build_rsim(base_settings, randomization, gpu_id, seed)
        self.rsim = rsim

        if validate:
            self.schema.validate_against(
                self._sensor_configs(), producer=f"worker{worker_id}"
            )

        self.assembler = AnchorAssembler(
            schema,
            worker_id=worker_id,
            spec_id=spec_id,
            ring_caps=ring_caps,
            strict_owned=strict_owned,
        )

    @staticmethod
    def _build_rsim(
        base_settings: dict, randomization: dict | None, gpu_id: int, seed: int
    ):
        # Lazy import: keeps Habitat out of the import path for unit tests.
        import copy

        from neurosim.sims.synchronous_simulator import RandomizedSimulator

        settings = copy.deepcopy(base_settings)
        settings.setdefault("visual_backend", {})["gpu_id"] = gpu_id
        return RandomizedSimulator(
            settings, randomization=randomization, visualizer_disabled=True, seed=seed
        )

    @property
    def sim(self):
        """The inner ``SynchronousSimulator`` (post-build / post-randomize)."""
        return self.rsim.sim

    def _sensor_configs(self) -> dict[str, dict[str, Any]]:
        """Merged ``uuid -> config`` for visual + additional sensors."""
        cfg = self.sim.config
        merged: dict[str, dict[str, Any]] = {}
        merged.update(dict(getattr(cfg, "visual_sensors", {})))
        merged.update(dict(getattr(cfg, "additional_sensors", {})))
        return merged

    def _to_host(self, uuid: str, raw: Any) -> Any:
        """Convert one raw measurement to an owned host payload by sensor kind."""
        kind = self.schema.kind_of(uuid)
        if kind is SensorKind.EVENT_STREAM:
            return _events_to_host(raw)
        # frame / vector / vector_stream all arrive as array-likes here.
        return _to_host_array(raw)

    def _on_sim_step(
        self, measurements: dict, state: dict, sim_time: float, simsteps: int
    ) -> None:
        """Callback for ``sim.run``: convert + feed the assembler, route samples."""
        host: dict[str, Any] = {}
        for uuid in self.schema.deliver_uuids():
            if uuid in measurements:
                host[uuid] = self._to_host(uuid, measurements[uuid])
        t_us = int(round(sim_time * 1e6))
        for sample in self.assembler.on_step(host, t_us):
            self._emit_fn(sample)

    def run_episode(self, episode_idx: int | None = None) -> int:
        """Randomize, fly one trajectory, and emit its samples. Returns count."""
        if episode_idx is None:
            episode_idx = self._episode_idx
        self._episode_idx = episode_idx + 1

        # RandomizedSimulator owns the cadence: it reconfigures scene+sensors
        # every ``resample_every`` episodes (from the DR config) and rebuilds
        # the trajectory in-place every episode. (Dynamics DR stays RL-only.)
        self.rsim.randomize(self._rng)
        scene = ""
        sampled = getattr(self.rsim, "last_sampled_settings", None)
        if sampled:
            scene = sampled.get("visual_backend", {}).get("scene", "")

        before = self.assembler.stats["emitted"]
        self.assembler.begin_episode(
            episode_idx=episode_idx, scene=scene, seed=self.seed
        )
        self.sim.run(callback_hook_=self._on_sim_step)
        for sample in self.assembler.end_episode():
            self._emit_fn(sample)
        return self.assembler.stats["emitted"] - before

    def close(self) -> None:
        if self.rsim is not None:
            self.rsim.close()
