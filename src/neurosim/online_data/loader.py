"""OnlineDataLoader

Construct with a schema + producer settings, iterate batches, close::

    loader = OnlineDataLoader(schema, batch_size=32, base_settings=settings,
                              num_producers=8, gpu_ids=[0])   # 8 sims on gpu0
    for batch in loader:          # Batch: dict[uuid -> payload] (+ batch.meta)
        events = batch["event_camera_1"]   # (counts, events)
        depth  = batch["depth_camera_1"]   # (B, H, W)
    loader.close()

**M producer processes** each own a :class:`~neurosim.online_data.sim_worker.SimulatorWorker`
on an assigned GPU and push time-aligned samples to one bounded
:class:`~neurosim.online_data.bus.SampleBus`; the consumer (this process) pops them
and builds batches inline with a :class:`~neurosim.online_data.batcher.ShuffledBatcher`.
Because all producers share the bus, consecutive samples come from different specs →
**diverse batches**. The bounded bus decouples sim from training (producers run ahead)
and bounds memory (backpressure). Producers use ``spawn`` (Habitat/EGL + CUDA must
not fork).

``base_settings=None`` / ``start=False`` skips producers entirely so the consumer
path can be driven by feeding ``loader.bus`` directly (tests).
"""

import time
import queue
import logging
from pathlib import Path
from dataclasses import dataclass, field

from neurosim.online_data.bus import SampleBus
from neurosim.online_data.batcher import ShuffledBatcher
from neurosim.online_data.schema import SampleSchema

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ProducerSpec:
    """One simulator producer's configuration (GPU, seed, diversity tag)."""

    base_settings: dict
    randomization: dict | None = None
    gpu_id: int = 0
    seed: int = 0
    spec_id: int = 0
    ring_caps: dict = field(default_factory=dict)


def build_producer_specs(
    base_settings: dict,
    *,
    num_producers: int,
    gpu_ids: list[int] | None = None,
    base_seed: int = 0,
    randomization: dict | None = None,
    ring_caps: dict | None = None,
) -> list[ProducerSpec]:
    """Build ``num_producers`` specs sharing one config, distinct seeds + GPUs.

    Diversity follows the RL convention: the *same* settings + DR with
    ``seed = base_seed + i`` yields a different domain-randomization realization
    per producer (distinct scenes/sensor params + trajectories). ``gpu_ids`` is
    cycled (e.g. ``[0]`` places every producer on gpu0). Randomization cadence
    (scene reconfigure every N episodes, trajectory every episode) lives in the
    ``randomization`` dict and is owned by ``RandomizedSimulator.randomize``.
    """
    from neurosim.core.utils import explicit_gpu_map  # lazy: keeps import light

    gpus = explicit_gpu_map(num_producers, gpu_ids)
    return [
        ProducerSpec(
            base_settings=base_settings,
            randomization=randomization,
            gpu_id=gpus[i],
            seed=base_seed + i,
            spec_id=i,
            ring_caps=ring_caps or {},
        )
        for i in range(num_producers)
    ]


def _configure_producer_logger(log_dir: Path, worker_id: int, gpu_id: int) -> None:
    """Attach a per-producer file logger inside the child process."""
    workers = Path(log_dir) / "producers"
    workers.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fh = logging.FileHandler(
        workers / f"producer_{worker_id:03d}.log", encoding="utf-8"
    )
    fh.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(fh)
    logger.info("producer %d start gpu_id=%d", worker_id, gpu_id)


def _write_run_setup(log_dir: Path, payload: dict) -> None:
    import yaml

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "run_setup.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, default_flow_style=False, sort_keys=False)


def _run_producer_process(
    bus: SampleBus,
    stop_event,
    schema: SampleSchema,
    spec: ProducerSpec,
    worker_id: int,
    log_dir,
) -> None:
    """Producer-process entry point: run a worker, push samples until stopped.

    Module-level (picklable for ``spawn``). Episodes run to completion; shutdown
    is via process termination from :meth:`OnlineDataLoader.close` (the blocking
    ``sim.run`` inside an episode does not poll ``stop_event``).
    """
    from neurosim.online_data.sim_worker import SimulatorWorker

    if log_dir is not None:
        _configure_producer_logger(log_dir, worker_id, spec.gpu_id)

    def emit(sample):
        # Block until queued, but wake periodically to honor shutdown.
        while not stop_event.is_set():
            try:
                bus.put(sample, timeout=0.5)
                return
            except queue.Full:
                continue

    worker = SimulatorWorker(
        schema,
        base_settings=spec.base_settings,
        randomization=spec.randomization,
        emit_fn=emit,
        worker_id=worker_id,
        spec_id=spec.spec_id,
        gpu_id=spec.gpu_id,
        seed=spec.seed,
        ring_caps=spec.ring_caps,
    )

    # Survive occasional bad episodes instead of dying.
    max_consecutive_failures = 10
    consecutive = 0
    try:
        while not stop_event.is_set():
            try:
                worker.run_episode()
                consecutive = 0
            except Exception:
                consecutive += 1
                logger.exception(
                    "producer %d: episode failed (%d consecutive); retrying next episode",
                    worker_id,
                    consecutive,
                )
                if consecutive >= max_consecutive_failures:
                    logger.error(
                        "producer %d: %d consecutive episode failures; stopping producer",
                        worker_id,
                        consecutive,
                    )
                    break
    finally:
        worker.close()


class OnlineDataLoader:
    """Façade over M producer processes + bus + batcher.

    Args:
        schema: Resolved delivery schema.
        batch_size: Samples per batch.
        producers: Explicit list of :class:`ProducerSpec` (heterogeneous specs).
            If given, ``base_settings``/``num_producers``/... are ignored.
        base_settings: Settings shared by ``num_producers`` auto-built specs
            (``None`` => no producers; feed ``self.bus`` manually).
        randomization: Domain-randomization dict for auto-built specs.
        num_producers: Number of producers when auto-building from settings.
        gpu_ids: GPU ids cycled across producers (default ``[0]``).
        base_seed: Per-producer seed = ``base_seed + i``.
        ring_caps: ``uuid -> max rows`` cap per stream sensor (all producers).
            Randomization cadence (scene reconfigure every N episodes; trajectory
            every episode) is set in ``randomization`` (``resample_every`` /
            ``trajectory``) and owned by ``RandomizedSimulator.randomize``.
        bus_maxsize: Bus capacity (backpressure bound).
        mp_context: Start method (``"spawn"`` for CUDA/Habitat).
        get_timeout: Consumer poll interval / producer-death check.
        stall_warn_s: If no sample arrives for this many seconds while producers are
            still alive, log a one-time warning.``0`` disables the watchdog.
        log_dir: If set, write ``run_setup.yaml`` + per-producer logs.
        start: Start producers immediately.
    """

    def __init__(
        self,
        schema: SampleSchema,
        batch_size: int,
        *,
        producers: list[ProducerSpec] | None = None,
        base_settings: dict | None = None,
        randomization: dict | None = None,
        num_producers: int = 1,
        gpu_ids: list[int] | None = None,
        base_seed: int = 0,
        ring_caps: dict | None = None,
        bus_maxsize: int = 256,
        mp_context: str = "spawn",
        get_timeout: float = 1.0,
        stall_warn_s: float = 60.0,
        log_dir=None,
        start: bool = True,
    ):
        import multiprocessing as mp

        self.schema = schema
        self.batch_size = batch_size
        self._get_timeout = get_timeout
        self._stall_warn_s = stall_warn_s
        self._log_dir = Path(log_dir) if log_dir is not None else None

        if producers is not None:
            self._specs = list(producers)
        elif base_settings is not None:
            self._specs = build_producer_specs(
                base_settings,
                num_producers=num_producers,
                gpu_ids=gpu_ids,
                base_seed=base_seed,
                randomization=randomization,
                ring_caps=ring_caps,
            )
        else:
            self._specs = []

        self.batcher = ShuffledBatcher(schema, batch_size)

        self._ctx = mp.get_context(mp_context)
        self.bus = SampleBus(maxsize=bus_maxsize, ctx=self._ctx)
        self._stop = self._ctx.Event()
        self._procs: list = []

        if start and self._specs:
            self.start()

    @classmethod
    def from_config(cls, config, *, start: bool = True, log_dir=None, **overrides):
        """Build a loader from a single YAML/dict — schema roles, DR, and knobs.

        ``config`` is a path or dict with an ``online_data`` block; everything else
        in the dict (``simulator`` / ``visual_backend`` / ``dynamics`` / ...) is the
        base simulator settings, **unless** ``online_data.base_settings`` names a
        separate settings file/dict to use instead. The ``online_data`` block::

            online_data:
              # base_settings: configs/some-settings.yaml   # optional; else use this file
              batch_size: 4
              num_producers: 2
              gpu_ids: [0]
              base_seed: 0
              bus_maxsize: 256            # optional
              ring_caps: {event_camera_1: 5000000}   # optional
              roles:
                anchor: [depth_camera_1]
                stream: [event_camera_1]
                latest: []               # optional
                deliver: [...]           # optional (default: anchor+stream+latest)
              randomization:             # optional; same grammar as elsewhere
                resample_every: 1
                scenes:  [{name: ..., path: ...}]
                scenes_glob: data/hm3d/*/*.basis.glb   # optional; expands to scenes
                sensors: {...}
                trajectory: {...}

        ``overrides`` are forwarded to ``__init__`` (e.g. ``mp_context``,
        ``get_timeout``), taking precedence over the config.
        """
        import copy

        import yaml

        if isinstance(config, (str, Path)):
            with open(config, "r", encoding="utf-8") as fh:
                config = yaml.safe_load(fh)
        cfg = copy.deepcopy(config)

        od = cfg.pop("online_data", None)
        if not od:
            raise ValueError("config is missing the required `online_data` block")

        base = od.get("base_settings")
        if base is None:
            base_settings = cfg  # the rest of this file is the sim settings
        elif isinstance(base, (str, Path)):
            with open(base, "r", encoding="utf-8") as fh:
                base_settings = yaml.safe_load(fh)
        else:
            base_settings = base

        roles = od.get("roles") or {}
        if not roles.get("anchor"):
            raise ValueError("config `online_data.roles.anchor` must be non-empty")

        # Sensors visible to the schema: visual + additional (IMU, ...), like the worker.
        all_sensors = {
            **base_settings.get("visual_backend", {}).get("sensors", {}),
            **base_settings.get("simulator", {}).get("additional_sensors", {}),
        }
        role_uuids = list(
            dict.fromkeys(
                [
                    *roles.get("anchor", []),
                    *roles.get("stream", []),
                    *roles.get("latest", []),
                    *roles.get("deliver", []),
                ]
            )
        )
        missing = [u for u in role_uuids if u not in all_sensors]
        if missing:
            raise ValueError(
                f"roles reference sensors not in settings: {missing}; "
                f"available: {sorted(all_sensors)}"
            )

        schema = SampleSchema.from_sensor_configs(
            {u: all_sensors[u] for u in role_uuids},
            anchor=roles["anchor"],
            stream=roles.get("stream"),
            latest=roles.get("latest"),
            deliver=roles.get("deliver"),
        )

        # batch_size may come from the config or be supplied by the caller (e.g. a
        # training script that sources it from its own training config).
        batch_size = overrides.pop("batch_size", None)
        if batch_size is None:
            batch_size = od.get("batch_size")
        if batch_size is None:
            raise ValueError(
                "batch_size must be set in `online_data.batch_size` or passed to "
                "from_config(..., batch_size=...)"
            )

        kwargs = dict(
            base_settings=base_settings,
            randomization=od.get("randomization"),
            num_producers=int(od.get("num_producers", 1)),
            gpu_ids=od.get("gpu_ids", [0]),
            base_seed=int(od.get("base_seed", 0)),
            bus_maxsize=int(od.get("bus_maxsize", 256)),
            ring_caps=od.get("ring_caps"),
            log_dir=log_dir,
            start=start,
        )
        kwargs.update(overrides)
        return cls(schema, int(batch_size), **kwargs)

    def start(self) -> None:
        """Spawn one producer process per spec."""
        if self._procs:
            raise RuntimeError("producers already started")
        if not self._specs:
            raise ValueError("no producer specs to start")

        if self._log_dir is not None:
            _write_run_setup(
                self._log_dir,
                {
                    "batch_size": self.batch_size,
                    "num_producers": len(self._specs),
                    "sensors": self.schema.deliver_uuids(),
                    "producers": [
                        {
                            "worker_id": i,
                            "spec_id": s.spec_id,
                            "gpu_id": s.gpu_id,
                            "seed": s.seed,
                        }
                        for i, s in enumerate(self._specs)
                    ],
                },
            )

        for worker_id, spec in enumerate(self._specs):
            proc = self._ctx.Process(
                target=_run_producer_process,
                args=(
                    self.bus,
                    self._stop,
                    self.schema,
                    spec,
                    worker_id,
                    self._log_dir,
                ),
                daemon=True,
            )
            proc.start()
            self._procs.append(proc)
        logger.info(
            "started %d producer process(es): pids=%s",
            len(self._procs),
            [p.pid for p in self._procs],
        )

    def _all_producers_dead(self) -> bool:
        return bool(self._procs) and all(not p.is_alive() for p in self._procs)

    def __iter__(self):
        """Yield batches forever.

        Watchdog: if no sample arrives for ``stall_warn_s`` while producers are still
        alive, log a one-time warning.
        """
        last_sample_t = time.monotonic()
        stall_warned = False
        while True:
            try:
                sample = self.bus.get(timeout=self._get_timeout)
            except queue.Empty:
                if self._all_producers_dead():
                    logger.error("all producer processes died; stopping iteration")
                    return
                if (
                    self._stall_warn_s
                    and not stall_warned
                    and time.monotonic() - last_sample_t > self._stall_warn_s
                ):
                    logger.warning(
                        "no sample in %.0fs but %d producer(s) still alive — still waiting...",
                        self._stall_warn_s,
                        sum(p.is_alive() for p in self._procs),
                    )
                    stall_warned = True
                continue
            last_sample_t = time.monotonic()
            stall_warned = False
            batch = self.batcher.add(sample)
            if batch is not None:
                yield batch

    def close(self) -> None:
        """Stop all producers and release the bus (idempotent).

        Producers are force-terminated because ``sim.run()`` is uninterruptible
        mid-episode (it cannot poll ``stop_event``).
        """
        self._stop.set()
        for proc in self._procs:
            proc.terminate()
        for proc in self._procs:
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=2.0)
        if self._procs:
            logger.info("terminated %d producer process(es)", len(self._procs))
        self._procs = []
        self.bus.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
