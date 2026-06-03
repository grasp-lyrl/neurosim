"""OnlineDataLoader: the torch-``DataLoader``-like façade.

Construct with a schema + producer settings, iterate batches, close::

    loader = OnlineDataLoader(schema, batch_size=32, base_settings=settings)
    for batch in loader:          # Batch: dict[uuid -> payload] (+ batch.meta)
        events = batch["event_camera_1"]   # (counts, events)
        depth  = batch["depth_camera_1"]   # (B, H, W)
    loader.close()

Topology (v1, single producer → single consumer): a **producer process** runs a
:class:`~neurosim.online_data.sim_worker.SimulatorWorker`, pushing time-aligned
samples to a bounded :class:`~neurosim.online_data.bus.SampleBus`; the consumer
(this process) pops them and builds batches inline with a
:class:`~neurosim.online_data.batcher.ShuffledBatcher`. The bounded bus decouples
sim from training (producer runs ahead) and bounds memory (backpressure).

Producers use ``spawn`` (Habitat/EGL + CUDA must not fork). The Habitat-bearing
work lives in the child via the lazy import in ``SimulatorWorker``. Constructing
with ``base_settings=None``/``start=False`` skips the producer entirely so the
consumer path can be driven by feeding ``loader.bus`` directly (used in tests).
"""

import queue
import logging

from neurosim.online_data.bus import SampleBus
from neurosim.online_data.batcher import ShuffledBatcher, EventNorm, Batch
from neurosim.online_data.schema import SampleSchema, SensorKind

logger = logging.getLogger(__name__)


def _run_producer_process(
    bus: SampleBus,
    stop_event,
    schema: SampleSchema,
    base_settings: dict,
    randomization: dict | None,
    worker_kwargs: dict,
) -> None:
    """Producer-process entry point: run a worker, push samples until stopped.

    Module-level (picklable for ``spawn``). Episodes run to completion; shutdown
    is via process termination from :meth:`OnlineDataLoader.close` (the blocking
    ``sim.run`` inside an episode does not poll ``stop_event``).
    """
    from neurosim.online_data.sim_worker import SimulatorWorker

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
        base_settings=base_settings,
        randomization=randomization,
        emit_fn=emit,
        **(worker_kwargs or {}),
    )
    try:
        while not stop_event.is_set():
            worker.run_episode()
    except Exception:  # pragma: no cover - surfaced via producer death in __iter__
        logger.exception("producer process crashed")
    finally:
        worker.close()


class OnlineDataLoader:
    """Façade over producer process + bus + batcher.

    Args:
        schema: Resolved delivery schema.
        batch_size: Samples per batch.
        base_settings: Simulator settings for the producer (``None`` => no
            producer is started; feed ``self.bus`` manually).
        randomization: Domain-randomization dict passed to the worker.
        worker_kwargs: Extra kwargs for :class:`SimulatorWorker` (worker_id,
            spec_id, gpu_id, seed, ring_caps, strict_owned).
        bus_maxsize: Bus capacity (backpressure bound).
        event_time_window_us: Normalization window for event streams.
        normalize_events: Whether to normalize event coordinates/time.
        mp_context: Multiprocessing start method (``"spawn"`` for CUDA/Habitat).
        get_timeout: Poll interval (s) for the consumer loop / producer-death check.
        start: Start the producer immediately (requires ``base_settings``).
    """

    def __init__(
        self,
        schema: SampleSchema,
        batch_size: int,
        *,
        base_settings: dict | None = None,
        randomization: dict | None = None,
        worker_kwargs: dict | None = None,
        bus_maxsize: int = 256,
        event_time_window_us: float = 50_000,
        normalize_events: bool = True,
        mp_context: str = "spawn",
        get_timeout: float = 1.0,
        start: bool = True,
    ):
        import multiprocessing as mp

        self.schema = schema
        self.batch_size = batch_size
        self._base_settings = base_settings
        self._randomization = randomization
        self._worker_kwargs = worker_kwargs or {}
        self._get_timeout = get_timeout

        event_norm = {
            uuid: EventNorm(
                width=int(schema.specs[uuid].extras["width"]),
                height=int(schema.specs[uuid].extras["height"]),
                time_window_us=event_time_window_us,
                enabled=normalize_events,
            )
            for uuid in schema.deliver_uuids()
            if schema.kind_of(uuid) is SensorKind.EVENT_STREAM
        }
        self.batcher = ShuffledBatcher(schema, batch_size, event_norm=event_norm)

        self._ctx = mp.get_context(mp_context)
        self.bus = SampleBus(maxsize=bus_maxsize, ctx=self._ctx)
        self._stop = self._ctx.Event()
        self._proc = None

        if start and base_settings is not None:
            self.start()

    def start(self) -> None:
        """Spawn the producer process."""
        if self._proc is not None:
            raise RuntimeError("producer already started")
        if self._base_settings is None:
            raise ValueError("cannot start producer without base_settings")
        self._proc = self._ctx.Process(
            target=_run_producer_process,
            args=(
                self.bus,
                self._stop,
                self.schema,
                self._base_settings,
                self._randomization,
                self._worker_kwargs,
            ),
            daemon=True,
        )
        self._proc.start()
        logger.info("started producer process pid=%s", self._proc.pid)

    def __iter__(self):
        """Yield batches forever (use ``itertools.islice`` to bound)."""
        while True:
            try:
                sample = self.bus.get(timeout=self._get_timeout)
            except queue.Empty:
                if self._proc is not None and not self._proc.is_alive():
                    logger.error("producer process died; stopping iteration")
                    return
                continue
            batch = self.batcher.add(sample)
            if batch is not None:
                yield batch

    def close(self) -> None:
        """Stop the producer and release the bus (idempotent)."""
        self._stop.set()
        proc = self._proc
        if proc is not None:
            proc.terminate()
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=2.0)
            self._proc = None
            logger.info("producer process terminated")
        self.bus.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
