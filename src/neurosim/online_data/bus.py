"""SampleBus: transport from simulator producers to per-rank batch builders.

Single bounded ``mp.Queue`` shared by producers (put) and consumers (get).
With one consumer it is a plain FIFO; with N consumers each ``get`` pops a
distinct sample, so the stream is sharded disjointly across consumers
(work-stealing) and batches mix producers — that is the ``ROUND_ROBIN`` policy.

The bound provides **backpressure**: when the queue is full, producers block on
``put`` so a fast simulator throttles instead of exhausting host memory. A
``by_episode`` policy (episode-affinity routing for the recurrent batcher) and a
shared-memory / Cortex-backed implementation can slot in behind this interface later.
"""

import logging
import multiprocessing as mp
from enum import Enum

from neurosim.online_data.sample import TimeAlignedSample

logger = logging.getLogger(__name__)


class RoutingPolicy(str, Enum):
    """How samples are routed from the shared producer stream to consumers."""

    ROUND_ROBIN = "round_robin"  # shared FIFO; consumers pull (v1)
    BY_EPISODE = "by_episode"  # recurrent: keep each episode on one lane (later)


class SampleBus:
    """A bounded, process-shared queue of :class:`TimeAlignedSample`s.

    Args:
        maxsize: Queue capacity (backpressure bound). ``0`` means unbounded.
        ctx: Multiprocessing context (or its name, e.g. ``"spawn"``). Defaults to
            the default context. Must match the context used to spawn producers.
        policy: Routing policy. Only ``ROUND_ROBIN`` (shared FIFO) is implemented.
    """

    def __init__(
        self,
        maxsize: int = 256,
        ctx=None,
        policy: RoutingPolicy = RoutingPolicy.ROUND_ROBIN,
    ):
        policy = RoutingPolicy(policy)
        if policy != RoutingPolicy.ROUND_ROBIN:
            raise NotImplementedError(
                f"routing policy {policy!r} not implemented; only "
                f"{RoutingPolicy.ROUND_ROBIN!r} is available right now."
            )
        if isinstance(ctx, str):
            ctx = mp.get_context(ctx)
        self.policy = policy
        self.maxsize = maxsize
        # Only the queue is retained (and pickled when passed to a child process);
        # the context is used solely to construct the queue from the right method.
        self._queue = (ctx or mp).Queue(maxsize)

    def put(self, sample: TimeAlignedSample, timeout: float | None = None) -> None:
        """Put a sample (blocks when full; raises ``queue.Full`` on timeout)."""
        self._queue.put(sample, timeout=timeout)

    def get(self, timeout: float | None = None) -> TimeAlignedSample:
        """Get the next sample (blocks when empty; raises ``queue.Empty`` on timeout)."""
        return self._queue.get(timeout=timeout)

    def qsize(self) -> int:
        try:
            return self._queue.qsize()
        except NotImplementedError:  # qsize unavailable on some platforms (macOS)
            return -1

    def close(self) -> None:
        """Close the underlying queue and stop its feeder thread."""
        try:
            self._queue.close()
            self._queue.join_thread()
        except Exception:  # best-effort teardown
            pass
