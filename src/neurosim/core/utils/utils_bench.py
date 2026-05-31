"""Lightweight, low-overhead profiling for the simulation loop.

Design notes
------------
* **Correct GPU attribution.** Habitat keeps renderings on the GPU
  (``gpu2gpu_transfer=True``), so wall-clock (`perf_counter`) timing around a
  render only measures kernel *launch* time, not execution. GPU-heavy sections
  are therefore timed with ``torch.cuda.Event`` pairs and resolved with a single
  ``torch.cuda.synchronize`` per step (only when profiling is ON). CPU-bound
  sections (dynamics, control, logging) use ``perf_counter``.
* **Hierarchical sections.** Sections auto-nest: one opened while another is
  active is recorded under the dotted path ``parent.child``, so call sites pass
  only a relative leaf name (see :meth:`Profiler.section`). The first path
  component is the top-level, mutually-exclusive bucket (``dynamics_step``,
  ``render_sensors``, ...); deeper components decompose it — e.g.
  ``render_sensors.event_camera_1.event_kernel``. Summaries/plots render the
  dotted paths as a tree.
"""

import json
import time
import logging
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProfileSectionStat:
    """Accumulated timing statistics for a single profiled section.

    All times are in milliseconds. Optionally retains a (capped) per-call
    timeline of samples so that time-series and histogram plots can be produced
    after a run.

    Attributes:
        name: Section name (dotted names denote nested sub-sections).
        total_calls: Number of recorded calls.
        total_time_ms: Sum of all recorded times in milliseconds.
        min_time_ms: Fastest single call in milliseconds.
        max_time_ms: Slowest single call in milliseconds.
        samples_ms: Per-call timeline (only populated when ``store_timeline``).
    """

    name: str
    total_calls: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float("inf")
    max_time_ms: float = 0.0
    samples_ms: list = field(default_factory=list, repr=False)
    store_timeline: bool = True
    max_samples: int = 500_000

    @property
    def avg_time_ms(self) -> float:
        """Average time per call in milliseconds."""
        return self.total_time_ms / self.total_calls if self.total_calls else 0.0

    def percentile(self, q: float) -> float:
        """Return the ``q``-th percentile of recorded times (0 if no samples)."""
        return float(np.percentile(self.samples_ms, q)) if self.samples_ms else 0.0

    def record(self, elapsed_ms: float) -> None:
        """Record a single timing measurement in milliseconds."""
        self.total_calls += 1
        self.total_time_ms += elapsed_ms
        self.min_time_ms = min(self.min_time_ms, elapsed_ms)
        self.max_time_ms = max(self.max_time_ms, elapsed_ms)
        if self.store_timeline and len(self.samples_ms) < self.max_samples:
            self.samples_ms.append(elapsed_ms)

    def to_dict(self, include_timeline: bool = False) -> dict:
        """Serialise statistics to a plain dict (optionally with the timeline)."""
        d = {
            "name": self.name,
            "total_calls": self.total_calls,
            "total_time_ms": self.total_time_ms,
            "avg_time_ms": self.avg_time_ms,
            "min_time_ms": self.min_time_ms if self.total_calls else 0.0,
            "max_time_ms": self.max_time_ms,
            "p50_ms": self.percentile(50),
            "p99_ms": self.percentile(99),
        }
        if include_timeline:
            d["samples_ms"] = list(self.samples_ms)
        return d


class _NullCtx:
    """Shared no-op context manager returned when profiling is disabled."""

    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_NULL_CTX = _NullCtx()


class Profiler:
    """Section-based profiler with mixed CPU (perf_counter) / GPU (CUDA event)
    timing.

    When ``enabled`` is False every hook is a cheap no-op, so the same
    instrumented code path can be left in place in production with negligible
    cost.

    Args:
        enabled: Master switch. When False, all methods are no-ops.
        device: Torch device used for ``synchronize`` (e.g. ``"cuda:0"``).
        use_cuda_events: Time GPU sections with CUDA events. When False (or when
            CUDA is unavailable) GPU sections fall back to ``perf_counter``.
        store_timeline: Retain per-call samples for time-series/histogram plots.
        max_samples: Per-section cap on retained samples (bounds memory).
    """

    def __init__(
        self,
        enabled: bool = False,
        device: str | None = None,
        use_cuda_events: bool = True,
        store_timeline: bool = True,
        max_samples: int = 500_000,
    ):
        self.enabled = bool(enabled)
        self._device = device
        self._store_timeline = store_timeline
        self._max_samples = max_samples
        self._sections: dict[str, ProfileSectionStat] = {}
        self._n_steps = 0

        # GPU timing is only meaningful when enabled + torch CUDA is present.
        self._gpu = bool(
            self.enabled
            and use_cuda_events
            and torch.cuda.is_available()
            and (device is None or str(device).startswith("cuda"))
        )
        # Deferred CUDA-event pairs resolved at each step() boundary.
        self._pending: list[
            tuple[ProfileSectionStat, "torch.cuda.Event", "torch.cuda.Event"]
        ] = []
        self._event_pool: list["torch.cuda.Event"] = []
        self._pc = time.perf_counter  # local bind
        # Active section path, used to auto-nest sections hierarchically.
        self._stack: list[str] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _stat(self, name: str) -> ProfileSectionStat:
        stat = self._sections.get(name)
        if stat is None:
            stat = ProfileSectionStat(
                name=name,
                store_timeline=self._store_timeline,
                max_samples=self._max_samples,
            )
            self._sections[name] = stat
        return stat

    def _get_event(self) -> "torch.cuda.Event":
        if self._event_pool:
            return self._event_pool.pop()
        return torch.cuda.Event(enable_timing=True)

    @contextmanager
    def _timed(self, stat: ProfileSectionStat, leaf: str, use_gpu: bool):
        # Push the relative leaf so any nested section() composes its full path.
        self._stack.append(leaf)
        try:
            if use_gpu:
                start = self._get_event()
                start.record()
                try:
                    yield
                finally:
                    stop = self._get_event()
                    stop.record()
                    self._pending.append((stat, start, stop))
            else:
                t0 = self._pc()
                try:
                    yield
                finally:
                    stat.record((self._pc() - t0) * 1e3)
        finally:
            self._stack.pop()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def section(self, name: str, gpu: bool = False):
        """Return a context manager that times the wrapped block.

        Section names compose **hierarchically**: a section opened while another
        is active is recorded under the dotted path ``parent.child``. Call sites
        therefore pass only a *relative* leaf name and never need to know their
        parent — e.g. the loop opens ``render_sensors`` then ``<uuid>``, and the
        backend opens ``event_kernel``, yielding
        ``render_sensors.<uuid>.event_kernel`` automatically. The first path
        component is the top-level (mutually-exclusive) bucket.

        Args:
            name: Relative section name (no dots needed; nesting is automatic).
            gpu: If True and CUDA timing is available, time with CUDA events
                (resolved on the next :meth:`step`). Otherwise use perf_counter.
        """
        if not self.enabled:
            return _NULL_CTX
        full = ".".join(self._stack) + "." + name if self._stack else name
        return self._timed(self._stat(full), name, gpu and self._gpu)

    def step(self) -> None:
        """Mark a step boundary: synchronise once and resolve pending GPU timings.

        No-op when disabled. This is the only place a ``cuda.synchronize`` is
        issued, so GPU timing perturbs the loop at most once per step and only
        while profiling is active.
        """
        if not self.enabled:
            return
        self._n_steps += 1
        if self._pending:
            torch.cuda.synchronize(self._device)
            for stat, start, stop in self._pending:
                stat.record(start.elapsed_time(stop))
                self._event_pool.append(start)
                self._event_pool.append(stop)
            self._pending.clear()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def summary(self, include_timeline: bool = False) -> dict:
        """Return a serialisable summary of all sections."""
        return {
            "n_steps": self._n_steps,
            "device": str(self._device),
            "gpu_timing": self._gpu,
            "sections": {
                name: stat.to_dict(include_timeline=include_timeline)
                for name, stat in self._sections.items()
            },
        }

    def save(self, path: str | Path, include_timeline: bool = True) -> Path:
        """Write the summary (and, by default, the per-call timeline) to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.summary(include_timeline=include_timeline), f, indent=2)
        logger.info(f"Profiling data written to {path}")
        return path

    def _child_sections(self, parent: str) -> list[tuple[str, ProfileSectionStat]]:
        """Direct children of ``parent`` (``""`` = top-level), sorted by total."""
        prefix = parent + "." if parent else ""
        depth = prefix.count(".")
        kids = [
            (n, s)
            for n, s in self._sections.items()
            if n.startswith(prefix) and n.count(".") == depth
        ]
        return sorted(kids, key=lambda kv: kv[1].total_time_ms, reverse=True)

    def log_summary(self) -> None:
        """Log a human-readable, hierarchical breakdown tree.

        Top-level (non-dotted) sections are mutually exclusive; their sum is the
        per-step total used for percentages. Dotted sections decompose their
        parent (and overlap it) — they are shown indented beneath it, so e.g.
        ``render_sensors`` expands into its per-uuid children and, for an event
        camera, into the ``habitat_render`` / ``event_kernel`` / ... kernels.
        """
        if not self._sections:
            logger.info("Profiler: no data collected.")
            return

        total_ms = (
            sum(s.total_time_ms for n, s in self._sections.items() if "." not in n)
            or 1.0
        )
        n = max(self._n_steps, 1)

        def _walk(parent: str, depth: int) -> None:
            for name, stat in self._child_sections(parent):
                leaf = name.rsplit(".", 1)[-1]
                label = "  " * depth + ("└ " if depth else "") + leaf
                pct = 100.0 * stat.total_time_ms / total_ms
                logger.info(
                    f"  {label:<30}{stat.total_time_ms:>11.1f}{pct:>6.1f}%"
                    f"{stat.avg_time_ms:>10.3f}{stat.percentile(99):>9.3f}"
                    f"{stat.total_calls:>9}"
                )
                _walk(name, depth + 1)

        logger.info(
            "══════════════════════════ Profiling breakdown ══════════════════════════"
        )
        logger.info(f"  steps={self._n_steps}  gpu_timing={self._gpu}")
        logger.info(
            f"  {'section':<30}{'total_ms':>11}{'%':>7}{'mean_ms':>10}"
            f"{'p99_ms':>9}{'calls':>9}"
        )
        logger.info("  " + "─" * 74)
        _walk("", 0)
        logger.info("  " + "─" * 74)
        logger.info(
            f"  {'TOTAL (top-level)':<30}{total_ms:>11.1f}{100.0:>6.1f}%"
            f"{total_ms / n:>10.3f}"
        )
        logger.info(
            "══════════════════════════════════════════════════════════════════════════"
        )
