import numpy as np
from dataclasses import dataclass, field


@dataclass
class RenderEventsBenchmark:
    """Benchmark statistics for render_events calls.

    This class stores timing information collected using CUDA events,
    which is the recommended way to accurately measure GPU kernel execution time.

    Attributes:
        total_calls: Total number of render_events calls.
        total_time_ms: Total elapsed time in milliseconds.
        min_time_ms: Minimum time for a single call in milliseconds.
        max_time_ms: Maximum time for a single call in milliseconds.
        times_ms: List of individual call times in milliseconds.
    """

    total_calls: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float("inf")
    max_time_ms: float = 0.0
    times_ms: list = field(default_factory=list)

    @property
    def avg_time_ms(self) -> float:
        """Average time per call in milliseconds."""
        return self.total_time_ms / self.total_calls if self.total_calls > 0 else 0.0

    @property
    def std_dev_ms(self) -> float:
        """Standard deviation of call times in milliseconds."""
        if self.total_calls < 2:
            return 0.0
        return np.std(self.times_ms)

    def record(self, elapsed_ms: float) -> None:
        """Record a new timing measurement.

        Args:
            elapsed_ms: Elapsed time in milliseconds.
        """
        self.total_calls += 1
        self.total_time_ms += elapsed_ms
        self.min_time_ms = min(self.min_time_ms, elapsed_ms)
        self.max_time_ms = max(self.max_time_ms, elapsed_ms)
        self.times_ms.append(elapsed_ms)

    def reset(self) -> None:
        """Reset all benchmark statistics."""
        self.total_calls = 0
        self.total_time_ms = 0.0
        self.min_time_ms = float("inf")
        self.max_time_ms = 0.0
        self.times_ms.clear()

    def __repr__(self) -> str:
        if self.total_calls == 0:
            return "RenderEventsBenchmark(no data)"
        return (
            f"RenderEventsBenchmark(\n"
            f"  total_calls={self.total_calls},\n"
            f"  avg_time_ms={self.avg_time_ms:.3f},\n"
            f"  std_dev_ms={self.std_dev_ms:.3f},\n"
            f"  min_time_ms={self.min_time_ms:.3f},\n"
            f"  max_time_ms={self.max_time_ms:.3f},\n"
            f"  total_time_ms={self.total_time_ms:.3f}\n"
            f")"
        )

    def to_dict(self) -> dict:
        """Convert benchmark statistics to a dictionary."""
        return {
            "total_calls": self.total_calls,
            "avg_time_ms": self.avg_time_ms,
            "std_dev_ms": self.std_dev_ms,
            "min_time_ms": self.min_time_ms,
            "max_time_ms": self.max_time_ms,
            "total_time_ms": self.total_time_ms,
        }
