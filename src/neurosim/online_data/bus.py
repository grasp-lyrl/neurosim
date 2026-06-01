"""SampleBus: transport from simulator producers to per-rank batch builders.

**Scaffold (implemented in PR3+).** v1 will use a bounded ``mp.Queue`` with a
``round_robin`` routing policy (shuffled training). A ``by_episode`` policy
(episode-affinity routing that preserves per-episode order) is added with the
recurrent batcher. A shared-memory / Cortex-backed implementation can later slot
in behind this same interface without touching producers or builders.
"""

from enum import Enum


class RoutingPolicy(str, Enum):
    """How samples are routed from the shared producer stream to builders."""

    ROUND_ROBIN = "round_robin"  # shuffled training (v1)
    BY_EPISODE = "by_episode"  # recurrent: keep each episode on one lane (later)


class SampleBus:
    """Producer→builder transport. Interface placeholder; see module docstring."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("SampleBus is implemented in PR3.")
