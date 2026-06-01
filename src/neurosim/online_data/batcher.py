"""Batchers: turn a stream of TimeAlignedSamples into batched tensors.

**Scaffold (implemented in PR3+).** Dispatch is by sensor *kind* via
:data:`neurosim.online_data.schema.BATCH_STRATEGY_FOR_KIND`:

* ``stack``         — frames/vectors stacked to ``(B, …)``.
* ``concat_counts`` — event/vector streams concatenated with a ``counts`` vector
  (ports the preallocated-buffer logic from the archived ``dataloader.py``).

``ShuffledBatcher`` (v1) builds batches in arrival order. ``SequenceBatcher``
(PR7) maintains B persistent lanes and emits ``(B, L)`` windows + reset masks for
truncated BPTT.
"""

from abc import ABC, abstractmethod

from neurosim.online_data.schema import SampleSchema


class Batcher(ABC):
    """Assembles samples into batches according to a :class:`SampleSchema`."""

    def __init__(self, schema: SampleSchema, batch_size: int):
        self.schema = schema
        self.batch_size = batch_size

    @abstractmethod
    def add(self, sample) -> dict | None:
        """Add one sample; return a finished batch dict or ``None``."""
        raise NotImplementedError


class ShuffledBatcher(Batcher):
    """Arrival-order feed-forward batcher (v1). Implemented in PR3."""

    def add(self, sample) -> dict | None:
        raise NotImplementedError("ShuffledBatcher is implemented in PR3.")
