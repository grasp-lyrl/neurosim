"""SimulatorWorker: one producer process that emits time-aligned samples.

**Scaffold (implemented in PR2).** Each worker owns a ``RandomizedSimulator`` on
an assigned GPU, flies a (re-randomized) trajectory per episode via
``sim.run(callback=...)``, and runs the **anchor-driven assembler** (plan §6.1):
stream sensors accumulate into bounded (drop-oldest) packets between anchor ticks;
on each anchor tick it assembles a :class:`~neurosim.online_data.sample.TimeAlignedSample`
of OWNED host memory (copied out of reused sim buffers) and pushes it to the bus.
"""

from neurosim.online_data.schema import SampleSchema


class SimulatorWorker:
    """Producer process. Interface placeholder; see module docstring."""

    def __init__(self, schema: SampleSchema, *args, **kwargs):
        raise NotImplementedError("SimulatorWorker is implemented in PR2.")
