"""OnlineDataLoader: the torch-``DataLoader``-like façade.

**Scaffold (implemented in PR3/PR4).** Construct with a spec, iterate batches,
close — all producer / GPU / bus machinery stays internal::

    loader = OnlineDataLoader(spec, batch_size=32,
                              sensors=["event_camera_1", "depth_camera_1"])
    for batch in loader:          # dict[uuid -> tensor] (+ batch.meta)
        ...
    loader.close()

Internally: spawns producers (``SimulatorWorker`` per GPU slot), wires the
``SampleBus``, runs a per-rank builder (``Batcher``), and yields batches moved to
this rank's GPU. DDP sharding is inferred from ``torch.distributed`` env.
"""


class OnlineDataLoader:
    """Façade over producers + bus + batcher. Implemented in PR3/PR4."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("OnlineDataLoader is implemented in PR3/PR4.")
