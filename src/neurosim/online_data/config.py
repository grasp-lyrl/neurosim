"""DataGenConfig: the spec for an online data-generation run.

**Scaffold (implemented in PR3/PR4).** Reuses the RL env-style settings
(scenes / sensors / simulator / dynamics / domain_randomization handled by
``RandomizedSimulator`` + ``DomainRandomizationConfig``) and adds:

* a ``controller`` + ``trajectory`` block (data-gen is open-loop: each producer
  flies a precomputed trajectory via ``sim.run`` rather than ``sim.step``), and
* a ``data:`` block — the only genuinely new schema:

  .. code-block:: yaml

      data:
        anchor:  [depth_camera_1]      # SensorRole.ANCHOR (boundary)
        stream:  [event_camera_1]      # SensorRole.STREAM (packeted, ring-capped)
        latest:  []                    # SensorRole.LATEST (held snapshot)
        deliver: [event_camera_1, depth_camera_1]
        batch_size: 32
        prefetch: 8
        ring_caps: {event_camera_1: 4000000}
        num_producers: 8
        gpu_map: {producers: [0,0,0,0,0,0,0,0], trainer: [1]}

The old ``sims/online_dataloader`` ``DatasetConfig`` (static scenes×trajs×seeds
cross-product) is retired in favour of per-episode randomization (plan §14).
"""


class DataGenConfig:
    """Online data-generation spec. Interface placeholder; see module docstring."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("DataGenConfig is implemented in PR3/PR4.")
