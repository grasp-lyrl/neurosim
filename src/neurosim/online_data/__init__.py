"""Online data-generation pipeline: many GPU-parallel simulators → DDP training.

A torch-``DataLoader``-like façade fed by heterogeneous, time-aligned simulator
producers. See ``scaling-online-depth-training-plan.md`` for the full design.

Implemented so far:
* PR1 — the wire unit (:class:`TimeAlignedSample`, :class:`SampleMeta`) and the
  sensor schema (:class:`SampleSchema`, :class:`SensorKind`, :class:`SensorRole`,
  :class:`SensorSpec`).
* PR2 — anchor-driven assembly (:class:`AnchorAssembler`, :class:`StreamAccumulator`)
  and the producer (:class:`SimulatorWorker`).
* PR3 — transport (:class:`SampleBus`), batching (:class:`ShuffledBatcher`,
  :class:`Batch`, :class:`BatchMeta`, :func:`shift_events`), and the
  :class:`OnlineDataLoader` façade (single producer → single consumer).

The multi-producer config and DDP sharding land in PR4+.
"""

from .sample import SampleMeta, TimeAlignedSample, assert_owned_array
from .schema import (
    SensorKind,
    SensorRole,
    SensorSpec,
    SampleSchema,
    infer_kind,
    SENSOR_TYPE_TO_KIND,
    BATCH_STRATEGY_FOR_KIND,
)
from .assembler import AnchorAssembler, StreamAccumulator
from .sim_worker import SimulatorWorker, build_randomized_sim
from .recorder import record_episodes, record_dataset
from .batcher import (
    ShuffledBatcher,
    Batch,
    BatchMeta,
    FrameBuffer,
    shift_events,
)
from .bus import SampleBus, RoutingPolicy
from .loader import OnlineDataLoader, ProducerSpec, build_producer_specs

__all__ = [
    # Sample (wire unit)
    "SampleMeta",
    "TimeAlignedSample",
    "assert_owned_array",
    # Schema
    "SensorKind",
    "SensorRole",
    "SensorSpec",
    "SampleSchema",
    "infer_kind",
    "SENSOR_TYPE_TO_KIND",
    "BATCH_STRATEGY_FOR_KIND",
    # Producer (PR2)
    "AnchorAssembler",
    "StreamAccumulator",
    "SimulatorWorker",
    "build_randomized_sim",
    # Offline H5 recorder
    "record_episodes",
    "record_dataset",
    # Transport + batching (PR3)
    "SampleBus",
    "RoutingPolicy",
    "ShuffledBatcher",
    "Batch",
    "BatchMeta",
    "FrameBuffer",
    "shift_events",
    # Façade (PR3) + multi-producer (PR4)
    "OnlineDataLoader",
    "ProducerSpec",
    "build_producer_specs",
]
