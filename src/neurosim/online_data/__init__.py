"""Online data-generation pipeline: many GPU-parallel simulators → DDP training.

A torch-``DataLoader``-like façade fed by heterogeneous, time-aligned simulator
producers. See ``scaling-online-depth-training-plan.md`` for the full design.
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
    # Producer
    "AnchorAssembler",
    "StreamAccumulator",
    "SimulatorWorker",
    "build_randomized_sim",
    # Offline H5 recorder
    "record_episodes",
    "record_dataset",
    # Transport + batching
    "SampleBus",
    "RoutingPolicy",
    "ShuffledBatcher",
    "Batch",
    "BatchMeta",
    "FrameBuffer",
    "shift_events",
    # Façade + multi-producer
    "OnlineDataLoader",
    "ProducerSpec",
    "build_producer_specs",
]
