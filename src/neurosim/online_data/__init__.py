"""Online data-generation pipeline: many GPU-parallel simulators → DDP training.

A torch-``DataLoader``-like façade fed by heterogeneous, time-aligned simulator
producers. See ``scaling-online-depth-training-plan.md`` for the full design.

Implemented so far (PR1): the wire unit (:class:`TimeAlignedSample`,
:class:`SampleMeta`) and the sensor schema (:class:`SampleSchema`,
:class:`SensorKind`, :class:`SensorRole`, :class:`SensorSpec`). The transport,
batcher, simulator worker, config, and loader façade are scaffolded and land in
PR2–PR4.
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
from .bus import RoutingPolicy

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
    # Transport (routing policy enum is stable; SampleBus impl lands in PR3)
    "RoutingPolicy",
]
