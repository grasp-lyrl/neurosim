"""Unit tests for SampleBus (PR3). In-process mp.Queue — no subprocess.

Verifies the wire unit survives the (pickling) queue, FIFO order, bounded
backpressure, and policy guarding.
"""

import queue

import numpy as np
import pytest

from neurosim.online_data import (
    SampleMeta,
    TimeAlignedSample,
    SampleBus,
    RoutingPolicy,
)


def _sample(uid):
    meta = SampleMeta(
        worker_id=0,
        spec_id=0,
        scene="s",
        seed=0,
        t_us=uid * 1000,
        window_us=1000,
        anchor_uuids=("depth_1",),
        episode_id=SampleMeta.make_episode_id(0, 0),
        step_idx=uid,
        is_first=uid == 0,
        is_last=False,
        sample_uid=uid,
    )
    return TimeAlignedSample(
        meta, {"depth_1": np.full((2, 2), uid, np.float32), "event_1": {"t": np.arange(uid, dtype=np.uint64)}}
    )


def test_bus_roundtrip_and_fifo():
    bus = SampleBus(maxsize=8)
    try:
        for i in range(3):
            bus.put(_sample(i))
        got = [bus.get(timeout=2.0) for _ in range(3)]
        assert [s.meta.sample_uid for s in got] == [0, 1, 2]
        assert np.all(got[2].sensors["depth_1"] == 2)
        assert got[2].sensors["event_1"]["t"].tolist() == [0, 1]
    finally:
        bus.close()


def test_bus_backpressure_raises_full_on_timeout():
    bus = SampleBus(maxsize=1)
    try:
        bus.put(_sample(0))
        with pytest.raises(queue.Full):
            bus.put(_sample(1), timeout=0.1)
    finally:
        bus.close()


def test_bus_empty_raises_on_timeout():
    bus = SampleBus(maxsize=1)
    try:
        with pytest.raises(queue.Empty):
            bus.get(timeout=0.1)
    finally:
        bus.close()


def test_bus_rejects_unimplemented_policy():
    with pytest.raises(NotImplementedError):
        SampleBus(policy=RoutingPolicy.BY_EPISODE)
