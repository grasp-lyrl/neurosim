"""Unit tests for the OnlineDataLoader consumer path (PR3). No Habitat / producer.

The loader is built with ``start=False`` (no producer process); samples are fed
into ``loader.bus`` directly so the consumer + batcher + façade are exercised
without Habitat. The full producer-process path is covered by the gated Habitat
test.
"""

import itertools

import numpy as np

from neurosim.online_data import (
    SampleMeta,
    TimeAlignedSample,
    SampleSchema,
    OnlineDataLoader,
)

W = H = 4
SENSOR_CONFIGS = {
    "depth_1": {"type": "depth", "height": H, "width": W},
    "event_1": {"type": "event", "height": H, "width": W},
}


def _schema():
    return SampleSchema.from_sensor_configs(
        SENSOR_CONFIGS, anchor=["depth_1"], stream=["event_1"]
    )


def _sample(uid):
    n = uid + 1
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
    sensors = {
        "depth_1": np.full((H, W), float(uid), np.float32),
        "event_1": {
            "x": np.arange(n, dtype=np.uint16),
            "y": np.zeros(n, np.uint16),
            "t": np.arange(n, dtype=np.uint64),
            "p": np.ones(n, np.uint8),
        },
    }
    return TimeAlignedSample(meta, sensors)


def test_loader_consumer_builds_batches():
    loader = OnlineDataLoader(
        _schema(), batch_size=2, base_settings=None, start=False, get_timeout=0.2
    )
    try:
        for i in range(4):
            loader.bus.put(_sample(i))
        batches = list(itertools.islice(loader, 2))
        assert len(batches) == 2
        b0 = batches[0]
        assert b0["depth_1"].shape == (2, H, W)
        counts, events = b0["event_1"]
        assert counts.tolist() == [1, 2]  # samples 0 and 1
        assert events.shape == (3, 4)
        assert b0.meta.sample_uid.tolist() == [0, 1]
        assert batches[1].meta.sample_uid.tolist() == [2, 3]
    finally:
        loader.close()


def test_loader_close_idempotent_without_producer():
    loader = OnlineDataLoader(_schema(), batch_size=2, base_settings=None, start=False)
    loader.close()
    loader.close()  # no raise


def test_loader_context_manager():
    with OnlineDataLoader(
        _schema(), batch_size=1, base_settings=None, start=False, get_timeout=0.2
    ) as loader:
        loader.bus.put(_sample(0))
        batch = next(iter(loader))
        assert batch["depth_1"].shape == (1, H, W)
