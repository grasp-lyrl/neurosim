"""Unit tests for the OnlineDataLoader consumer path (PR3). No Habitat / producer.

The loader is built with ``start=False`` (no producer process); samples are fed
into ``loader.bus`` directly so the consumer + batcher + façade are exercised
without Habitat. The full producer-process path is covered by the gated Habitat
test.
"""

import itertools
import logging
import queue
import time

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


def test_build_producer_specs():
    from neurosim.online_data import build_producer_specs

    specs = build_producer_specs(
        {"k": 1}, num_producers=8, gpu_ids=[0], base_seed=10, randomization={"d": 1}
    )
    assert len(specs) == 8
    assert [s.gpu_id for s in specs] == [0] * 8
    assert [s.seed for s in specs] == list(range(10, 18))
    assert [s.spec_id for s in specs] == list(range(8))
    assert all(s.randomization == {"d": 1} for s in specs)

    # GPU ids cycle across producers.
    multi = build_producer_specs({"k": 1}, num_producers=4, gpu_ids=[0, 1])
    assert [s.gpu_id for s in multi] == [0, 1, 0, 1]


def test_loader_context_manager():
    with OnlineDataLoader(
        _schema(), batch_size=1, base_settings=None, start=False, get_timeout=0.2
    ) as loader:
        loader.bus.put(_sample(0))
        batch = next(iter(loader))
        assert batch["depth_1"].shape == (1, H, W)


# --------------------------------------------------------------------------- #
# from_config: YAML/dict-driven construction (no producers spawned)
# --------------------------------------------------------------------------- #
def _config_dict():
    return {
        "simulator": {"sim_time": 1.0},
        "visual_backend": {
            "sensors": {
                "depth_1": {"type": "depth", "height": H, "width": W},
                "event_1": {"type": "event", "height": H, "width": W},
            }
        },
        "online_data": {
            "batch_size": 3,
            "num_producers": 2,
            "gpu_ids": [0],
            "base_seed": 5,
            "roles": {"anchor": ["depth_1"], "stream": ["event_1"]},
            "randomization": {"resample_every": 2},
        },
    }


def test_from_config_builds_schema_and_specs():
    loader = OnlineDataLoader.from_config(_config_dict(), start=False)
    try:
        assert loader.batch_size == 3
        # roles -> schema
        assert loader.schema.deliver_uuids() == ["depth_1", "event_1"]
        assert loader.schema.role_of("depth_1").value == "anchor"
        assert loader.schema.role_of("event_1").value == "stream"
        # knobs -> producer specs (no processes started)
        assert len(loader._specs) == 2
        assert [s.seed for s in loader._specs] == [5, 6]  # base_seed + i
        assert all(s.randomization == {"resample_every": 2} for s in loader._specs)
        # consumer path still works when fed directly
        for i in range(3):
            loader.bus.put(_sample(i))
        batch = next(iter(loader))
        assert batch["depth_1"].shape == (3, H, W)
    finally:
        loader.close()


def test_from_config_separate_base_settings(tmp_path):
    import yaml

    base = tmp_path / "settings.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "simulator": {"sim_time": 2.0},
                "visual_backend": {
                    "sensors": {"depth_1": {"type": "depth", "height": H, "width": W}}
                },
            }
        )
    )
    cfg = {
        "online_data": {
            "base_settings": str(base),
            "batch_size": 1,
            "roles": {"anchor": ["depth_1"]},
        }
    }
    loader = OnlineDataLoader.from_config(cfg, start=False)
    try:
        assert loader.schema.deliver_uuids() == ["depth_1"]
        assert loader._specs[0].base_settings["simulator"]["sim_time"] == 2.0
    finally:
        loader.close()


class _AliveProc:
    """Stand-in for a live producer process (no real spawn)."""

    def is_alive(self):
        return True


def test_stall_watchdog_warns_once_when_producers_alive_but_silent(caplog):
    loader = OnlineDataLoader(
        _schema(),
        batch_size=1,
        base_settings=None,
        start=False,
        get_timeout=0.0,
        stall_warn_s=0.05,
    )
    loader._procs = [_AliveProc()]  # alive -> not "all dead", so the watchdog path runs

    # Empty a few times (long enough to trip the watchdog), then deliver one sample.
    state = {"n": 0}
    sample = _sample(0)

    def fake_get(timeout=None):
        state["n"] += 1
        if state["n"] <= 3:
            time.sleep(0.03)  # let monotonic advance past stall_warn_s
            raise queue.Empty
        if state["n"] == 4:
            return sample
        raise queue.Empty  # deliver once; stay empty (so close()'s drain terminates)

    loader.bus.get = fake_get
    try:
        with caplog.at_level(logging.WARNING):
            batch = next(iter(loader))
        assert batch["depth_1"].shape == (1, H, W)
        warnings = [r.message for r in caplog.records if "still alive" in r.message]
        assert len(warnings) == 1  # one-time, not spammed
    finally:
        loader._procs = []  # avoid terminate() on the fake proc
        loader.close()


def test_stall_watchdog_disabled_when_zero(caplog):
    loader = OnlineDataLoader(
        _schema(),
        batch_size=1,
        base_settings=None,
        start=False,
        get_timeout=0.0,
        stall_warn_s=0.0,  # disabled
    )
    loader._procs = [_AliveProc()]
    state = {"n": 0}
    sample = _sample(0)

    def fake_get(timeout=None):
        state["n"] += 1
        if state["n"] <= 3:
            time.sleep(0.03)
            raise queue.Empty
        if state["n"] == 4:
            return sample
        raise queue.Empty  # deliver once; stay empty (so close()'s drain terminates)

    loader.bus.get = fake_get
    try:
        with caplog.at_level(logging.WARNING):
            next(iter(loader))
        assert not [r for r in caplog.records if "still alive" in r.message]
    finally:
        loader._procs = []
        loader.close()


def test_from_config_rejects_missing_block_and_anchor():
    import pytest

    with pytest.raises(ValueError, match="online_data"):
        OnlineDataLoader.from_config({"simulator": {}}, start=False)
    with pytest.raises(ValueError, match="anchor"):
        OnlineDataLoader.from_config(
            {"visual_backend": {"sensors": {}}, "online_data": {"batch_size": 1}},
            start=False,
        )
