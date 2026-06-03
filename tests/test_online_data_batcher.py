"""Unit tests for batching (PR3). Pure numpy — no Habitat / mp.

Covers ShuffledBatcher (shapes by kind, event counts, batch.meta), event
normalization parity with the archived `_process_sample`, FrameBuffer, and the
empty-events / partial-batch edge cases.
"""

import numpy as np
import pytest

from neurosim.online_data import (
    SampleMeta,
    TimeAlignedSample,
    SampleSchema,
    ShuffledBatcher,
    EventNorm,
    FrameBuffer,
    normalize_events,
)

W = H = 4
SENSOR_CONFIGS = {
    "depth_1": {"type": "depth", "height": H, "width": W},
    "event_1": {"type": "event", "height": H, "width": W},
}
TIME_WINDOW = 50_000


def _schema():
    return SampleSchema.from_sensor_configs(
        SENSOR_CONFIGS, anchor=["depth_1"], stream=["event_1"]
    )


def _norm(enabled=True):
    return {"event_1": EventNorm(width=W, height=H, time_window_us=TIME_WINDOW, enabled=enabled)}


def _sample(uid, *, depth_fill, xs, ts, t_us=1000, step_idx=0, is_first=False, is_last=False):
    xs = np.asarray(xs, np.uint16)
    ts = np.asarray(ts, np.uint64)
    n = len(xs)
    meta = SampleMeta(
        worker_id=0,
        spec_id=uid % 2,
        scene=f"scene{uid % 2}",
        seed=0,
        t_us=t_us,
        window_us=t_us,
        anchor_uuids=("depth_1",),
        episode_id=SampleMeta.make_episode_id(0, 0),
        step_idx=step_idx,
        is_first=is_first,
        is_last=is_last,
        sample_uid=uid,
    )
    sensors = {
        "depth_1": np.full((H, W), float(depth_fill), np.float32),
        "event_1": {
            "x": xs,
            "y": np.zeros(n, np.uint16),
            "t": ts,
            "p": np.ones(n, np.uint8),
        },
    }
    return TimeAlignedSample(meta, sensors)


# --------------------------------------------------------------------------- #
# normalization
# --------------------------------------------------------------------------- #
def test_normalize_events_parity():
    ev = {
        "x": np.array([0, 2], np.uint16),
        "y": np.array([0, 1], np.uint16),
        "t": np.array([10_000, 40_000], np.uint64),
        "p": np.array([1, 0], np.uint8),
    }
    out = normalize_events(ev, EventNorm(W, H, TIME_WINDOW))
    # parity with archived _process_sample column_stack formula
    expected = np.column_stack(
        [
            ev["x"] / W,
            ev["y"] / H,
            (ev["t"][-1] - ev["t"]) / TIME_WINDOW,
            ev["p"],
        ]
    ).astype(np.float32)
    assert np.allclose(out, expected)
    assert out.dtype == np.float32


def test_normalize_events_empty_and_disabled():
    assert normalize_events({}, EventNorm(W, H, TIME_WINDOW)).shape == (0, 4)
    ev = {
        "x": np.array([3], np.uint16),
        "y": np.array([1], np.uint16),
        "t": np.array([5], np.uint64),
        "p": np.array([1], np.uint8),
    }
    raw = normalize_events(ev, EventNorm(W, H, TIME_WINDOW, enabled=False))
    assert raw.tolist() == [[3, 1, 5, 1]]


def test_frame_buffer():
    fb = FrameBuffer(3, (2, 2), np.float32)
    fb.set(0, np.ones((2, 2), np.float32))
    fb.set(1, np.full((2, 2), 2.0, np.float32))
    out = fb.stacked_copy()
    assert out.shape == (3, 2, 2)
    assert np.all(out[0] == 1.0) and np.all(out[1] == 2.0)
    out[0, 0, 0] = 99.0  # copy is independent of the buffer
    assert fb.data[0, 0, 0] == 1.0


# --------------------------------------------------------------------------- #
# ShuffledBatcher
# --------------------------------------------------------------------------- #
def test_batcher_emits_on_full_and_shapes():
    b = ShuffledBatcher(_schema(), batch_size=3, event_norm=_norm())
    assert b.add(_sample(0, depth_fill=1, xs=[0, 1], ts=[10, 20], step_idx=0)) is None
    assert b.add(_sample(1, depth_fill=2, xs=[2], ts=[30], step_idx=1)) is None
    batch = b.add(_sample(2, depth_fill=3, xs=[], ts=[], step_idx=2))
    assert batch is not None

    depth = batch["depth_1"]
    assert depth.shape == (3, H, W)
    assert np.all(depth[0] == 1) and np.all(depth[2] == 3)

    counts, events = batch["event_1"]
    assert counts.tolist() == [2, 1, 0]  # third sample had no events
    assert events.shape == (3, 4)  # 2 + 1 + 0 events, 4 cols
    assert events[:, 0].max() <= 1.0  # x normalized into [0,1]


def test_batcher_meta_stacked():
    b = ShuffledBatcher(_schema(), batch_size=2, event_norm=_norm())
    b.add(_sample(10, depth_fill=1, xs=[0], ts=[1], t_us=100, step_idx=0, is_first=True))
    batch = b.add(_sample(11, depth_fill=2, xs=[1], ts=[2], t_us=200, step_idx=1, is_last=True))
    m = batch.meta
    assert m.t_us.tolist() == [100, 200]
    assert m.sample_uid.tolist() == [10, 11]
    assert m.step_idx.tolist() == [0, 1]
    assert m.is_first.tolist() == [True, False]
    assert m.is_last.tolist() == [False, True]
    assert m.scene == ["scene0", "scene1"]
    assert m.anchor_uuids == ("depth_1",)


def test_batcher_multiple_batches_reset():
    b = ShuffledBatcher(_schema(), batch_size=2, event_norm=_norm())
    out = []
    for i in range(4):
        r = b.add(_sample(i, depth_fill=i, xs=[i], ts=[i + 1], step_idx=i))
        if r is not None:
            out.append(r)
    assert len(out) == 2
    assert out[0].meta.sample_uid.tolist() == [0, 1]
    assert out[1].meta.sample_uid.tolist() == [2, 3]


def test_batcher_event_counts_concatenation_order():
    b = ShuffledBatcher(_schema(), batch_size=2, event_norm=_norm(enabled=False))
    b.add(_sample(0, depth_fill=1, xs=[1, 2], ts=[1, 2]))
    batch = b.add(_sample(1, depth_fill=2, xs=[3], ts=[3]))
    counts, events = batch["event_1"]
    assert counts.tolist() == [2, 1]
    # rows are concatenated in sample order (raw x preserved when norm disabled)
    assert events[:, 0].tolist() == [1, 2, 3]
