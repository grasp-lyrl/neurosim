"""Unit tests for batching (PR3). Pure numpy — no Habitat / mp.

Covers ShuffledBatcher (shapes by kind, event counts, batch.meta), anchor-relative
event packing (`shift_events`), FrameBuffer, and the empty-events / partial-batch
edge cases.
"""

import numpy as np

from neurosim.online_data import (
    LaneBatcher,
    SampleMeta,
    TimeAlignedSample,
    SampleSchema,
    ShuffledBatcher,
    FrameBuffer,
    shift_events,
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


def _sample(
    uid,
    *,
    depth_fill,
    xs,
    ts,
    t_us=1000,
    step_idx=0,
    is_first=False,
    is_last=False,
    worker_id=0,
):
    xs = np.asarray(xs, np.uint16)
    ts = np.asarray(ts, np.uint64)
    n = len(xs)
    meta = SampleMeta(
        worker_id=worker_id,
        spec_id=uid % 2,
        scene=f"scene{uid % 2}",
        seed=0,
        t_us=t_us,
        window_us=t_us,
        anchor_uuids=("depth_1",),
        episode_id=SampleMeta.make_episode_id(worker_id, 0),
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
# event packing (anchor-relative time, raw spatial/polarity)
# --------------------------------------------------------------------------- #
def test_shift_events_anchor_relative():
    ev = {
        "x": np.array([0, 2], np.uint16),
        "y": np.array([0, 1], np.uint16),
        "t": np.array([10_000, 40_000], np.uint64),
        "p": np.array([1, 0], np.uint8),
    }
    out = shift_events(ev, t_anchor_us=50_000)
    # raw x, y, p; time measured backwards from the anchor in µs.
    expected = np.column_stack(
        [ev["x"], ev["y"], 50_000 - ev["t"].astype(np.int64), ev["p"]]
    ).astype(np.float32)
    assert np.allclose(out, expected)
    assert out.dtype == np.float32
    assert out[:, 2].tolist() == [40_000.0, 10_000.0]  # newest -> smaller


def test_shift_events_empty_and_single():
    assert shift_events({}, t_anchor_us=10).shape == (0, 4)
    ev = {
        "x": np.array([3], np.uint16),
        "y": np.array([1], np.uint16),
        "t": np.array([4], np.uint64),
        "p": np.array([1], np.uint8),
    }
    out = shift_events(ev, t_anchor_us=10)
    assert out.tolist() == [[3, 1, 6, 1]]  # t_rel = 10 - 4


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
    b = ShuffledBatcher(_schema(), batch_size=3)
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
    assert events[:, 0].tolist() == [0, 1, 2]  # raw pixel x preserved (no scaling)


def test_batcher_meta_stacked():
    b = ShuffledBatcher(_schema(), batch_size=2)
    b.add(
        _sample(10, depth_fill=1, xs=[0], ts=[1], t_us=100, step_idx=0, is_first=True)
    )
    batch = b.add(
        _sample(11, depth_fill=2, xs=[1], ts=[2], t_us=200, step_idx=1, is_last=True)
    )
    m = batch.meta
    assert m.t_us.tolist() == [100, 200]
    assert m.sample_uid.tolist() == [10, 11]
    assert m.step_idx.tolist() == [0, 1]
    assert m.is_first.tolist() == [True, False]
    assert m.is_last.tolist() == [False, True]
    assert m.scene == ["scene0", "scene1"]
    assert m.anchor_uuids == ("depth_1",)


def test_batcher_multiple_batches_reset():
    b = ShuffledBatcher(_schema(), batch_size=2)
    out = []
    for i in range(4):
        r = b.add(_sample(i, depth_fill=i, xs=[i], ts=[i + 1], step_idx=i))
        if r is not None:
            out.append(r)
    assert len(out) == 2
    assert out[0].meta.sample_uid.tolist() == [0, 1]
    assert out[1].meta.sample_uid.tolist() == [2, 3]


def test_batcher_event_counts_concatenation_order():
    b = ShuffledBatcher(_schema(), batch_size=2)
    b.add(_sample(0, depth_fill=1, xs=[1, 2], ts=[1, 2]))
    batch = b.add(_sample(1, depth_fill=2, xs=[3], ts=[3]))
    counts, events = batch["event_1"]
    assert counts.tolist() == [2, 1]
    # rows are concatenated in sample order (raw x preserved)
    assert events[:, 0].tolist() == [1, 2, 3]


# --------------------------------------------------------------------------- #
# LaneBatcher: row i always continues producer i's own episode
# --------------------------------------------------------------------------- #
LANES = 3


def _lane_sample(worker_id: int, step_idx: int) -> TimeAlignedSample:
    """One step of one producer, tagged so the row it lands in is identifiable."""
    return _sample(
        uid=worker_id * 100 + step_idx,
        depth_fill=worker_id,
        xs=[worker_id],
        ts=[1000],
        step_idx=step_idx,
        is_first=(step_idx == 0),
        worker_id=worker_id,
    )


def test_lane_batcher_waits_for_every_lane():
    batcher = LaneBatcher(_schema(), LANES)

    # Two producers ahead by several steps still cannot fill a batch on their own.
    for step in range(4):
        for worker in (0, 1):
            assert batcher.add(_lane_sample(worker, step)) is None

    assert batcher.add(_lane_sample(2, 0)) is not None, (
        "the batch should complete as soon as the last lane delivers its first sample"
    )


def test_lane_batcher_orders_rows_by_worker():
    batcher = LaneBatcher(_schema(), LANES)

    batch = None
    for worker in (2, 0, 1):  # arrival order is deliberately scrambled
        batch = batcher.add(_lane_sample(worker, 0))

    assert list(batch.meta.worker_id) == [0, 1, 2], (
        "rows must be ordered by worker id, not by arrival"
    )
    assert [d[0, 0] for d in batch["depth_1"]] == [0.0, 1.0, 2.0]


def test_lane_batcher_keeps_each_row_on_its_own_episode():
    batcher = LaneBatcher(_schema(), LANES)
    batches = []

    for step in range(3):
        for worker in range(LANES):
            out = batcher.add(_lane_sample(worker, step))
            if out is not None:
                batches.append(out)

    assert len(batches) == 3
    for row in range(LANES):
        workers = [int(b.meta.worker_id[row]) for b in batches]
        steps = [int(b.meta.step_idx[row]) for b in batches]
        assert workers == [row] * 3, f"row {row} changed producer mid-stream"
        assert steps == [0, 1, 2], f"row {row} skipped or repeated a step: {steps}"


def test_lane_batcher_marks_episode_starts_per_row():
    batcher = LaneBatcher(_schema(), LANES)

    for worker in range(LANES):
        first = batcher.add(_lane_sample(worker, 0))
    for worker in range(LANES):
        # Lane 1 begins a fresh episode while the others continue theirs.
        second = batcher.add(_lane_sample(worker, 0 if worker == 1 else 1))

    assert list(first.meta.is_first) == [True] * LANES
    assert list(second.meta.is_first) == [False, True, False], (
        "is_first must mark exactly the lanes that restarted, so only they reset memory"
    )


def test_lane_batcher_never_mixes_two_steps_of_one_lane():
    batcher = LaneBatcher(_schema(), LANES)

    # A single fast producer must not be allowed to fill more than its own row.
    for step in range(LANES + 2):
        assert batcher.add(_lane_sample(0, step)) is None
