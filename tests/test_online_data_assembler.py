"""Unit tests for anchor-driven assembly (PR2). Pure numpy — no Habitat/torch.

Covers the load-bearing correctness properties: time-alignment of streams to the
anchor window, window length, monotonic step indexing, exact is_first/is_last via
hold-back-by-one, warm-up gating, ring-cap drop-oldest, always-emit on empty
windows, episode reset, determinism, ownership, and no cross-sample corruption.
"""

import numpy as np
import pytest

from neurosim.online_data import (
    AnchorAssembler,
    StreamAccumulator,
    SampleSchema,
)


SENSOR_CONFIGS = {
    "depth_1": {"type": "depth", "height": 4, "width": 4},
    "event_1": {"type": "event", "height": 4, "width": 4},
    "color_1": {"type": "color", "height": 4, "width": 4},
}


def _schema(latest=None, stream=("event_1",)):
    return SampleSchema.from_sensor_configs(
        SENSOR_CONFIGS,
        anchor=["depth_1"],
        stream=list(stream),
        latest=list(latest or []),
    )


def _depth(fill: float) -> np.ndarray:
    return np.full((4, 4), fill, dtype=np.float32)


def _events(ts) -> dict:
    ts = np.asarray(ts, dtype=np.uint64)
    n = len(ts)
    return {
        "x": np.zeros(n, np.uint16),
        "y": np.zeros(n, np.uint16),
        "t": ts.copy(),
        "p": np.zeros(n, np.uint8),
    }


def _run_three_windows(asm, depths=(1.0, 2.0, 3.0)):
    """Drive 3 anchor windows: events every 10ms, depth anchor every 50ms."""
    emitted = []
    asm.begin_episode(episode_idx=0, scene="apartment_1", seed=42)
    anchor_t = 0
    for k, fill in enumerate(depths):
        # 4 pre-anchor event packets in (prev, anchor)
        for j in range(1, 5):
            t = anchor_t + j * 10000
            emitted += asm.on_step({"event_1": _events([t])}, t)
        anchor_t += 50000
        emitted += asm.on_step(
            {"event_1": _events([anchor_t]), "depth_1": _depth(fill)}, anchor_t
        )
    emitted += asm.end_episode()
    return emitted


# --------------------------------------------------------------------------- #
# StreamAccumulator
# --------------------------------------------------------------------------- #
def test_stream_accumulator_concat_and_reset():
    acc = StreamAccumulator()
    acc.append(_events([1, 2]))
    acc.append(_events([3]))
    assert len(acc) == 3
    out = acc.flush()
    assert list(out["t"]) == [1, 2, 3]
    assert out["t"].flags["OWNDATA"]
    assert len(acc) == 0  # reset
    # Empty flush after a known packet -> zero-length arrays with same keys.
    empty = acc.flush()
    assert set(empty) == {"x", "y", "t", "p"} and empty["t"].shape == (0,)


def test_stream_accumulator_empty_before_any_packet():
    assert StreamAccumulator().flush() == {}


def test_stream_accumulator_ring_cap_drop_oldest():
    acc = StreamAccumulator(ring_cap=5)
    dropped = 0
    dropped += acc.append(_events([1, 2, 3]))
    dropped += acc.append(_events([4, 5, 6, 7]))  # total 7 -> drop 2 oldest
    assert dropped == 2
    assert len(acc) == 5
    out = acc.flush()
    assert list(out["t"]) == [3, 4, 5, 6, 7]  # newest kept


def test_stream_accumulator_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="share leading dim"):
        StreamAccumulator().append({"x": np.zeros(3), "t": np.zeros(2)})


# --------------------------------------------------------------------------- #
# Alignment / windowing / indexing
# --------------------------------------------------------------------------- #
def test_alignment_and_pairing():
    asm = AnchorAssembler(_schema(), worker_id=0, spec_id=0)
    samples = _run_three_windows(asm)
    assert len(samples) == 3

    prev = 0
    for k, s in enumerate(samples):
        t = (k + 1) * 50000
        assert s.meta.t_us == t
        assert s.meta.window_us == t - prev
        # depth paired with this anchor tick
        assert np.all(s.sensors["depth_1"] == float(k + 1))
        # every event timestamp lies in (prev, t]
        ev_t = s.sensors["event_1"]["t"]
        assert ev_t.min() > prev and ev_t.max() <= t
        prev = t


def test_step_idx_and_boundary_flags():
    asm = AnchorAssembler(_schema())
    samples = _run_three_windows(asm)
    assert [s.meta.step_idx for s in samples] == [0, 1, 2]
    assert [s.meta.is_first for s in samples] == [True, False, False]
    assert [s.meta.is_last for s in samples] == [False, False, True]
    # all share one episode id; sample_uid is monotonic
    assert len({s.meta.episode_id for s in samples}) == 1
    assert [s.meta.sample_uid for s in samples] == [0, 1, 2]


def test_single_sample_episode_is_first_and_last():
    asm = AnchorAssembler(_schema())
    asm.begin_episode(episode_idx=0)
    out = asm.on_step({"event_1": _events([10000]), "depth_1": _depth(1.0)}, 10000)
    assert out == []  # held back
    out += asm.end_episode()
    assert len(out) == 1
    assert out[0].meta.is_first and out[0].meta.is_last


def test_always_emit_empty_window():
    asm = AnchorAssembler(_schema())
    asm.begin_episode(episode_idx=0)
    emitted = []
    # window 1 has events; window 2 has none
    emitted += asm.on_step({"event_1": _events([10000]), "depth_1": _depth(1.0)}, 10000)
    emitted += asm.on_step({"depth_1": _depth(2.0)}, 20000)  # anchor, no events
    emitted += asm.end_episode()
    assert len(emitted) == 2
    # second sample emitted with empty (zero-length) event arrays, not dropped
    assert emitted[1].sensors["event_1"]["t"].shape == (0,)


# --------------------------------------------------------------------------- #
# Warm-up (latest sensor) + ring cap end-to-end
# --------------------------------------------------------------------------- #
def test_warmup_waits_for_latest_sensor():
    asm = AnchorAssembler(_schema(latest=["color_1"]))
    asm.begin_episode(episode_idx=0)
    # anchor fires before color ever seen -> skipped, nothing assembled
    out = asm.on_step({"depth_1": _depth(1.0), "event_1": _events([10000])}, 10000)
    assert out == [] and asm.stats["warmup_skips"] == 1
    # now color arrives, then anchor -> assembles (held back) then flush at end
    asm.on_step({"color_1": np.zeros((4, 4, 3), np.uint8)}, 15000)
    asm.on_step({"depth_1": _depth(2.0), "event_1": _events([20000])}, 20000)
    out = asm.end_episode()
    assert len(out) == 1
    assert out[0].sensors["color_1"].shape == (4, 4, 3)


def test_ring_cap_applied_per_window():
    asm = AnchorAssembler(_schema(), ring_caps={"event_1": 3})
    asm.begin_episode(episode_idx=0)
    asm.on_step({"event_1": _events([1, 2])}, 2)  # total 2, ok
    asm.on_step({"event_1": _events([3, 4, 5])}, 5)  # total 5 -> drop 2 oldest
    asm.on_step({"depth_1": _depth(1.0), "event_1": _events([6])}, 6)  # +1 -> drop 1
    out = asm.end_episode()
    # Cap enforced on every append: newest 3 kept; drops accumulate (2 + 1).
    assert list(out[0].sensors["event_1"]["t"]) == [4, 5, 6]
    assert asm.stats["overflow_drops:event_1"] == 3


# --------------------------------------------------------------------------- #
# Episode reset / determinism / ownership / aliasing
# --------------------------------------------------------------------------- #
def test_episode_reset_indices_and_ids():
    asm = AnchorAssembler(_schema())
    s0 = _run_three_windows(asm)
    s1 = _run_three_windows(asm)  # begin_episode(episode_idx=0) again
    # NOTE: _run_three_windows reuses episode_idx=0; assert per-run step reset
    assert [s.meta.step_idx for s in s1] == [0, 1, 2]
    # sample_uid keeps climbing globally across episodes
    assert s1[0].meta.sample_uid == s0[-1].meta.sample_uid + 1


def test_distinct_episode_ids():
    asm = AnchorAssembler(_schema())
    asm.begin_episode(episode_idx=0)
    asm.on_step({"depth_1": _depth(1.0), "event_1": _events([1])}, 1)
    e0 = asm._pending.meta.episode_id
    asm.end_episode()
    asm.begin_episode(episode_idx=1)
    asm.on_step({"depth_1": _depth(1.0), "event_1": _events([1])}, 1)
    e1 = asm._pending.meta.episode_id
    asm.end_episode()
    assert e0 != e1


def test_determinism_same_input_same_output():
    a = _run_three_windows(AnchorAssembler(_schema()))
    b = _run_three_windows(AnchorAssembler(_schema()))
    for sa, sb in zip(a, b):
        assert sa.meta == sb.meta
        assert np.array_equal(sa.sensors["depth_1"], sb.sensors["depth_1"])
        assert np.array_equal(sa.sensors["event_1"]["t"], sb.sensors["event_1"]["t"])


def test_emitted_payloads_are_owned():
    samples = _run_three_windows(AnchorAssembler(_schema()))
    for s in samples:
        assert s.sensors["depth_1"].flags["OWNDATA"]
        assert s.sensors["event_1"]["t"].flags["OWNDATA"]


def test_no_cross_sample_corruption():
    asm = AnchorAssembler(_schema())
    samples = _run_three_windows(asm)
    # snapshot the first sample, then it must be unaffected by later flushing
    first_depth = samples[0].sensors["depth_1"].copy()
    first_evt = samples[0].sensors["event_1"]["t"].copy()
    _run_three_windows(asm)  # more activity / flushes
    assert np.array_equal(samples[0].sensors["depth_1"], first_depth)
    assert np.array_equal(samples[0].sensors["event_1"]["t"], first_evt)


def test_strict_owned_rejects_view():
    asm = AnchorAssembler(_schema())
    asm.begin_episode(episode_idx=0)
    base = _depth(1.0)
    view = base[1:3]  # OWNDATA False
    with pytest.raises(ValueError):
        asm.on_step({"depth_1": view, "event_1": _events([1])}, 1)


def test_on_step_before_begin_episode_raises():
    asm = AnchorAssembler(_schema())
    with pytest.raises(RuntimeError, match="begin_episode"):
        asm.on_step({"depth_1": _depth(1.0)}, 1)
