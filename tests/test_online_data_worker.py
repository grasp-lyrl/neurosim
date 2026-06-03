"""Unit tests for SimulatorWorker + the host-copy boundary (PR2).

No Habitat: the worker is driven by a mock RandomizedSimulator that replays a
scripted measurement sequence. ``torch`` is required only to exercise the
GPU→host conversion (skipped if unavailable).
"""

from types import SimpleNamespace

import pytest
import numpy as np

from neurosim.online_data import SampleSchema, SimulatorWorker
from neurosim.online_data.sim_worker import _to_host_array, _events_to_host

torch = pytest.importorskip("torch")


SENSOR_CONFIGS = {
    "depth_1": {"type": "depth", "height": 4, "width": 4},
    "event_1": {"type": "event", "height": 4, "width": 4},
}


def _schema():
    return SampleSchema.from_sensor_configs(
        SENSOR_CONFIGS, anchor=["depth_1"], stream=["event_1"]
    )


class _FakeEvents:
    """Stand-in for the cu_esim events struct (reused GPU buffers in practice)."""

    def __init__(self, ts):
        n = len(ts)
        self.x = torch.zeros(n, dtype=torch.int32)
        self.y = torch.zeros(n, dtype=torch.int32)
        self.t = torch.tensor(ts, dtype=torch.int64)
        self.p = torch.zeros(n, dtype=torch.int32)


class _FakeSim:
    def __init__(self, configs, script):
        self.config = SimpleNamespace(
            visual_sensors=dict(configs), additional_sensors={}
        )
        self._script = script

    def run(self, callback_hook_=None, **kwargs):
        for measurements, t_s in self._script:
            callback_hook_(measurements, {}, t_s, 0)


class _FakeRSim:
    def __init__(self, sim, scene="apartment_1"):
        self.sim = sim
        self.last_sampled_settings = {"visual_backend": {"scene": scene}}
        self.closed = False

    def randomize(self, rng):
        pass

    def close(self):
        self.closed = True


def _depth(fill):
    return torch.full((4, 4), float(fill), dtype=torch.float32)


def _episode_script():
    """Two anchor windows (depth at 50ms, 100ms); events every 10ms."""
    steps = []
    anchor = 0
    for fill in (1.0, 2.0):
        for j in range(1, 5):
            t = anchor + j * 10000
            steps.append(({"event_1": _FakeEvents([t])}, t / 1e6))
        anchor += 50000
        steps.append(
            ({"event_1": _FakeEvents([anchor]), "depth_1": _depth(fill)}, anchor / 1e6)
        )
    return steps


# --------------------------------------------------------------------------- #
# Host-copy boundary (§0.10)
# --------------------------------------------------------------------------- #
def test_to_host_array_is_owned_and_independent():
    t = torch.zeros((4, 4), dtype=torch.float32)
    arr = _to_host_array(t)
    assert isinstance(arr, np.ndarray) and arr.flags["OWNDATA"]
    # mutating the source tensor must NOT change the host copy (anti-aliasing)
    t[0, 0] = 99.0
    assert arr[0, 0] == 0.0


def test_events_to_host_dtypes_and_none():
    out = _events_to_host(_FakeEvents([10, 20, 30]))
    assert set(out) == {"x", "y", "t", "p"}
    assert out["x"].dtype == np.uint16 and out["t"].dtype == np.uint64
    assert out["p"].dtype == np.uint8
    assert list(out["t"]) == [10, 20, 30]
    for v in out.values():
        assert v.flags["OWNDATA"]
    assert _events_to_host(None) == {}


# --------------------------------------------------------------------------- #
# SimulatorWorker end-to-end (mock sim)
# --------------------------------------------------------------------------- #
def _make_worker(seed=0):
    sim = _FakeSim(SENSOR_CONFIGS, _episode_script())
    rsim = _FakeRSim(sim)
    return SimulatorWorker(_schema(), rsim=rsim, worker_id=3, spec_id=1, seed=seed)


def test_worker_emits_aligned_samples():
    w = _make_worker()
    n = w.run_episode(episode_idx=0)
    assert n == 2 and len(w.samples) == 2

    prev = 0
    for k, s in enumerate(w.samples):
        t = (k + 1) * 50000
        assert s.meta.t_us == t
        assert s.meta.worker_id == 3 and s.meta.spec_id == 1
        assert s.meta.scene == "apartment_1"
        assert np.all(s.sensors["depth_1"] == float(k + 1))
        ev_t = s.sensors["event_1"]["t"]
        assert ev_t.dtype == np.uint64
        assert ev_t.min() > prev and ev_t.max() <= t
        prev = t
    assert w.samples[0].meta.is_first and w.samples[-1].meta.is_last


def test_worker_validate_rejects_mismatched_sim():
    bad = {
        "depth_1": {"type": "depth", "height": 8, "width": 8},  # wrong shape
        "event_1": {"type": "event", "height": 4, "width": 4},
    }
    rsim = _FakeRSim(_FakeSim(bad, []))
    with pytest.raises(ValueError, match="shape mismatch"):
        SimulatorWorker(_schema(), rsim=rsim)


def test_worker_emit_fn_routing():
    sink = []
    sim = _FakeSim(SENSOR_CONFIGS, _episode_script())
    w = SimulatorWorker(_schema(), rsim=_FakeRSim(sim), emit_fn=sink.append)
    w.run_episode(episode_idx=0)
    assert len(sink) == 2 and w.samples == []  # routed to sink, not default list


def test_worker_determinism():
    a = _make_worker(seed=7)
    b = _make_worker(seed=7)
    a.run_episode(episode_idx=0)
    b.run_episode(episode_idx=0)
    for sa, sb in zip(a.samples, b.samples):
        assert sa.meta == sb.meta
        assert np.array_equal(sa.sensors["depth_1"], sb.sensors["depth_1"])


def test_worker_close():
    w = _make_worker()
    w.close()
    assert w.rsim.closed
