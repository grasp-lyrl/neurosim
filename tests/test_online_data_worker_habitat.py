"""Habitat integration test for SimulatorWorker (PR2).

Runs a real (short) simulation and asserts the producer emits time-aligned
events+depth samples. Skips cleanly when Habitat or the apartment_1 scene asset
is unavailable. Run in the ``neurosim`` conda env.
"""

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("habitat_sim")

from neurosim.online_data import SampleSchema, SimulatorWorker

_SETTINGS = Path("configs/apartment_1-settings.yaml")
_SCENE = Path("data/scene_datasets/habitat-test-scenes/apartment_1.glb")

if not _SETTINGS.exists() or not _SCENE.exists():
    pytest.skip(
        "apartment_1 settings/scene asset not available", allow_module_level=True
    )


def _short_settings(sim_time: float = 1.0) -> dict:
    with open(_SETTINGS) as f:
        settings = yaml.safe_load(f)
    settings["simulator"]["sim_time"] = sim_time
    return settings


@pytest.fixture(scope="module")
def worker_samples():
    settings = _short_settings(sim_time=1.0)
    schema = SampleSchema.from_sensor_configs(
        {
            uuid: cfg
            for uuid, cfg in settings["visual_backend"]["sensors"].items()
            if uuid in ("depth_camera_1", "event_camera_1")
        },
        anchor=["depth_camera_1"],
        stream=["event_camera_1"],
    )
    worker = SimulatorWorker(
        schema, base_settings=copy.deepcopy(settings), worker_id=0, gpu_id=0, seed=0
    )
    try:
        worker.run_episode(episode_idx=0)
        yield worker.samples
    finally:
        worker.close()


def test_produces_samples(worker_samples):
    # depth anchor @20Hz over ~1s -> ~20 samples (allow warm-up slack).
    assert len(worker_samples) >= 5


def test_samples_are_time_aligned(worker_samples):
    prev = 0
    for k, s in enumerate(worker_samples):
        assert s.meta.t_us > prev
        assert s.meta.window_us == s.meta.t_us - prev
        assert s.meta.step_idx == k

        depth = s.sensors["depth_camera_1"]
        assert depth.shape == (480, 640) and depth.flags["OWNDATA"]

        ev = s.sensors["event_camera_1"]
        assert ev["t"].dtype == np.uint64
        if ev["t"].size:  # events fall inside the anchor window (t_prev, t]
            assert ev["t"].min() > prev and ev["t"].max() <= s.meta.t_us
        prev = s.meta.t_us


def test_boundary_flags_and_scene(worker_samples):
    assert worker_samples[0].meta.is_first
    assert worker_samples[-1].meta.is_last
    assert sum(s.meta.is_first for s in worker_samples) == 1
    assert sum(s.meta.is_last for s in worker_samples) == 1
    assert worker_samples[0].meta.scene.endswith("apartment_1.glb")


def test_cheap_trajectory_reset_on_light_episodes():
    """Real-Habitat check of the cost-tiered hotfix: a light episode goes through
    ``sim.renew_trajectory`` (rebuild traj + clock reset, no scene reload) and a
    heavy episode after it still reconfigures cleanly."""
    settings = _short_settings(sim_time=0.5)
    schema = SampleSchema.from_sensor_configs(
        {
            uuid: cfg
            for uuid, cfg in settings["visual_backend"]["sensors"].items()
            if uuid in ("depth_camera_1", "event_camera_1")
        },
        anchor=["depth_camera_1"],
        stream=["event_camera_1"],
    )
    # scene reconfigure every 2 episodes -> ep0 heavy, ep1 light, ep2 heavy.
    # Cadence is owned by RandomizedSimulator via the DR config; the trajectory is
    # reseeded every episode automatically.
    worker = SimulatorWorker(
        schema,
        base_settings=copy.deepcopy(settings),
        randomization={"resample_every": 2},
        worker_id=0,
        gpu_id=0,
        seed=0,
    )
    try:
        for ep in range(3):
            worker.run_episode(episode_idx=ep)
    finally:
        worker.close()

    by_ep: dict = {}
    for s in worker.samples:
        by_ep.setdefault(s.meta.episode_id, []).append(s)

    assert len(by_ep) == 3  # all three episodes (incl. the light one) produced data
    for samples in by_ep.values():
        assert len(samples) >= 1
        assert samples[0].meta.step_idx == 0  # clock reset each episode
        ts = [s.meta.t_us for s in samples]
        assert ts == sorted(ts)  # monotonic within the episode
        assert samples[0].meta.is_first and samples[-1].meta.is_last
