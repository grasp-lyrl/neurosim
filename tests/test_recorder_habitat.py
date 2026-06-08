"""Habitat integration test for the offline H5 recorder.

Generates a couple of short domain-randomized episodes and asserts each produced a
canonical H5 file (per-sensor groups + state + sim_time) plus provenance metadata.
Skips cleanly when Habitat or the apartment_1 scene asset is unavailable. Run in the
``neurosim`` conda env.
"""

from pathlib import Path

import h5py
import pytest
import yaml

pytest.importorskip("habitat_sim")

from neurosim.online_data import record_dataset

_SETTINGS = Path("configs/apartment_1-settings.yaml")
_SCENE = Path("data/scene_datasets/habitat-test-scenes/apartment_1.glb")

if not _SETTINGS.exists() or not _SCENE.exists():
    pytest.skip(
        "apartment_1 settings/scene asset not available", allow_module_level=True
    )


def _short_settings(sim_time: float = 0.5) -> dict:
    with open(_SETTINGS) as f:
        settings = yaml.safe_load(f)
    settings["simulator"]["sim_time"] = sim_time
    return settings


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("recorder_ds")
    # 2 episodes, single in-process worker; randomize sensors so meta differs.
    randomization = {
        "resample_every": 1,
        "sensors": {
            "event_camera_1": {"contrast_threshold_pos": {"range": [0.1, 0.3]}}
        },
    }
    record_dataset(
        _short_settings(sim_time=0.5),
        out_dir=out,
        num_episodes=2,
        num_workers=1,
        gpu_ids=[0],
        base_seed=0,
        randomization=randomization,
    )
    return out


def test_per_episode_h5_files_exist(dataset_dir):
    files = sorted(dataset_dir.glob("episode_*.h5"))
    assert [f.name for f in files] == ["episode_000000.h5", "episode_000001.h5"]
    assert (dataset_dir / "dataset_setup.yaml").exists()


def test_h5_has_canonical_layout(dataset_dir):
    with h5py.File(dataset_dir / "episode_000000.h5", "r") as f:
        assert "depth_camera_1" in f and "data" in f["depth_camera_1"]
        assert f["depth_camera_1"]["data"].shape[1:] == (480, 640)
        # event group: flat-concatenated x/y/t/p
        for k in ("x", "y", "t", "p"):
            assert k in f["event_camera_1"]
        assert "state" in f
        # per-frame metadata, clock reset at episode start
        sim_step = f["depth_camera_1"]["sim_step"][:]
        assert len(sim_step) >= 1
        assert sim_step[0] >= 0
        assert list(sim_step) == sorted(sim_step)  # monotonic


def test_meta_provenance_differs_with_dr(dataset_dir):
    m0 = yaml.safe_load((dataset_dir / "episode_000000.meta.yaml").read_text())
    m1 = yaml.safe_load((dataset_dir / "episode_000001.meta.yaml").read_text())
    c0 = m0["visual_backend"]["sensors"]["event_camera_1"]["contrast_threshold_pos"]
    c1 = m1["visual_backend"]["sensors"]["event_camera_1"]["contrast_threshold_pos"]
    assert c0 != c1  # sensor DR produced distinct realizations per episode
