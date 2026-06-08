"""Unit tests for the offline H5 recorder. Pure — no Habitat / GPU / mp.

Covers:
- ``_plan_shards`` sharding (full coverage, contiguous, unique, empty-shard drop,
  GPU cycling, per-worker seeds).
- ``record_episodes`` with ``build_randomized_sim`` mocked: per-episode randomize +
  ``run(log_h5=...)`` path, ``.meta.yaml`` provenance, close.
- ``record_dataset`` single-worker (in-process) path: ``dataset_setup.yaml`` + files.
"""

from itertools import chain
from unittest.mock import MagicMock, patch

import yaml

from neurosim.online_data.recorder import (
    _plan_shards,
    record_dataset,
    record_episodes,
)


def _base_settings():
    return {
        "simulator": {"sim_time": 1.0},
        "visual_backend": {"scene": "scene_a.glb", "sensors": {}},
    }


# --------------------------------------------------------------------------- #
# _plan_shards
# --------------------------------------------------------------------------- #
def test_plan_shards_full_partition_contiguous_unique():
    plan = _plan_shards(num_episodes=10, num_workers=3, gpu_ids=[0], base_seed=0)
    assert len(plan) == 3
    eps = [w["episodes"] for w in plan]
    # contiguous within a worker
    for shard in eps:
        assert shard == list(range(shard[0], shard[0] + len(shard)))
    # exact partition of range(10), no overlaps
    flat = list(chain.from_iterable(eps))
    assert sorted(flat) == list(range(10))
    assert len(flat) == len(set(flat))


def test_plan_shards_drops_empty_when_workers_exceed_episodes():
    plan = _plan_shards(num_episodes=2, num_workers=5, gpu_ids=[0], base_seed=0)
    assert len(plan) == 2  # only two non-empty shards
    assert all(len(w["episodes"]) == 1 for w in plan)
    assert sorted(chain.from_iterable(w["episodes"] for w in plan)) == [0, 1]


def test_plan_shards_gpu_cycling_and_seeds():
    plan = _plan_shards(num_episodes=4, num_workers=4, gpu_ids=[0, 1], base_seed=100)
    assert [w["gpu_id"] for w in plan] == [0, 1, 0, 1]  # cycled
    assert [w["seed"] for w in plan] == [100, 101, 102, 103]  # base_seed + i


# --------------------------------------------------------------------------- #
# record_episodes (mocked simulator)
# --------------------------------------------------------------------------- #
def _mock_rsim():
    rsim = MagicMock()
    rsim.last_sampled_settings = {"visual_backend": {"scene": "scene_a.glb"}}
    return rsim


@patch("neurosim.online_data.recorder.build_randomized_sim")
def test_record_episodes_writes_files_and_calls(mock_build, tmp_path):
    rsim = _mock_rsim()
    mock_build.return_value = rsim

    record_episodes(
        _base_settings(),
        out_dir=tmp_path,
        episodes=[0, 1, 2],
        gpu_id=3,
        seed=7,
        randomization={"resample_every": 5},
    )

    # built once on the right gpu/seed
    mock_build.assert_called_once_with(_base_settings(), {"resample_every": 5}, 3, 7)
    # randomize + run per episode, run targets the per-episode H5 path
    assert rsim.randomize.call_count == 3
    run_paths = [c.kwargs["log_h5"] for c in rsim.run.call_args_list]
    assert run_paths == [
        str(tmp_path / "episode_000000.h5"),
        str(tmp_path / "episode_000001.h5"),
        str(tmp_path / "episode_000002.h5"),
    ]
    # provenance meta written per episode
    for ep in (0, 1, 2):
        meta = tmp_path / f"episode_{ep:06d}.meta.yaml"
        assert meta.exists()
        assert (
            yaml.safe_load(meta.read_text())["visual_backend"]["scene"] == "scene_a.glb"
        )
    rsim.close.assert_called_once()


@patch("neurosim.online_data.recorder.build_randomized_sim")
def test_record_episodes_write_meta_false(mock_build, tmp_path):
    mock_build.return_value = _mock_rsim()
    record_episodes(_base_settings(), out_dir=tmp_path, episodes=[0], write_meta=False)
    assert not (tmp_path / "episode_000000.meta.yaml").exists()


@patch("neurosim.online_data.recorder.build_randomized_sim")
def test_record_episodes_closes_on_error(mock_build, tmp_path):
    rsim = _mock_rsim()
    rsim.run.side_effect = RuntimeError("boom")
    mock_build.return_value = rsim
    try:
        record_episodes(_base_settings(), out_dir=tmp_path, episodes=[0])
    except RuntimeError:
        pass
    rsim.close.assert_called_once()  # cleanup in finally


# --------------------------------------------------------------------------- #
# record_dataset (single-worker in-process path)
# --------------------------------------------------------------------------- #
@patch("neurosim.online_data.recorder.build_randomized_sim")
def test_record_dataset_single_worker_inprocess(mock_build, tmp_path):
    mock_build.return_value = _mock_rsim()

    record_dataset(
        _base_settings(),
        out_dir=tmp_path,
        num_episodes=3,
        num_workers=1,
        gpu_ids=[2],
        base_seed=0,
    )

    setup = yaml.safe_load((tmp_path / "dataset_setup.yaml").read_text())
    assert setup["num_episodes"] == 3
    assert setup["num_workers"] == 1
    assert setup["gpus"] == [2]
    # all three episodes recorded in-process (mock writes no real H5, but meta is real)
    for ep in (0, 1, 2):
        assert (tmp_path / f"episode_{ep:06d}.meta.yaml").exists()
    mock_build.assert_called_once_with(_base_settings(), None, 2, 0)
