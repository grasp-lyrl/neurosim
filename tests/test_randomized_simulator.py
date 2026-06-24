"""Unit tests for domain randomization (RandomizedSimulator, DomainRandomizationConfig)."""

import copy
from unittest.mock import MagicMock, patch

import numpy as np

from neurosim.sims.synchronous_simulator.randomized_simulator import (
    DomainRandomizationConfig,
    RandomizedSimulator,
    _sample_value,
)


def _minimal_base_settings() -> dict:
    return {
        "simulator": {"world_rate": 100, "control_rate": 10, "sim_time": 1.0},
        "visual_backend": {
            "scene": "scene_a.glb",
            "sensors": {
                "event_camera_1": {
                    "type": "event",
                    "contrast_threshold_neg": 0.2,
                    "contrast_threshold_pos": 0.2,
                    "hfov": 90,
                }
            },
        },
        "dynamics": {"model": "rotorpy_multirotor_euler", "vehicle": "crazyflie"},
        "controller": {"model": "rotorpy_se3", "vehicle": "crazyflie"},
    }


class TestSampleValue:
    def test_range_samples_float_in_bounds(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            v = _sample_value({"range": [0.1, 0.6]}, rng)
            assert isinstance(v, float)
            assert 0.1 <= v <= 0.6

    def test_choices_picks_from_list(self):
        rng = np.random.default_rng(1)
        options = [90, 100, 120]
        for _ in range(30):
            v = _sample_value({"choices": options}, rng)
            assert v in options

    def test_plain_scalar_unchanged(self):
        rng = np.random.default_rng(2)
        assert _sample_value(42, rng) == 42
        assert _sample_value("fixed", rng) == "fixed"


class TestDomainRandomizationConfigSample:
    def test_scenes_replaces_visual_backend_scene(self):
        cfg = DomainRandomizationConfig.from_dict(
            {
                "scenes": [
                    {"name": "a", "path": "path/a.glb"},
                    {"name": "b", "path": "path/b.glb"},
                ]
            }
        )
        rng = np.random.default_rng(123)
        seen = set()
        for _ in range(50):
            out = cfg.sample(_minimal_base_settings(), rng)
            seen.add(out["visual_backend"]["scene"])
        assert seen <= {"path/a.glb", "path/b.glb"}
        assert len(seen) == 2

    def test_sensors_range_and_choices(self):
        cfg = DomainRandomizationConfig.from_dict(
            {
                "sensors": {
                    "event_camera_1": {
                        "contrast_threshold_neg": {"range": [0.1, 0.2]},
                        "hfov": {"choices": [60, 120]},
                    }
                }
            }
        )
        rng = np.random.default_rng(7)
        out = cfg.sample(_minimal_base_settings(), rng)
        s = out["visual_backend"]["sensors"]["event_camera_1"]
        assert 0.1 <= s["contrast_threshold_neg"] <= 0.2
        assert s["hfov"] in (60, 120)
        assert s["contrast_threshold_pos"] == 0.2

    def test_base_unchanged_after_sample(self):
        base = _minimal_base_settings()
        orig_scene = base["visual_backend"]["scene"]
        cfg = DomainRandomizationConfig.from_dict(
            {"scenes": [{"name": "x", "path": "other.glb"}]}
        )
        cfg.sample(base, np.random.default_rng(0))
        assert base["visual_backend"]["scene"] == orig_scene

    def test_scenes_glob_expands_into_scenes(self, tmp_path):
        # scenes_glob is expanded here -> works for loader AND recorder alike.
        (tmp_path / "00000-AAA").mkdir()
        (tmp_path / "00001-BBB").mkdir()
        (tmp_path / "00000-AAA" / "AAA.basis.glb").write_text("")
        (tmp_path / "00001-BBB" / "BBB.basis.glb").write_text("")

        cfg = DomainRandomizationConfig.from_dict(
            {"scenes_glob": str(tmp_path / "*" / "*.basis.glb")}
        )
        assert sorted(s["name"] for s in cfg.scenes) == ["AAA", "BBB"]
        assert all(s["path"].endswith(".basis.glb") for s in cfg.scenes)

    def test_scenes_glob_merges_with_explicit_scenes(self, tmp_path):
        (tmp_path / "X.basis.glb").write_text("")
        cfg = DomainRandomizationConfig.from_dict(
            {
                "scenes": [{"name": "explicit", "path": "given/e.glb"}],
                "scenes_glob": str(tmp_path / "*.basis.glb"),
            }
        )
        names = sorted(s["name"] for s in cfg.scenes)
        assert names == ["X", "explicit"]


class TestRandomizedSimulatorMocked:
    """Avoid Habitat by mocking SynchronousSimulator construction."""

    @patch(
        "neurosim.sims.synchronous_simulator.randomized_simulator.SynchronousSimulator"
    )
    def test_build_calls_simulator(self, mock_cls):
        mock_inst = MagicMock()
        mock_cls.return_value = mock_inst

        r = RandomizedSimulator(
            _minimal_base_settings(),
            randomization=None,
            visualizer_disabled=True,
        )
        assert mock_cls.call_count == 1
        call_kw = mock_cls.call_args
        assert call_kw[1]["visualizer_disabled"] is True
        r.close()
        mock_inst.close.assert_called()

    @patch(
        "neurosim.sims.synchronous_simulator.randomized_simulator.SynchronousSimulator"
    )
    def test_build_bootstraps_empty_scene_from_pool(self, mock_cls):
        # Empty base scene -> build() fills it from the DR scene pool (scenes[0]).
        mock_cls.return_value = MagicMock()
        base = _minimal_base_settings()
        base["visual_backend"]["scene"] = ""
        r = RandomizedSimulator(
            base,
            randomization={"scenes": [{"name": "a", "path": "boot.glb"}]},
            visualizer_disabled=True,
        )
        settings_arg = mock_cls.call_args[0][0]  # SynchronousSimulator(settings, ...)
        assert settings_arg["visual_backend"]["scene"] == "boot.glb"
        r.close()

    @patch(
        "neurosim.sims.synchronous_simulator.randomized_simulator.SynchronousSimulator"
    )
    def test_build_empty_scene_without_pool_raises(self, mock_cls):
        import pytest

        mock_cls.return_value = MagicMock()
        base = _minimal_base_settings()
        base["visual_backend"]["scene"] = ""
        with pytest.raises(ValueError, match="scenes/scenes_glob"):
            RandomizedSimulator(base, randomization=None, visualizer_disabled=True)

    @patch(
        "neurosim.sims.synchronous_simulator.randomized_simulator.SynchronousSimulator"
    )
    def test_randomize_rebuilds_with_sampled_settings(self, mock_cls):
        mock_inst = MagicMock()
        mock_cls.return_value = mock_inst
        dr = {
            "scenes": [{"name": "a", "path": "picked.glb"}],
            "sensors": {"event_camera_1": {"hfov": {"choices": [77]}}},
        }
        r = RandomizedSimulator(
            _minimal_base_settings(),
            randomization=dr,
            visualizer_disabled=True,
        )
        assert mock_cls.call_count == 1
        r.randomize(np.random.default_rng(0))
        assert mock_cls.call_count == 1
        mock_inst.reconfigure.assert_called_once()
        last_settings = mock_inst.reconfigure.call_args[0][0]
        assert last_settings["visual_backend"]["scene"] == "picked.glb"
        assert (
            last_settings["visual_backend"]["sensors"]["event_camera_1"]["hfov"] == 77
        )
        r.close()

    @patch(
        "neurosim.sims.synchronous_simulator.randomized_simulator.SynchronousSimulator"
    )
    def test_randomize_without_config_does_not_reconfigure(self, mock_cls):
        # No DR config -> scene is fixed, so randomize must NOT reconfigure.
        mock_inst = MagicMock()
        mock_cls.return_value = mock_inst
        r = RandomizedSimulator(
            _minimal_base_settings(), randomization=None, visualizer_disabled=True
        )
        mock_inst.reset_mock()
        rebuilt = r.randomize(np.random.default_rng(5))
        assert rebuilt is False
        mock_inst.reconfigure.assert_not_called()
        r.close()

    @patch(
        "neurosim.sims.synchronous_simulator.randomized_simulator.SynchronousSimulator"
    )
    def test_resample_every_cadence(self, mock_cls):
        # Reconfigure only on episodes where episode % resample_every == 0.
        mock_inst = MagicMock()
        mock_cls.return_value = mock_inst
        dr = {"scenes": [{"name": "a", "path": "a.glb"}], "resample_every": 3}
        r = RandomizedSimulator(
            _minimal_base_settings(), randomization=dr, visualizer_disabled=True
        )
        mock_inst.reset_mock()  # ignore the build()
        flags = [r.randomize(np.random.default_rng(0)) for _ in range(6)]
        # episodes 0..5 -> rebuild at 0 and 3
        assert flags == [True, False, False, True, False, False]
        assert mock_inst.reconfigure.call_count == 2
        r.close()

    @patch(
        "neurosim.sims.synchronous_simulator.randomized_simulator.SynchronousSimulator"
    )
    def test_trajectory_seed_deterministic_and_decoupled(self, mock_cls):
        # Per-episode trajectory seed depends only on (seed, episode) — reproducible
        # and independent of scene/sensor cadence.
        def per_episode_seeds(resample_every):
            inst = MagicMock()
            inst.settings = {
                "trajectory": {"model": "x"}
            }  # real dict -> "trajectory" in settings
            mock_cls.return_value = inst
            r = RandomizedSimulator(
                _minimal_base_settings(),
                randomization={
                    "scenes": [{"name": "a", "path": "a.glb"}],
                    "resample_every": resample_every,
                },
                visualizer_disabled=True,
                seed=123,
            )
            seeds = []
            for _ in range(4):
                r.randomize(np.random.default_rng(0))
                seeds.append(inst.renew_trajectory.call_args[0][0]["seed"])
            r.close()
            return seeds

        s1 = per_episode_seeds(resample_every=1)
        s3 = per_episode_seeds(resample_every=3)
        assert s1 == s3  # decoupled from scene cadence + reproducible
        assert len(set(s1)) == 4  # a fresh seed every episode


class TestTrajectoryAndCadenceConfig:
    def test_resample_every(self):
        assert (
            DomainRandomizationConfig.from_dict({"resample_every": 50}).resample_every
            == 50
        )
        assert DomainRandomizationConfig.from_dict({}).resample_every == 1

    def test_sample_trajectory_always_seeds_plus_params(self):
        cfg = DomainRandomizationConfig.from_dict(
            {"trajectory": {"v_avg": {"range": [0.8, 1.5]}}}
        )
        out = cfg.sample_trajectory(np.random.default_rng(0))
        assert isinstance(out["seed"], int)  # always a fresh seed, no flag
        assert 0.8 <= out["v_avg"] <= 1.5

    def test_sample_trajectory_seeds_without_params(self):
        cfg = DomainRandomizationConfig.from_dict({})
        assert "seed" in cfg.sample_trajectory(np.random.default_rng(0))

    def test_sample_trajectory_ignores_legacy_reseed_key(self):
        # A stale `reseed` key must not leak into trajectory settings.
        cfg = DomainRandomizationConfig.from_dict({"trajectory": {"reseed": True}})
        out = cfg.sample_trajectory(np.random.default_rng(0))
        assert "reseed" not in out and "seed" in out


def test_preflight_randomize_distinct_seeds():
    """Different seeds produce different sampled settings."""
    dr = {
        "sensors": {
            "event_camera_1": {
                "contrast_threshold_neg": {"range": [0.01, 0.99]},
            }
        }
    }
    base = _minimal_base_settings()
    contrasts = []
    with patch(
        "neurosim.sims.synchronous_simulator.randomized_simulator.SynchronousSimulator"
    ):
        for env_idx in (0, 1):
            cfg = copy.deepcopy(base)
            rng = np.random.default_rng(42 + env_idx)
            rsim = RandomizedSimulator(
                cfg,
                randomization=dr,
                visualizer_disabled=True,
            )
            rsim.randomize(rng)
            contrasts.append(
                rsim.last_sampled_settings["visual_backend"]["sensors"][
                    "event_camera_1"
                ]["contrast_threshold_neg"]
            )
            rsim.close()
    assert contrasts[0] != contrasts[1]


def test_sims_package_exports_randomized_simulator():
    from neurosim.sims.synchronous_simulator import (
        DomainRandomizationConfig,
        RandomizedSimulator,
    )

    assert RandomizedSimulator is not None
    assert DomainRandomizationConfig is not None
