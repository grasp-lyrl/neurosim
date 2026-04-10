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
                    {"name": "a", "scene_path": "path/a.glb"},
                    {"name": "b", "scene_path": "path/b.glb"},
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

    def test_simulator_visual_backend_nested(self):
        base = _minimal_base_settings()
        cfg = DomainRandomizationConfig.from_dict(
            {
                "simulator": {"sim_time": {"range": [5.0, 6.0]}},
                "visual_backend": {"seed": {"choices": [1, 2, 3]}},
            }
        )
        rng = np.random.default_rng(99)
        out = cfg.sample(base, rng)
        assert 5.0 <= out["simulator"]["sim_time"] <= 6.0
        assert out["visual_backend"]["seed"] in (1, 2, 3)

    def test_base_unchanged_after_sample(self):
        base = _minimal_base_settings()
        orig_scene = base["visual_backend"]["scene"]
        cfg = DomainRandomizationConfig.from_dict(
            {"scenes": [{"name": "x", "scene_path": "other.glb"}]}
        )
        cfg.sample(base, np.random.default_rng(0))
        assert base["visual_backend"]["scene"] == orig_scene


class TestRandomizedSimulatorMocked:
    """Avoid Habitat by mocking SynchronousSimulator construction."""

    @patch(
        "neurosim.sims.synchronous_simulator.randomized_simulator.SynchronousSimulator"
    )
    def test_build_calls_simulator_with_transformed_settings(self, mock_cls):
        mock_inst = MagicMock()
        mock_cls.return_value = mock_inst

        def transform(d: dict) -> dict:
            d = dict(d)
            d["dynamics"] = {"patched": True}
            return d

        r = RandomizedSimulator(
            _minimal_base_settings(),
            randomization=None,
            visualizer_disabled=True,
            settings_transform=transform,
        )
        assert mock_cls.call_count == 1
        call_kw = mock_cls.call_args
        assert call_kw[1]["visualizer_disabled"] is True
        settings_passed = call_kw[0][0]
        assert settings_passed["dynamics"] == {"patched": True}
        r.close()
        mock_inst.close.assert_called()

    @patch(
        "neurosim.sims.synchronous_simulator.randomized_simulator.SynchronousSimulator"
    )
    def test_randomize_rebuilds_with_sampled_settings(self, mock_cls):
        mock_cls.return_value = MagicMock()
        dr = {
            "scenes": [{"name": "a", "scene_path": "picked.glb"}],
            "sensors": {"event_camera_1": {"hfov": {"choices": [77]}}},
        }
        r = RandomizedSimulator(
            _minimal_base_settings(),
            randomization=dr,
            visualizer_disabled=True,
        )
        assert mock_cls.call_count == 1
        r.randomize(np.random.default_rng(0))
        assert mock_cls.call_count == 2
        last_settings = mock_cls.call_args_list[-1][0][0]
        assert last_settings["visual_backend"]["scene"] == "picked.glb"
        assert (
            last_settings["visual_backend"]["sensors"]["event_camera_1"]["hfov"] == 77
        )
        r.close()

    @patch(
        "neurosim.sims.synchronous_simulator.randomized_simulator.SynchronousSimulator"
    )
    def test_randomize_without_config_same_as_rebuild_from_base(self, mock_cls):
        mock_cls.return_value = MagicMock()
        r = RandomizedSimulator(
            _minimal_base_settings(),
            randomization=None,
            visualizer_disabled=True,
        )
        mock_cls.reset_mock()
        r.randomize(np.random.default_rng(5))
        assert mock_cls.call_count == 1
        r.close()


def test_preflight_randomize_matches_train_sb3_seed_pattern():
    """Same RNG pattern as train_sb3.make_env: default_rng(seed + env_idx)."""
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
