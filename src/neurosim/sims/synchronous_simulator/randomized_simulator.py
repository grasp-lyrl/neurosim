"""Wrapper around SynchronousSimulator with declarative domain randomization.

Provides :class:`RandomizedSimulator`, a thin wrapper that holds a *base*
settings dict and an optional randomization specification.  When
:meth:`randomize` is called it deep-copies the base settings, samples every
randomizable parameter, and rebuilds the inner
:class:`SynchronousSimulator`.  When no randomization config is supplied the
wrapper is a transparent pass-through.

The randomization config uses an explicit ``range`` / ``choices`` syntax so
there is never ambiguity between a two-element choice list and a continuous
range::

    domain_randomization:
      scenes:
        - name: apartment_1
          scene_path: data/scene_datasets/.../apartment_1.glb
        - name: skokloster
          scene_path: data/scene_datasets/.../skokloster-castle.glb
      sensors:
        event_camera_1:
          contrast_threshold_neg:
            range: [0.1, 0.6]
          hfov:
            choices: [90, 100, 120]
      simulator: {}
      visual_backend: {}
"""

import copy
import yaml
import logging
import numpy as np
from typing import Any
from pathlib import Path
from dataclasses import dataclass, field

from neurosim.sims.synchronous_simulator.simulator import SynchronousSimulator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def _sample_value(spec: Any, rng: np.random.Generator) -> Any:
    """Resolve a single randomization spec to a concrete value.

    *spec* is one of:
    - ``{"range": [lo, hi]}`` -> ``rng.uniform(lo, hi)``
    - ``{"choices": [a, b, ...]}`` -> ``rng.choice(...)``
    - any other value -> returned as-is (fixed override)
    """
    if isinstance(spec, dict):
        if "range" in spec:
            lo, hi = spec["range"]
            return float(rng.uniform(lo, hi))
        if "choices" in spec:
            options = list(spec["choices"])
            idx = int(rng.integers(0, len(options)))
            return options[idx]
    return spec


def _apply_randomization_layer(
    target: dict[str, Any],
    overrides: dict[str, Any],
    rng: np.random.Generator,
) -> None:
    """Recursively walk *overrides* and apply sampled values into *target*.

    Leaf nodes that are ``{"range": ...}`` or ``{"choices": ...}`` dicts are
    sampled; plain dicts recurse; everything else is a fixed override.
    """
    for key, spec in overrides.items():
        if isinstance(spec, dict) and "range" not in spec and "choices" not in spec:
            if key not in target:
                target[key] = {}
            if isinstance(target[key], dict):
                _apply_randomization_layer(target[key], spec, rng)
            else:
                target[key] = _sample_value(spec, rng)
        else:
            target[key] = _sample_value(spec, rng)


@dataclass
class DomainRandomizationConfig:
    """Parsed representation of a ``domain_randomization`` YAML section.

    Attributes:
        scenes: List of ``{"name": ..., "scene_path": ...}`` dicts.  One is
            chosen uniformly at random per :meth:`sample`.
        sensors: Per-sensor-UUID dict of randomizable parameters.
        simulator: Randomizable overrides for the ``simulator`` settings block.
        visual_backend: Randomizable overrides for ``visual_backend`` (non-sensor
            fields).
    """

    scenes: list[dict[str, str]] = field(default_factory=list)
    sensors: dict[str, dict[str, Any]] = field(default_factory=dict)
    simulator: dict[str, Any] = field(default_factory=dict)
    visual_backend: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DomainRandomizationConfig":
        return cls(
            scenes=list(data.get("scenes", [])),
            sensors=dict(data.get("sensors", {})),
            simulator=dict(data.get("simulator", {})),
            visual_backend=dict(data.get("visual_backend", {})),
        )

    def sample(
        self,
        base_settings: dict[str, Any],
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        """Return a new settings dict with randomized parameters applied.

        The *base_settings* dict is deep-copied before mutation.
        """
        settings = copy.deepcopy(base_settings)

        # -- Scene randomization ------------------------------------------------
        if self.scenes:
            scene = self.scenes[int(rng.integers(0, len(self.scenes)))]
            settings.setdefault("visual_backend", {})["scene"] = scene["scene_path"]

        # -- Sensor parameter randomization -------------------------------------
        if self.sensors:
            vb_sensors = settings.setdefault("visual_backend", {}).setdefault(
                "sensors", {}
            )
            for uuid, param_specs in self.sensors.items():
                if uuid not in vb_sensors:
                    logger.warning(
                        "Randomization references sensor '%s' not present in base settings; skipping",
                        uuid,
                    )
                    continue
                _apply_randomization_layer(vb_sensors[uuid], param_specs, rng)

        # -- Simulator-level randomization --------------------------------------
        if self.simulator:
            _apply_randomization_layer(
                settings.setdefault("simulator", {}), self.simulator, rng
            )

        # -- Visual-backend-level randomization (non-sensor fields) -------------
        if self.visual_backend:
            _apply_randomization_layer(
                settings.setdefault("visual_backend", {}), self.visual_backend, rng
            )

        return settings


class RandomizedSimulator:
    """Wrapper around :class:`SynchronousSimulator` with domain randomization.

    When *randomization* is ``None`` (the default) the wrapper simply builds a
    ``SynchronousSimulator`` from *base_settings* and delegates every attribute
    access to it -- zero overhead, fully transparent.

    When a randomization config is provided, calling :meth:`randomize` samples
    new settings and rebuilds the simulator.

    Parameters
    ----------
    base_settings:
        Path to a YAML file **or** a raw settings ``dict``.
    randomization:
        Optional dict matching the ``domain_randomization`` schema (scenes,
        sensors, simulator, visual_backend).  ``None`` disables randomization.
    visualizer_disabled:
        Forwarded to ``SynchronousSimulator``.
    settings_transform:
        Optional callable ``(dict) -> dict`` applied to settings *after*
        randomization and *before* building the simulator.  Useful for
        consumers that need to patch settings (e.g. RL env strips trajectory
        and overrides dynamics).
    """

    def __init__(
        self,
        base_settings: str | Path | dict[str, Any],
        randomization: dict[str, Any] | None = None,
        visualizer_disabled: bool = False,
        settings_transform: Any | None = None,
    ):
        self._base_settings = self._load_settings(base_settings)
        self._rand_cfg: DomainRandomizationConfig | None = (
            DomainRandomizationConfig.from_dict(randomization)
            if randomization
            else None
        )
        self._viz_disabled = visualizer_disabled
        self._settings_transform = settings_transform
        self.sim: SynchronousSimulator | None = None
        self._last_sampled_settings: dict[str, Any] | None = None

        self.build()

    # -- Settings loading ---------------------------------------------------

    @staticmethod
    def _load_settings(settings: str | Path | dict[str, Any]) -> dict[str, Any]:
        if isinstance(settings, dict):
            return copy.deepcopy(settings)
        path = Path(settings)
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    # -- Build / randomize --------------------------------------------------

    def _prepare_settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        if self._settings_transform is not None:
            settings = self._settings_transform(settings)
        return settings

    def build(self) -> None:
        """(Re-)build the simulator from base settings without randomization."""
        if self.sim is not None:
            self.sim.close()
        settings = self._prepare_settings(copy.deepcopy(self._base_settings))
        self._last_sampled_settings = settings
        self.sim = SynchronousSimulator(
            settings, visualizer_disabled=self._viz_disabled
        )

    def randomize(self, rng: np.random.Generator) -> None:
        """Sample new settings from the randomization config and rebuild.

        If no randomization config was provided this is equivalent to
        :meth:`build` (rebuilds from base settings).
        """
        if self.sim is not None:
            self.sim.close()

        if self._rand_cfg is not None:
            settings = self._rand_cfg.sample(self._base_settings, rng)
        else:
            settings = copy.deepcopy(self._base_settings)

        settings = self._prepare_settings(settings)
        self._last_sampled_settings = settings
        self.sim = SynchronousSimulator(
            settings, visualizer_disabled=self._viz_disabled
        )

    @property
    def last_sampled_settings(self) -> dict[str, Any] | None:
        """The settings dict used to build the current simulator instance."""
        return self._last_sampled_settings

    # -- Delegation ---------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        sim = self.__dict__.get("sim")
        if sim is not None:
            return getattr(sim, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}' "
            "(inner simulator not yet built)"
        )

    def close(self) -> None:
        """Close the inner simulator."""
        if self.sim is not None:
            self.sim.close()
            self.sim = None
