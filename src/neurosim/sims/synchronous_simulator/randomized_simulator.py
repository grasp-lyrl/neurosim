"""Wrapper around SynchronousSimulator with declarative domain randomization.

Provides :class:`RandomizedSimulator`, a thin wrapper that holds a *base*
settings dict and an optional randomization specification.  When
:meth:`randomize` is called it deep-copies the base settings, samples every
randomizable parameter, and applies them via
:meth:`~neurosim.sims.synchronous_simulator.simulator.SynchronousSimulator.reconfigure`
(reusing the visual backend context).  When no randomization config is supplied the
wrapper is a transparent pass-through.

The randomization config uses an explicit ``range`` / ``choices`` syntax.
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

    Two kinds of randomization, applied through different mechanisms (see
    :meth:`RandomizedSimulator.randomize`):

    * **scene + sensors** — expensive (Habitat reconfigure / scene reload);
      resampled only every :attr:`resample_every` episodes.
    * **trajectory** — cheap (rebuilt in-place on the loaded navmesh); resampled
      **every episode** with a **fresh seed** (so each episode flies a new path),
      plus optional param ranges (e.g. ``v_avg``) using the same ``range`` /
      ``choices`` grammar as sensors.

    Attributes:
        scenes: List of ``{"name": ..., "path": ...}`` dicts; one chosen per :meth:`sample`.
        sensors: Per-sensor-UUID dict of randomizable parameters.
        resample_every: Episodes between scene/sensor reconfigures (same meaning as
            the RL ``domain_randomization.resample_every``).
        trajectory: Optional per-param ``{range|choices}`` specs for the trajectory.
    """

    scenes: list[dict[str, str]] = field(default_factory=list)
    sensors: dict[str, dict[str, Any]] = field(default_factory=dict)
    resample_every: int = 1
    trajectory: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DomainRandomizationConfig":
        return cls(
            scenes=list(data.get("scenes", [])),
            sensors=dict(data.get("sensors", {})),
            resample_every=max(1, int(data.get("resample_every", 1))),
            trajectory=dict(data.get("trajectory", {})),
        )

    def sample(
        self,
        base_settings: dict[str, Any],
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        """Return a new settings dict with scene + sensor randomization applied.

        The *base_settings* dict is deep-copied before mutation. (Trajectory is
        handled separately by :meth:`sample_trajectory` / ``renew_trajectory``.)
        """
        settings = copy.deepcopy(base_settings)

        if self.scenes:
            scene = self.scenes[int(rng.integers(0, len(self.scenes)))]
            settings.setdefault("visual_backend", {})["scene"] = scene["path"]

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

        return settings

    def sample_trajectory(self, rng: np.random.Generator) -> dict[str, Any]:
        """Sample trajectory overrides: always a fresh ``seed`` + any param ranges.

        ``rng`` should be the per-episode, deterministically-derived generator from
        :meth:`RandomizedSimulator.randomize` so the trajectory is reproducible and
        decoupled from scene/sensor sampling.
        """
        out: dict[str, Any] = {"seed": int(rng.integers(0, 2**31 - 1))}
        # (ignore a stray legacy ``reseed`` key if a config still carries one)
        params = {k: v for k, v in self.trajectory.items() if k != "reseed"}
        if params:
            _apply_randomization_layer(out, params, rng)
        return out


class RandomizedSimulator:
    """Wrapper around :class:`SynchronousSimulator` with domain randomization.

    When *randomization* is ``None`` (the default) the wrapper simply builds a
    ``SynchronousSimulator`` from *base_settings* and delegates every attribute
    access to it -- zero overhead, fully transparent.

    When a randomization config is provided, calling :meth:`randomize` samples
    new settings and reconfigures the inner simulator.

    Parameters
    ----------
    base_settings:
        Path to a YAML file **or** a raw settings ``dict``.
    randomization:
        Optional dict matching the ``domain_randomization`` schema (scenes,
        sensors, ``resample_every``, ``trajectory``).  ``None`` disables it.
    visualizer_disabled:
        Forwarded to ``SynchronousSimulator``.
    seed:
        Base entropy for the **deterministic** per-episode trajectory seed: each
        episode's trajectory is seeded from ``SeedSequence([seed, episode])``, so
        it is reproducible and independent of scene/sensor sampling. Distinct per
        producer (e.g. ``base_seed + worker_idx``) gives independent streams.
    """

    def __init__(
        self,
        base_settings: str | Path | dict[str, Any],
        randomization: dict[str, Any] | None = None,
        visualizer_disabled: bool = False,
        seed: int = 0,
    ):
        self._seed = int(seed)
        self._base_settings = self._load_settings(base_settings)
        self._rand_cfg: DomainRandomizationConfig | None = (
            DomainRandomizationConfig.from_dict(randomization)
            if randomization
            else None
        )
        self._viz_disabled = visualizer_disabled
        self.sim: SynchronousSimulator | None = None
        self._last_sampled_settings: dict[str, Any] | None = None

        self.build()

    @staticmethod
    def _load_settings(settings: str | Path | dict[str, Any]) -> dict[str, Any]:
        if isinstance(settings, dict):
            return copy.deepcopy(settings)
        path = Path(settings)
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def build(self) -> None:
        """(Re-)build the simulator from base settings without randomization."""
        if self.sim is not None:
            self.sim.close()
        settings = copy.deepcopy(self._base_settings)
        self._last_sampled_settings = settings
        self._episode = 0
        self.sim = SynchronousSimulator(
            settings, visualizer_disabled=self._viz_disabled
        )

    def randomize(self, rng: np.random.Generator) -> bool:
        """Advance one episode, resampling each part at its own cadence.

        Two fixed mechanisms (see :class:`DomainRandomizationConfig`):

        * **scene + sensors** — expensive; sampled and applied via
          ``reconfigure()`` only every ``resample_every`` episodes.
        * **trajectory** — cheap; if the sim has a ``trajectory`` block it is
          rebuilt **in-place every episode** via ``renew_trajectory`` (which also
          resets the sim clock for the new episode).

        Dynamics randomization is intentionally *not* handled here (it is
        object-level and owned by the RL vehicle abstraction).

        Args:
            rng: Generator driving the sampling.

        Returns:
            ``True`` iff a scene/sensor ``reconfigure`` happened this episode, so
            callers (e.g. the RL env) can re-derive spaces only when needed.
        """
        episode = self._episode
        self._episode += 1

        rebuilt = False
        if self._rand_cfg is not None and episode % self._rand_cfg.resample_every == 0:
            settings = self._rand_cfg.sample(self._base_settings, rng)
            self.sim.reconfigure(settings)
            self._last_sampled_settings = settings
            rebuilt = True

        if "trajectory" in self.sim.settings:
            # Deterministic per-episode trajectory: seed from (base_seed, episode)
            # so it is reproducible and decoupled from scene/sensor sampling on
            # the shared `rng`.
            traj_rng = np.random.default_rng(
                np.random.SeedSequence([self._seed, episode])
            )
            overrides = (
                self._rand_cfg.sample_trajectory(traj_rng)
                if self._rand_cfg is not None
                else {"seed": int(traj_rng.integers(0, 2**31 - 1))}
            )
            self.sim.renew_trajectory(overrides)

        return rebuilt

    @property
    def last_sampled_settings(self) -> dict[str, Any] | None:
        """The settings dict used to build the current simulator instance."""
        return self._last_sampled_settings

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
