"""Train a PPO policy for neurosim tasks using Stable-Baselines3.

Usage:
    python train_sb3.py \
        --experiment-config applications/rl/configs/hover_sb3_state_experiment.yaml

The script saves:
    outputs/rl/<run_name>/best_model.zip   - best checkpoint (by eval reward)
    outputs/rl/<run_name>/final_model.zip  - final checkpoint
    outputs/rl/<run_name>/vecnormalize.pkl - observation / reward normalizer state
    W&B run logs                    - training metrics and config

Notes for vectorized training:
    - `num_envs` controls PPO rollout collection parallelism.
    - `eval_freq` in config is interpreted in environment steps;
        converted to callback frequency so evaluation stays consistent.
    - Training uses subprocess vectorization for heavy simulators (Habitat).
    - Experiment configs are self-contained: scenes, sensors, ``dynamics``,
        the full ``simulator`` block, and domain randomization are in the YAML
        (no external settings file).  Training passes ``train=True`` into the RL
        env, which disables Rerun visualization regardless of YAML until
        short-episode logging is sorted out; use ``run_policy.py --visualize``
        for rollout logging.
    - Each vec-env worker is seeded with ``seed + env_idx`` so workers get
        distinct randomized configurations when DR is enabled.
    - ``simulator.domain_randomization.resample_every`` controls how often the
        Habitat-backed sim is rebuilt.  Optional dynamics DR lives under
        ``env.dynamics.domain_randomization`` (``enabled``, ``resample_every``,
        ``scales``).
    - Eval uses ``SubprocVecEnv`` (spawn) so Habitat teardown does not share a
        process with the training ``DummyVecEnv`` when ``num_envs == 1``.

Multi-GPU simulation:
    - ``n_gpus`` (int, default ``1``) uses physical GPU ids ``0 .. n_gpus-1``.
    - Worker placement uses a fixed skew (see :func:`sim_gpu_assignments`): with
      ``n`` envs and ``g > 1`` GPUs, ``max(0, n // g - 4)`` simulators sit on GPU
      0 and the remaining envs are split evenly across GPUs ``1 .. g - 1`` (so the
      ``4`` workers “moved off” the fair per-GPU share are absorbed by the other
      GPUs).  For ``g == 1`` every worker uses GPU 0.

Disk logs for each run: ``outputs/rl/<run>/logs/run_setup.yaml`` (layout),
``training.log`` (rollout timing), ``workers/{train,eval}_env_*.log`` (DR).
"""

import argparse
import sys
import copy
import numpy as np
import torch as th
from typing import Any
from pathlib import Path
from datetime import datetime

import wandb
from wandb.integration.sb3 import WandbCallback

from gymnasium import spaces
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from neurosim.core.utils.utils_gen import deep_update, load_yaml
from neurosim.core.utils import sim_gpu_assignments
from neurosim.rl import (
    AsymmetricActorCriticPolicy,
    AsymmetricRecurrentActorCriticPolicy,
    CombinedEventStateExtractor,
    EventCnnExtractor,
    env_class_for_task,
)
from neurosim.rl.sb3_features import PRIVILEGED_KEY
from neurosim.rl.disk_logging import (
    configure_training_disk_logger,
    gpu_assignment_summary,
    write_run_setup,
)


def load_experiment_config(config_path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    # Small curriculum configs can inherit a complete experiment and override
    # only the dimensions under test.  Keeping one source of truth avoids
    # silent drift in scene, dynamics, and sensor settings between phases.
    if "experiment_config" in cfg:
        base_path = Path(cfg["experiment_config"])
        if not base_path.is_absolute():
            base_path = Path.cwd() / base_path
        base = load_experiment_config(base_path)
        deep_update(base, cfg.get("overrides", {}))
        cfg = base

    required_keys = [
        "seed",
        "num_envs",
        "total_timesteps",
        "eval",
        "env",
        "ppo",
        "vecnormalize",
        "vec_env",
    ]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise ValueError(f"Missing required experiment config keys: {missing}")

    return cfg


def make_env(
    env_config: dict[str, Any],
    seed: int | None = None,
    env_idx: int = 0,
    *,
    train: bool = True,
    gpu_id: int = 0,
    worker_log_dir: Path | None = None,
    worker_log_role: str = "train",
):
    """Factory callable for DummyVecEnv / SubprocVecEnv.

    Each worker resets with ``seed + env_idx`` so that parallel workers
    get distinct initial randomizations (when DR is enabled in the config).

    ``train`` is forwarded to :class:`~neurosim.rl.env.BaseNeurosimRLEnv`; when
    true, Rerun visualization is forced off regardless of YAML.

    ``gpu_id`` assigns the Habitat simulator to a specific GPU, enabling
    multi-GPU simulation when workers are distributed across devices.

    ``worker_log_dir`` / ``worker_log_role`` enable per-env disk logs under
    ``<worker_log_dir>/workers/{role}_env_<idx>.log``.
    """

    def _init():
        import copy

        cfg = copy.deepcopy(env_config)
        cfg.setdefault("visual_backend", {})["gpu_id"] = gpu_id
        if worker_log_dir is not None:
            cfg["_neurosim_rl_worker_log_dir"] = str(worker_log_dir)
            cfg["_neurosim_rl_worker_log_role"] = worker_log_role
            cfg["_neurosim_rl_env_idx"] = env_idx
        env_cls = env_class_for_task(cfg["task"]["name"])
        env = env_cls(env_config=cfg, train=train)
        env = Monitor(env)
        if seed is not None:
            env.reset(seed=seed + env_idx)
        return env

    return _init


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SB3 PPO training for neurosim tasks")
    p.add_argument(
        "--experiment-config",
        type=str,
        default=None,
        help="YAML file containing experiment/training hyperparameters",
    )
    p.add_argument("--wandb-project", type=str, default="neurosim-rl")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Initialize PPO from a saved model (for example a BC warm start)",
    )
    p.add_argument(
        "--resume-vecnormalize",
        type=str,
        default=None,
        help=(
            "Load VecNormalize running obs/reward stats from this .pkl when "
            "resuming (e.g. the checkpoint's matching policy_vecnormalize_*.pkl). "
            "Without this a resumed run starts from fresh (mean 0, var 1) "
            "normalization stats even though the loaded policy was trained "
            "against near-converged ones."
        ),
    )
    p.add_argument(
        "--pretrained-encoder",
        type=str,
        default=None,
        help=(
            "Load pi_features_extractor/vf_features_extractor weights from "
            "a pretrain_encoder_oracle_vec.py checkpoint before training "
            "starts. Both towers are loaded (see the two-tower aux fix); "
            "unlike --bc-warm-start this seeds the VISUAL ENCODER, not the "
            "policy head, and works with any residual_control_mode since it "
            "never touches action-space-specific layers."
        ),
    )
    p.add_argument(
        "--bc-warm-start",
        type=str,
        default=None,
        help=(
            "Initialise the actor from a behavior-cloned checkpoint "
            "(train_velocity_dodge_bc.py --output ...pt). The critic stays "
            "at its initialisation; BC never trained one."
        ),
    )
    p.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging and WandbCallback",
    )
    return p.parse_args()


class TaskMetricsCallback(BaseCallback):
    """Surface the task's own reward terms to the SB3 logger.

    Without this nothing task-specific is logged at all -- only SB3's
    built-ins -- so the signals that actually show dodge learning are
    invisible. Two kinds are tracked differently:

    - per-step means (tracking error, how the correction splits between
      along-track and cross-track), and
    - per-episode finals (encounter counts and clear rate), which are
      running totals and only meaningful at episode end.

    ``encounter_clear_rate`` matters most early: episode ``success_rate``
    requires clearing *every* encounter, so with several throws per episode
    it stays at zero long after per-encounter skill starts improving.
    """

    STEP_MEAN_KEYS = (
        "pos_error",
        "correction_energy",
        "offset_cross_track",
        "boundary_margin_m",
    )
    STEP_ABS_MEAN_KEYS = ("offset_along_track",)
    EPISODE_FINAL_KEYS = (
        "peak_cross_in_threat",
        "peak_along_in_threat",
        "authority_used_in_threat",
        # Offset already held when each threat window opened, and the
        # fraction of the budget that leaves free. A dodge that starts
        # part-way to saturation cannot use its full range -- and if the
        # standing displacement is the wrong way, it has even less.
        "offset_at_threat_onset",
        "authority_free_at_onset",
        "encounter_clear_rate",
        "encounters_total",
        "encounters_cleared",
        "episode_min_clearance",
    )

    def __init__(self) -> None:
        super().__init__()
        self._step_sums: dict[str, float] = {}
        self._step_count = 0
        self._episode_values: dict[str, list[float]] = {}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []) or []:
            terms = info.get("reward_terms") or {}
            if not terms:
                continue
            self._step_count += 1
            for key in self.STEP_MEAN_KEYS:
                if key in terms and np.isfinite(terms[key]):
                    self._step_sums[key] = self._step_sums.get(key, 0.0) + float(
                        terms[key]
                    )
            for key in self.STEP_ABS_MEAN_KEYS:
                if key in terms and np.isfinite(terms[key]):
                    self._step_sums[key] = self._step_sums.get(key, 0.0) + abs(
                        float(terms[key])
                    )
            # Monitor injects "episode" only on the terminating step.
            if "episode" in info:
                for key in self.EPISODE_FINAL_KEYS:
                    if key in terms and np.isfinite(terms[key]):
                        self._episode_values.setdefault(key, []).append(
                            float(terms[key])
                        )
        return True

    def _on_rollout_end(self) -> None:
        if self._step_count:
            for key, total in self._step_sums.items():
                self.logger.record(f"task/{key}", total / self._step_count)
        for key, values in self._episode_values.items():
            if values:
                self.logger.record(f"task/{key}", float(np.mean(values)))
        self._step_sums.clear()
        self._step_count = 0
        self._episode_values.clear()


class SaveVecNormalizeCallback(BaseCallback):
    """Checkpoint the observation normaliser alongside the model.

    ``VecNormalize`` state is otherwise only written after ``learn()``
    returns, so for the entire duration of a long run every checkpoint on
    disk is unusable: loading the policy without its normaliser feeds it
    raw observations on a completely different scale than it trained on,
    and the resulting rollouts are not the trained behaviour. That silently
    invalidates any mid-run evaluation or video.
    """

    def __init__(self, vec_normalize, save_path: Path, save_freq: int):
        super().__init__()
        self._vec_normalize = vec_normalize
        self._save_path = Path(save_path)
        self._save_freq = max(int(save_freq), 1)

    def _on_step(self) -> bool:
        if self._vec_normalize is None:
            return True
        if self.n_calls % self._save_freq == 0:
            self._vec_normalize.save(str(self._save_path / "vecnormalize.pkl"))
        return True


class GateSupervisionCallback(BaseCallback):
    """Keep a learned threat gate honest against RL's pull to collapse it.

    ``gated_velocity_delta_integrated`` multiplies the correction by
    action[0], which hands PPO a single parameter that zeroes several cost
    terms at once (r_correction, r_along_track, r_no_threat_offset and most
    of r_track). v17 took exactly that shortcut: initialised from a clone
    whose gate matched the ground-truth threat on 100% of samples, PPO
    drove it shut on 100% of *threatened* steps within 900k steps and the
    policy went inert -- 8.6% against a 15% do-nothing control.

    A one-off BC initialisation cannot survive that pressure, so the gate
    is supervised for the whole run: after each rollout, a few gradient
    steps of binary cross-entropy pull action[0] toward the environment's
    own threat flag. The label rides in the privileged observation channel,
    which the actor is blinded to at forward time -- so this is a training
    signal, exactly like the privileged critic, and never an actor input.
    """

    def __init__(self, weight: float, steps: int, batch_size: int, threat_index: int):
        super().__init__()
        self._weight = float(weight)
        self._steps = int(steps)
        self._batch = int(batch_size)
        self._threat_index = int(threat_index)

    def _on_rollout_end(self) -> None:
        buf = self.model.rollout_buffer
        obs = buf.observations
        if not isinstance(obs, dict) or PRIVILEGED_KEY not in obs:
            return

        privileged = obs[PRIVILEGED_KEY].reshape(-1, obs[PRIVILEGED_KEY].shape[-1])
        labels_np = privileged[:, self._threat_index]
        # Stay on CPU here: the buffer holds n_steps * num_envs full-resolution
        # event frames (8192 x 2 x 480 x 640 is ~19 GiB), so materialising it
        # on the GPU OOMs instantly. Index first, transfer only the minibatch.
        flat_np = {k: v.reshape(-1, *v.shape[2:]) for k, v in obs.items()}
        n = labels_np.shape[0]
        if n == 0:
            return

        policy = self.model.policy
        total = 0.0
        rng = np.random.default_rng(self.model.num_timesteps)
        for _ in range(self._steps):
            idx = rng.integers(0, n, size=min(self._batch, n))
            batch = {
                k: th.as_tensor(v[idx], device=self.model.device)
                for k, v in flat_np.items()
            }
            labels = th.as_tensor(labels_np[idx], device=self.model.device).float()
            # get_distribution blinds the privileged channel for the actor,
            # so the gate is predicted from what the actor can really see.
            mean = policy.get_distribution(batch).distribution.mean
            loss = self._weight * th.nn.functional.binary_cross_entropy_with_logits(
                mean[:, 0], labels
            )
            policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(policy.parameters(), self.model.max_grad_norm)
            policy.optimizer.step()
            total += float(loss)

        self.logger.record("train/gate_supervision_loss", total / max(self._steps, 1))
        with th.no_grad():
            k_diag = min(256, n)
            sample = {
                k: th.as_tensor(v[:k_diag], device=self.model.device)
                for k, v in flat_np.items()
            }
            diag_labels = th.as_tensor(
                labels_np[:k_diag], device=self.model.device
            ).float()
            gate = policy.get_distribution(sample).distribution.mean[:, 0]
            self.logger.record("train/gate_mean", float(gate.mean()))
            self.logger.record(
                "train/gate_threat_agreement",
                float(((gate > 0).float() == diag_labels).float().mean()),
            )

    def _on_step(self) -> bool:
        return True


class ActorFreezeWarmupCallback(BaseCallback):
    """Hold the actor still until the critic can produce useful advantages.

    A behavior-cloned actor arrives with a policy worth keeping and a value
    head that knows nothing. PPO's first updates therefore push the actor
    with advantages computed from a near-random critic: measured on v15,
    ``explained_variance`` was -0.003 on the first update and needed ~8 to
    pass 0.7, during which the actor moved at ~0.01 KL per update. That was
    enough to walk the clone away before the critic could defend it -- v15
    fell to 10% success, below the 15% its own initialisation scored, and
    rebuilt the 0.25 m standing offset up to 0.68.

    Freezing the actor's parameters (and log_std) for the warmup lets the
    value head fit the cloned policy's own returns first. Gradients are
    disabled rather than zeroed so the optimizer cannot apply momentum to
    them either.
    """

    def __init__(
        self,
        updates: int,
        release_ev: float = 0.0,
        release_patience: int = 3,
        max_updates: int = 0,
    ):
        super().__init__()
        self._updates = int(updates)
        # Release on the CONDITION the update count was standing in for.
        # The message this callback prints -- "explained_variance now
        # usable" -- was never checked: at the default 15 updates EV
        # measured -0.041 (v20) and -0.008 (v21), i.e. the critic explained
        # nothing, so advantages were raw return noise and the first actor
        # updates walked the policy away from its starting point. Measured
        # across both runs, success fell 0.324 -> 0.189 (v20) and
        # 0.294 -> 0.264 (v21) at exactly the 30720-step release.
        #
        # release_ev 0 keeps the old pure-counter behaviour.
        self._release_ev = float(release_ev)
        self._release_patience = max(1, int(release_patience))
        # Hard cap so a critic that never reaches the bar still releases
        # rather than training an actor-frozen policy for the whole run.
        self._max_updates = int(max_updates)
        self._streak = 0
        self._frozen = False
        self._released = False

    def _actor_parameters(self):
        policy = self.model.policy
        groups = [
            getattr(policy, "pi_features_extractor", None),
            getattr(policy, "action_net", None),
        ]
        mlp = getattr(policy, "mlp_extractor", None)
        if mlp is not None:
            groups.append(getattr(mlp, "policy_net", None))
        for module in groups:
            if module is not None:
                yield from module.parameters()
        log_std = getattr(policy, "log_std", None)
        if log_std is not None:
            yield log_std

    def _set_actor_requires_grad(self, flag: bool) -> None:
        for param in self._actor_parameters():
            param.requires_grad_(flag)

    def _on_training_start(self) -> None:
        if self._updates > 0:
            self._set_actor_requires_grad(False)
            self._frozen = True
            print(f"actor frozen for the first {self._updates} updates")

    def _on_rollout_end(self) -> None:
        if not self._frozen or self._released:
            return
        # ``_n_updates`` counts gradient steps, not iterations.
        iterations = self.model._n_updates // max(self.model.n_epochs, 1)

        # EV from the previous train() call -- collect_rollouts runs before
        # train(), so this lags by one update. Harmless against a patience
        # window, and it is the only place SB3 exposes the value.
        ev = float(self.model.logger.name_to_value.get(
            "train/explained_variance", float("nan")
        ))
        if self._release_ev > 0.0 and ev == ev and ev >= self._release_ev:
            self._streak += 1
        else:
            self._streak = 0
        self.logger.record("train/actor_freeze_ev_streak", float(self._streak))

        reason = None
        if iterations < self._updates:
            pass  # minimum freeze budget not yet spent
        elif self._release_ev <= 0.0:
            reason = f"{iterations} updates (counter mode)"
        elif self._streak >= self._release_patience:
            reason = (
                f"explained_variance >= {self._release_ev:.3f} for "
                f"{self._streak} consecutive updates (now {ev:+.3f})"
            )
        elif self._max_updates and iterations >= self._max_updates:
            # The critic never got there. Say so plainly rather than
            # printing the success message on a fallback path.
            reason = (
                f"max_updates {self._max_updates} reached WITHOUT the critic "
                f"meeting explained_variance >= {self._release_ev:.3f} "
                f"(now {ev:+.3f}) -- advantages remain noisy"
            )

        if reason is not None:
            self._set_actor_requires_grad(True)
            self._released = True
            self.logger.record("train/actor_frozen", 0.0)
            print(f"actor released after {reason}", flush=True)
        else:
            self.logger.record("train/actor_frozen", 1.0)

    def _on_step(self) -> bool:
        return True


class CudaMemoryProbe(BaseCallback):
    """Localise where CUDA memory grows, instead of guessing at it.

    Seven OOMs were "fixed" seven ways -- n_steps 512/256/128/32, GPU
    placement, the BPTT window, freezing one feature extractor then both, and
    halving the event resolution -- and memory still climbed with success
    every time. Each was a real contributor and none was the dominant term.

    ``_on_rollout_start`` fires after ``train()``, ``_on_rollout_end`` fires
    before it, so the pair brackets both phases: comparing end-to-next-start
    isolates the optimiser step, and start-to-end isolates rollout collection
    plus the auxiliary update.
    """

    def __init__(self):
        super().__init__()
        self._last_end = None

    @staticmethod
    def _mib(x: float) -> float:
        return x / (1024.0 ** 2)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        if not th.cuda.is_available():
            return
        alloc = self._mib(th.cuda.memory_allocated())
        reserved = self._mib(th.cuda.memory_reserved())
        delta = "" if self._last_end is None else f" train_delta={alloc - self._last_end:+.0f}"
        print(f"[mem] post-train alloc={alloc:.0f}MiB reserved={reserved:.0f}MiB{delta}",
              flush=True)
        self.logger.record("mem/alloc_post_train_mib", alloc)

    def _on_rollout_end(self) -> None:
        if not th.cuda.is_available():
            return
        alloc = self._mib(th.cuda.memory_allocated())
        reserved = self._mib(th.cuda.memory_reserved())
        peak = self._mib(th.cuda.max_memory_allocated())
        self._last_end = alloc
        print(f"[mem] post-rollout alloc={alloc:.0f}MiB reserved={reserved:.0f}MiB "
              f"peak={peak:.0f}MiB", flush=True)
        self.logger.record("mem/alloc_post_rollout_mib", alloc)
        self.logger.record("mem/peak_mib", peak)
        th.cuda.reset_peak_memory_stats()


class DistanceAuxCallback(BaseCallback):
    """Supervise the actor's event encoder to localise obstacles.

    PPO's objective is indifferent to *how* the policy predicts a good action,
    so when a non-visual route exists the optimiser takes it -- the same
    dynamic that let behaviour cloning reach near-zero loss without using the
    camera. The distance map cannot be produced from proprioception, so this
    term can only be satisfied through vision.

    It is the one mechanism measured to work: with it, a cloned encoder
    reached held-out distance R2 = 0.334 on unseen episodes; without it, an
    otherwise identical run never left the noise band around zero.

    Trained alongside PPO on the rollout buffer rather than inside its loss,
    so the policy-gradient update is untouched.
    """

    def __init__(self, weight: float, bins: int = 12, batch: int = 64,
                 near_range_m: float = 0.0, occupancy_only: bool = False):
        super().__init__()
        self.weight = float(weight)
        self.bins = int(bins)
        self.batch = int(batch)
        # Only demand a prediction when the obstacle is close enough to
        # actually be resolvable. The target map runs to DISTANCE_MAX_M = 8 m,
        # but a 0.15 m sphere at 8 m subtends arcsin(0.15/8) ~ 1.07 deg --
        # under one 10 deg bin, about 6 px in a 640 px frame. Those samples
        # are unlearnable and MSE weights them the same as close-range ones
        # where the sphere is plainly visible, so the gradient is mostly
        # noise. Frames with NO obstacle are kept: "report clear" is both
        # learnable and necessary.
        self.near_range_m = float(near_range_m)
        # Segment, do not regress depth. The target map encodes distance/8 m
        # per bin, but depth from a single event time surface is ill-posed: a
        # near small sphere and a far large one give similar patterns, and
        # only their rate of change separates them. Asking a per-frame
        # decoder for it is close to unlearnable, which is the likeliest
        # reason R2 sat at -1.57 with weight 5.0. Binary occupancy -- WHICH
        # BEARINGS contain an obstacle -- is recoverable from one frame and
        # is all the aux term needs to do: force the encoder to localise the
        # obstacle instead of ignoring it. Depth, if wanted, is the recurrent
        # layer's job, not the CNN's.
        self.occupancy_only = bool(occupancy_only)
        self.stop_after = 0
        self._stopped = False
        # Both towers. share_features_extractor=False (see
        # AsymmetricActorCriticPolicy) gives the actor and critic separate
        # CNNs over the same event input. freeze_extractor_for_ppo freezes
        # BOTH for PPO's gradient, but this callback used to build its
        # optimiser over pi_features_extractor ONLY -- so once frozen, the
        # critic's visual encoder had no gradient source at all and sat at
        # random initialisation permanently, injecting noise into every value
        # estimate (and therefore every GAE advantage) for the rest of
        # training. Train both, with separate decoders since the towers are
        # deliberately not tied.
        self._TOWERS = ("pi_features_extractor", "vf_features_extractor")
        self._decoders: dict[str, th.nn.Module] = {}
        self._optimizers: dict[str, th.optim.Optimizer] = {}
        self.last_loss = float("nan")
        self.last_r2 = float("nan")
        self.last_r2_critic = float("nan")

    def _build(self) -> None:
        device = self.model.device
        for attr in self._TOWERS:
            extractor = getattr(self.model.policy, attr, None)
            if extractor is None or not hasattr(extractor, "forward_events"):
                continue
            decoder = th.nn.Sequential(
                th.nn.Linear(128, 64), th.nn.ReLU(), th.nn.Linear(64, self.bins)
            ).to(device)
            self._decoders[attr] = decoder
            self._optimizers[attr] = th.optim.Adam(
                list(decoder.parameters()) + list(extractor.parameters()), lr=3e-4
            )

    def _on_step(self) -> bool:
        return True

    def _train_tower(self, attr, obs, target):
        """One aux gradient step for one tower. Returns (mse, r2, occ_frac) or None."""
        extractor = getattr(self.model.policy, attr, None)
        decoder = self._decoders.get(attr)
        optimizer = self._optimizers.get(attr)
        if extractor is None or decoder is None:
            return None
        blinded = dict(obs)
        blinded[PRIVILEGED_KEY] = th.zeros_like(blinded[PRIVILEGED_KEY])
        # The extractor is frozen for PPO (see freeze_extractor_for_ppo) so
        # its forward builds no graph there and its activations are never
        # retained -- which is what kept OOMing a recurrent CNN. The aux term
        # is the thing that teaches it to see, so it turns gradients back on
        # for its own step only.
        frozen = [q for q in extractor.parameters() if not q.requires_grad]
        for q in frozen:
            q.requires_grad_(True)
        try:
            features = extractor.forward_events(blinded)
            if features is None:
                return None
            predicted = decoder(features)
            priv = obs[PRIVILEGED_KEY].detach()
            present = priv[:, 0] > 0.5
            if self.near_range_m > 0.0:
                distance = th.linalg.norm(priv[:, 1:4], dim=1)
                trainable = (~present) | (distance <= self.near_range_m)
            else:
                trainable = th.ones_like(present)
            if int(trainable.sum()) < 4:
                return None
            loss = self.weight * th.nn.functional.mse_loss(
                predicted[trainable], target[trainable]
            )
            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
        finally:
            # MUST be unconditional -- see the note on the leak this caused
            # when a `continue` from the near-range gate used to skip it.
            for q in frozen:
                q.requires_grad_(False)
        with th.no_grad():
            mse = float(((predicted - target) ** 2).mean())
            # Score only frames that actually contain an obstacle, on the
            # same population the loss trains on. See the note above the
            # class on why: the all-clear baseline has near-zero variance
            # otherwise, producing huge negative R2 regardless of accuracy.
            occupied = (target < 1.0).any(dim=1) & trainable
            occ_frac = float(occupied.float().mean())
            r2 = float("nan")
            if int(occupied.sum()) >= 4:
                t_occ, p_occ = target[occupied], predicted[occupied]
                base = ((th.ones_like(t_occ) - t_occ) ** 2).mean()
                occ_mse = ((p_occ - t_occ) ** 2).mean()
                if float(base) > 1e-9:
                    r2 = float(1.0 - occ_mse / base)
        return mse, r2, occ_frac

    def _on_rollout_end(self) -> None:
        if self.weight <= 0.0:
            return
        if self.stop_after > 0 and self.num_timesteps >= self.stop_after:
            if not self._stopped:
                self._stopped = True
                self._decoders.clear()
                self._optimizers.clear()
                th.cuda.empty_cache()
                print(f"[aux] stopping auxiliary supervision at "
                      f"{self.num_timesteps} steps; encoders frozen from here",
                      flush=True)
            return
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from train_velocity_dodge_bc import angular_distance_map  # noqa: PLC0415

        if not self._decoders:
            self._build()
        if not self._decoders:
            return

        buffer = self.model.rollout_buffer
        per_tower = {a: {"mse": [], "r2": [], "occ": []} for a in self._TOWERS}
        n_batches = 0
        for data in buffer.get(self.batch):
            obs = data.observations
            if not isinstance(obs, dict) or PRIVILEGED_KEY not in obs:
                break
            target = th.from_numpy(
                angular_distance_map(obs[PRIVILEGED_KEY].detach().cpu().numpy())
            ).to(self.model.device)
            if self.occupancy_only:
                # Keep the "1.0 == clear" convention so an empty scene still
                # maps to all ones; collapse every occupied bin to 0.0
                # regardless of range.
                target = (target >= 1.0).float()
            for attr in self._TOWERS:
                result = self._train_tower(attr, obs, target)
                if result is None:
                    continue
                mse, r2, occ = result
                per_tower[attr]["mse"].append(mse)
                per_tower[attr]["r2"].append(r2)
                per_tower[attr]["occ"].append(occ)
            n_batches += 1
            if n_batches >= 8:
                break

        pi = per_tower["pi_features_extractor"]
        vf = per_tower["vf_features_extractor"]
        if pi["mse"]:
            self.last_loss = float(np.mean(pi["mse"]))
            self.last_r2 = float(np.nanmean(pi["r2"])) if pi["r2"] else float("nan")
            self.logger.record("train/distance_mse", self.last_loss)
            self.logger.record("train/distance_r2", self.last_r2)
            self.logger.record(
                "train/distance_occupied_frac",
                float(np.mean(pi["occ"])) if pi["occ"] else 0.0,
            )
        if vf["mse"]:
            self.last_r2_critic = float(np.nanmean(vf["r2"])) if vf["r2"] else float("nan")
            self.logger.record("train/distance_r2_critic", self.last_r2_critic)


class PreTanhPenalty:
    """Hold the pre-tanh action mean away from the tanh rails.

    ``squash_output=True`` bounds the *action* but leaves the pre-tanh mean
    free, so nothing stops it from drifting outward. Measured on v24, the
    mean |mu| sat at 0.001-0.002 through 131k steps and then exploded to
    0.567 (max 3.661, 18.5% of components past 1) by 229k; success collapsed
    to 0.0% in the same window and train/entropy_loss crossed zero at 164k
    and climbed monotonically. v23 showed the identical trace and ended at
    +10.9. Past |mu| ~ 2 the tanh derivative is under 0.08, so the policy
    stops responding to the gradient at all.

    The fix is the standard SAC-style L2 on the pre-tanh mean. Rather than
    fork SB3's PPO.train() to add a term to its loss, this adds the penalty's
    gradient (d/dmu of lam * mean(mu^2) = 2 * lam * mu / N) straight onto the
    tensor during backward, which lands before clip_grad_norm_ and the
    optimizer step exactly as an in-loss term would.
    """

    def __init__(self, action_net: th.nn.Module, coef: float) -> None:
        self.coef = float(coef)
        self.abs_mean = 0.0
        self.frac_gt1 = 0.0
        self._max_abs = 0.0
        self._handle = action_net.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output: th.Tensor) -> None:
        with th.no_grad():
            self.abs_mean = float(output.abs().mean())
            self.frac_gt1 = float((output.abs() > 1.0).float().mean())
            self._max_abs = max(self._max_abs, float(output.abs().max()))
        # Rollout collection runs under no_grad; only the update pass has a
        # graph to attach to.
        if self.coef > 0.0 and output.requires_grad:
            n = output.shape[0]
            scale = 2.0 * self.coef / max(n, 1)
            # Detach and capture the VALUE before creating the closure, not
            # `output` itself. `output.register_hook(lambda g: ... output ...)`
            # closes over `output`, and register_hook stores that closure in
            # output's own hook list -- a reference cycle (output -> hooks ->
            # closure -> output) that Python's refcounting cannot collect;
            # only the periodic cyclic GC can, and it does not run
            # deterministically after every minibatch. Invisible with a small
            # CNN (tiny graphs), but with pi_features_extractor unfrozen and
            # a multi-million-parameter backbone in that graph, each
            # uncollected cycle holds several GB: measured a constant
            # +4.2 GB every update from the moment the actor released,
            # unbounded, which OOMed a run that had 15+ GB of headroom.
            # Capturing the already-detached value breaks the cycle -- it
            # has no grad_fn, so nothing links back to `output`.
            detached = output.detach()
            output.register_hook(lambda g, val=detached: g + scale * val)

    def pop_max(self) -> float:
        value = self._max_abs
        self._max_abs = 0.0
        return value

    def remove(self) -> None:
        self._handle.remove()


class PreTanhLogCallback(BaseCallback):
    """Surface the pre-tanh mean so the drift is visible in the log."""

    def __init__(self, penalty: PreTanhPenalty) -> None:
        super().__init__()
        self.penalty = penalty

    def _on_rollout_end(self) -> None:
        self.logger.record("train/pretanh_abs_mean", self.penalty.abs_mean)
        self.logger.record("train/pretanh_frac_gt1", self.penalty.frac_gt1)
        self.logger.record("train/pretanh_max_abs", self.penalty.pop_max())

    def _on_step(self) -> bool:
        return True


class EntCoefScheduleCallback(BaseCallback):
    """Linearly anneal ``model.ent_coef`` from ``initial`` to ``final``.

    SB3's ``ent_coef`` is a plain float, not a schedule like
    ``learning_rate``/``clip_range`` -- v3/v4 ran a combined ~1.2M steps at
    a constant 0.01 and ``log_std`` barely moved (-0.60 -> -0.67 over the
    back 600k), while deterministic (mean-action) eval success stayed flat
    near 0% despite a healthy stochastic-rollout encounter-clear rate: the
    entropy bonus was keeping exploration noise elevated indefinitely
    instead of letting the mean policy itself converge on a dodge.
    """

    def __init__(self, initial: float, final: float):
        super().__init__()
        self._initial = float(initial)
        self._final = float(final)

    def _on_step(self) -> bool:
        progress_remaining = self.model._current_progress_remaining
        self.model.ent_coef = (
            self._final + (self._initial - self._final) * progress_remaining
        )
        if self.n_calls % 2048 == 0:
            self.logger.record("train/ent_coef_current", self.model.ent_coef)
        return True


def build_policy_config(
    obs_mode: str,
    log_std_init: float,
    privileged: bool = False,
    event_presence_features: bool = False,
    event_high_resolution: bool = False,
    event_backbone: str = "small",
    recurrent: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Select the SB3 policy and kwargs for this observation layout.

    ``privileged=True`` means the env exposes a critic-only channel, which
    forces the dict-observation path and the asymmetric policy regardless
    of ``obs_mode``.
    """
    # Normalized action space [-1, 1] requires lower initial policy std
    # to prevent excessive early exploration.
    # squash_output puts a tanh on the action mean so the policy cannot leave
    # its own action box. Without it SB3 samples an unbounded Gaussian and the
    # *environment* clips: measured on v21, the lateral axis mean drifted to
    # 6.32 with std 0.52 -- about 10 sigma outside [-1, 1] -- so every sample
    # clipped to the same corner, all samples scored identically, and the
    # gradient w.r.t. the mean vanished. The policy was locked at |command|
    # = sqrt(2) with its sign changing on only 4% of in-threat steps, and no
    # action-magnitude penalty could pull it back (w_correction x10 moved
    # correction_energy 0.88 -> 0.89) because the penalty sees the already
    # clipped action. Requires use_sde=True, which this config sets.
    base_policy_kwargs = {
        "log_std_init": float(log_std_init),
        "squash_output": True,
    }
    event_extractor_kwargs = {
        "event_presence_features": bool(event_presence_features),
        "event_high_resolution": bool(event_high_resolution),
        "event_backbone": str(event_backbone),
    }

    if privileged:
        # The asymmetric critic and recurrence are independent: the actor is
        # blinded to the privileged channel either way, the LSTM only adds
        # temporal context that a single time surface cannot carry.
        return (
            AsymmetricRecurrentActorCriticPolicy
            if recurrent
            else AsymmetricActorCriticPolicy
        ), {
            "features_extractor_class": CombinedEventStateExtractor,
            "features_extractor_kwargs": event_extractor_kwargs,
            "normalize_images": False,
            **base_policy_kwargs,
        }

    if obs_mode == "state":
        return "MlpPolicy", base_policy_kwargs
    if obs_mode == "events":
        return "CnnPolicy", {
            "features_extractor_class": EventCnnExtractor,
            "features_extractor_kwargs": {
                "features_dim": 128,
                **event_extractor_kwargs,
            },
            "normalize_images": False,
            **base_policy_kwargs,
        }
    return "MultiInputLstmPolicy" if recurrent else "MultiInputPolicy", {
        "features_extractor_class": CombinedEventStateExtractor,
        "features_extractor_kwargs": {
            "features_dim": 192,
            **event_extractor_kwargs,
        },
        "normalize_images": False,
        **base_policy_kwargs,
    }


def build_vecnormalize_kwargs(
    obs_mode: str,
    normalize_obs: bool,
    normalize_reward: bool,
    training: bool,
    gamma: float | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "norm_obs": normalize_obs,
        "norm_reward": normalize_reward,
        "clip_obs": 10.0,
        "training": training,
    }
    # VecNormalize keeps a running DISCOUNTED-return estimate and divides
    # rewards by its standard deviation, so its gamma must match PPO's. It
    # was never passed, leaving SB3's default 0.99 against whatever the
    # config set. Return std scales roughly as 1/sqrt(1 - gamma**2), so at
    # PPO gamma 0.97 the default 0.99 over-estimates it by ~1.7x: value
    # targets come out ~1.7x too small and the value loss ~3x too small
    # against a fixed vf_coef, which slows value learning precisely when the
    # experiment is trying to measure whether the critic CAN learn.
    #
    # The mismatch was mild while configs used gamma ~0.99 (v16-v22 at 0.994
    # is a 1.3x error) and only became material with the horizon fix.
    if gamma is not None:
        kwargs["gamma"] = float(gamma)
    if normalize_obs and obs_mode == "events":
        # Event tensors are normalized in the RL env.
        kwargs["norm_obs"] = False
    elif normalize_obs and obs_mode == "combined":
        # Normalize privileged state only, not event frames.
        kwargs["norm_obs_keys"] = ["state"]
    return kwargs


def maybe_wrap_vecnormalize(
    vec_env,
    *,
    obs_mode: str,
    normalize_obs: bool,
    normalize_reward: bool,
    training: bool,
    enabled: bool | None = None,
    gamma: float | None = None,
):
    """Return ``VecNormalize(vec_env, ...)`` when normalization is active.

    ``enabled`` defaults to ``normalize_obs or normalize_reward``.  Eval envs
    pass an explicit ``enabled`` from the experiment vecnormalize block so a
    reward-only training setup still wraps eval (``norm_reward=False``) for SB3.
    """
    if enabled is None:
        enabled = normalize_obs or normalize_reward
    if not enabled:
        return vec_env
    return VecNormalize(
        vec_env,
        **build_vecnormalize_kwargs(
            obs_mode=obs_mode,
            normalize_obs=normalize_obs,
            normalize_reward=normalize_reward,
            training=training,
            gamma=gamma,
        ),
    )


def build_train_vec_env(
    env_fns: list,
    num_envs: int,
    vec_env_type: str,
    start_method: str,
):
    """Construct training VecEnv; use subprocess workers for heavy environments."""
    resolved_type = vec_env_type.strip().lower()
    if resolved_type not in {"auto", "dummy", "subproc"}:
        raise ValueError("vec_env_type must be one of: auto, dummy, subproc")

    if resolved_type == "dummy":
        return DummyVecEnv(env_fns)

    if resolved_type == "subproc" or (resolved_type == "auto" and num_envs > 1):
        resolved_start = start_method.strip().lower()
        if resolved_start not in {"fork", "forkserver", "spawn"}:
            raise ValueError(
                "vec_env_start_method must be one of: fork, forkserver, spawn"
            )
        return SubprocVecEnv(env_fns, start_method=resolved_start)

    return DummyVecEnv(env_fns)


def main():
    args = parse_args()
    exp = load_experiment_config(args.experiment_config)

    np.random.seed(int(exp["seed"]))
    num_envs = int(exp["num_envs"])
    n_steps = int(exp["ppo"]["n_steps"])
    batch_size = int(exp["ppo"]["batch_size"])
    rollout_batch = n_steps * num_envs
    if batch_size > rollout_batch:
        raise ValueError(
            f"batch_size ({batch_size}) must be <= n_steps * num_envs ({rollout_batch})"
        )
    if rollout_batch % batch_size != 0:
        print(
            "Warning: n_steps * num_envs is not divisible by batch_size; "
            "SB3 will use a truncated minibatch."
        )

    run_name = args.run_name
    if run_name is None:
        run_name = f"neurosim_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    output = Path("outputs/rl") / run_name
    output.mkdir(parents=True, exist_ok=True)

    # Training disk logger -------------------------------------------------------------------
    log_dir = output / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    train_disk_logger = configure_training_disk_logger(log_dir / "training.log")
    train_disk_logger.info(
        "run_start run_name=%s experiment_config=%s", run_name, args.experiment_config
    )

    run = None
    if not args.no_wandb:
        run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={"cli": vars(args), "experiment": exp},
            sync_tensorboard=True,
            save_code=True,
            dir=str(output),
        )

    vec_env_type = str(exp["vec_env"]["type"])
    vec_env_start_method = str(exp["vec_env"]["start_method"])

    env_config = dict(exp["env"])

    n_gpus = int(exp.get("n_gpus", 1))
    # GPU 0 also carries policy inference and the PPO update, so the right
    # number of simulators to leave on it depends on how big the network
    # input is -- the default skew was tuned for small observations and
    # oversubscribes GPU 0 at full resolution, leaving the other GPUs
    # stalled waiting on it. `envs_on_gpu0` overrides the heuristic.
    envs_on_gpu0 = exp.get("envs_on_gpu0")
    if envs_on_gpu0 is not None and n_gpus > 1:
        count0 = max(0, min(int(envs_on_gpu0), num_envs))
        train_gpu_assign = [0] * count0 + [
            1 + (i % (n_gpus - 1)) for i in range(num_envs - count0)
        ]
    else:
        train_gpu_assign = sim_gpu_assignments(num_envs, n_gpus)

    eval_num_envs = int(exp["eval"]["num_envs"])
    eval_gpu_assign = sim_gpu_assignments(eval_num_envs, n_gpus)
    eval_freq_passed = max(int(exp["eval"]["freq"]) // num_envs, 1)

    write_run_setup(
        log_dir,
        {
            "run_name": run_name,
            "experiment_config": str(args.experiment_config),
            "seed": int(exp["seed"]),
            "num_envs": num_envs,
            "n_gpus": n_gpus,
            "train_gpu_assignment": gpu_assignment_summary(train_gpu_assign),
            "eval": {
                "num_envs": eval_num_envs,
                **gpu_assignment_summary(eval_gpu_assign),
                "freq_yaml_total_timesteps": int(exp["eval"]["freq"]),
                "eval_callback_freq_per_vec_step": eval_freq_passed,
            },
            "ppo": {
                "n_steps": int(exp["ppo"]["n_steps"]),
                "batch_size": int(exp["ppo"]["batch_size"]),
                "n_epochs": int(exp["ppo"]["n_epochs"]),
                "device": str(exp["ppo"]["device"]),
            },
            "env": {
                "obs_mode": str(exp["env"]["obs_mode"]),
                "episode_seconds": float(exp["env"]["episode_seconds"]),
            },
            "simulator_domain_randomization": exp["env"]
            .get("simulator", {})
            .get("domain_randomization"),
            "logs_dir": str(log_dir.resolve()),
        },
    )

    # Training env (use multiprocessing for heavy simulators when num_envs > 1).
    # Each worker gets a distinct seed so domain randomization produces
    # different initial simulator configurations across workers.
    base_seed = int(exp["seed"])
    train_env_fns = [
        make_env(
            env_config,
            seed=base_seed,
            env_idx=env_idx,
            train=True,
            gpu_id=train_gpu_assign[env_idx],
            worker_log_dir=log_dir,
            worker_log_role="train",
        )
        for env_idx in range(num_envs)
    ]
    train_vec = build_train_vec_env(
        train_env_fns,
        num_envs=num_envs,
        vec_env_type=vec_env_type,
        start_method=vec_env_start_method,
    )
    vn = exp["vecnormalize"]
    train_vec = maybe_wrap_vecnormalize(
        train_vec,
        obs_mode=str(exp["env"]["obs_mode"]),
        normalize_obs=bool(vn["normalize_obs"]),
        normalize_reward=bool(vn["normalize_reward"]),
        training=True,
        gamma=float(exp["ppo"]["gamma"]),
    )
    if args.resume_vecnormalize:
        if not isinstance(train_vec, VecNormalize):
            raise ValueError(
                "--resume-vecnormalize given but normalization is disabled "
                "in this experiment config"
            )
        # Loading replaces the running obs/reward stats with the saved ones
        # but VecNormalize.load() reconstructs the wrapper around a fresh
        # venv, so re-point it at the venv we actually built above and
        # restore the live training flag load() doesn't preserve.
        train_vec = VecNormalize.load(args.resume_vecnormalize, train_vec.venv)
        train_vec.training = True
        print(f"resumed VecNormalize stats from {args.resume_vecnormalize}")

    # Eval env (separate instance, shared normalization stats)
    # Run eval in a subprocess so Habitat/OpenGL is not in the same process as a
    # training DummyVecEnv (avoids double GL context and Magnum teardown aborts).
    # In-loop eval deliberately keeps the same organic aim as training.
    # Aiming throws at the nominal trajectory would make a seed reproducible
    # across checkpoints, but the residual controller holds a standing
    # offset off that path, so it systematically throws where a dodging
    # drone is not: the same checkpoint and seeds scored 5/6 that way
    # against 0/8 with organic aim. That trades eval variance for a bias
    # toward exactly the behaviour being measured, so eval-vs-train
    # comparability wins here; beat the variance down with more episodes
    # instead (exp["eval"]["episodes"]).
    eval_env_config = copy.deepcopy(env_config)
    eval_vec = SubprocVecEnv(
        [
            make_env(
                eval_env_config,
                seed=base_seed + 1000,
                env_idx=env_idx,
                train=True,
                gpu_id=eval_gpu_assign[env_idx],
                worker_log_dir=log_dir,
                worker_log_role="eval",
            )
            for env_idx in range(eval_num_envs)
        ],
        start_method="spawn",
    )
    eval_vec = maybe_wrap_vecnormalize(
        eval_vec,
        obs_mode=str(exp["env"]["obs_mode"]),
        normalize_obs=bool(vn["normalize_obs"]),
        normalize_reward=False,
        training=False,
        enabled=bool(vn["normalize_obs"] or vn["normalize_reward"]),
        gamma=float(exp["ppo"]["gamma"]),
    )

    # Detect the critic-only channel from the built space rather than the
    # YAML, so it stays correct whatever the task decides to expose.
    privileged = (
        isinstance(train_vec.observation_space, spaces.Dict)
        and PRIVILEGED_KEY in train_vec.observation_space.spaces
    )
    recurrent = bool(exp["ppo"].get("recurrent", False))
    policy, policy_kwargs = build_policy_config(
        str(exp["env"]["obs_mode"]),
        float(exp["ppo"]["log_std_init"]),
        privileged=privileged,
        event_presence_features=bool(
            exp["ppo"].get("event_presence_features", False)
        ),
        event_high_resolution=bool(exp["ppo"].get("event_high_resolution", False)),
        event_backbone=str(exp["ppo"].get("event_backbone", "small")),
        recurrent=recurrent,
    )
    if privileged:
        print(
            f"asymmetric actor-critic: critic sees '{PRIVILEGED_KEY}', actor does not"
        )

    if args.resume_checkpoint:
        # PPO.load() restores every __init__ hyperparameter baked into the
        # checkpoint (SB3: model.__dict__.update(data) before applying our
        # kwargs) -- without passing the current config's values explicitly
        # here, a resume would silently keep whatever hyperparameters the
        # old run started with (e.g. sde_sample_freq=40 from v3) instead of
        # picking up changes made to the YAML before resuming.
        model = PPO.load(
            args.resume_checkpoint,
            env=train_vec,
            device=str(exp["ppo"]["device"]),
            tensorboard_log=str(output / "tensorboard"),
            print_system_info=False,
            learning_rate=float(exp["ppo"]["learning_rate"]),
            n_steps=int(exp["ppo"]["n_steps"]),
            batch_size=int(exp["ppo"]["batch_size"]),
            n_epochs=int(exp["ppo"]["n_epochs"]),
            gamma=float(exp["ppo"]["gamma"]),
            gae_lambda=float(exp["ppo"]["gae_lambda"]),
            clip_range=float(exp["ppo"]["clip_range"]),
            ent_coef=float(exp["ppo"]["ent_coef"]),
            vf_coef=float(exp["ppo"]["vf_coef"]),
            max_grad_norm=float(exp["ppo"]["max_grad_norm"]),
            target_kl=(
                None
                if exp["ppo"].get("target_kl") is None
                else float(exp["ppo"]["target_kl"])
            ),
            use_sde=bool(exp["ppo"].get("use_sde", False)),
            sde_sample_freq=int(exp["ppo"].get("sde_sample_freq", -1)),
        )
        print(f"resumed PPO parameters from {args.resume_checkpoint}")
        if bool(exp["ppo"].get("reset_log_std_on_resume", False)):
            # log_std is a learned parameter, so it comes back from the
            # checkpoint already converged (v7 finished at -0.85, std 0.43).
            # When a resume deliberately changes the reward or the scene,
            # that narrow distribution can no longer explore the behaviour
            # the new landscape rewards, and log_std_init is otherwise
            # silently ignored on this path.
            log_std_init = float(exp["ppo"]["log_std_init"])
            with th.no_grad():
                model.policy.log_std.fill_(log_std_init)
            if model.use_sde:
                model.policy.reset_noise()
            print(f"reset policy log_std to {log_std_init}")
    else:
        # Recurrence, not frame stacking. A single time surface shows WHERE
        # edges fired, not how fast the obstacle is closing, so distance and
        # time-to-impact are not recoverable from one frame. Stacking buys a
        # fixed window at 3x the observation size -- which forced n_steps
        # 512 -> 128 to fit the rollout buffer -- while an LSTM carries
        # unbounded history in a few hundred floats at no buffer cost. The
        # event-camera reference (arXiv 2603.07578) inserts two recurrent
        # layers for the same reason.
        ctor = RecurrentPPO if recurrent else PPO
        extra = {}
        if recurrent:
            extra["policy_kwargs"] = dict(
                policy_kwargs,
                lstm_hidden_size=int(exp["ppo"].get("lstm_hidden_size", 128)),
                n_lstm_layers=int(exp["ppo"].get("n_lstm_layers", 1)),
            )
        model = ctor(
            policy,
            train_vec,
            learning_rate=float(exp["ppo"]["learning_rate"]),
            n_steps=int(exp["ppo"]["n_steps"]),
            batch_size=int(exp["ppo"]["batch_size"]),
            n_epochs=int(exp["ppo"]["n_epochs"]),
            gamma=float(exp["ppo"]["gamma"]),
            gae_lambda=float(exp["ppo"]["gae_lambda"]),
            clip_range=float(exp["ppo"]["clip_range"]),
            ent_coef=float(exp["ppo"]["ent_coef"]),
            vf_coef=float(exp["ppo"]["vf_coef"]),
            max_grad_norm=float(exp["ppo"]["max_grad_norm"]),
            # Hard per-update cap on policy drift; None disables it.
            target_kl=(
                None
                if exp["ppo"].get("target_kl") is None
                else float(exp["ppo"]["target_kl"])
            ),
            # Independent per-step Gaussian exploration gets almost entirely
            # cancelled by the residual-control leaky integrator: a leaky
            # integrator (time constant tau) driven by i.i.d. noise resampled
            # every dt reaches only ~sqrt(dt / 2*tau) of the displacement a
            # *sustained* action of the same magnitude would -- with the 50 Hz
            # policy rate and offset_tau_s=0.8s that's ~11% of nominal reach,
            # matching the ~0.06-0.09m peak in-threat offsets measured against
            # a 0.56m budget. SDE holds exploration noise correlated across
            # sde_sample_freq steps instead of resampling it independently
            # every step, so it survives the integrator's averaging instead
            # of being filtered out by it.
            use_sde=bool(exp["ppo"].get("use_sde", False)),
            sde_sample_freq=int(exp["ppo"].get("sde_sample_freq", -1)),
            seed=int(exp["seed"]),
            device=str(exp["ppo"]["device"]),
            verbose=1,
            tensorboard_log=str(output / "tensorboard"),
            **(extra if recurrent else {"policy_kwargs": policy_kwargs}),
        )

    if bool(exp["ppo"].get("freeze_extractor_for_ppo", False)):
        # Keep the event CNN out of PPO's autograd graph. With a recurrent
        # policy, BPTT retains every timestep's convolutional activations,
        # and memory therefore grows as episodes lengthen -- the learner
        # climbed 4.8 -> 24 GB as success rose and died in backward, five
        # times over. Frozen parameters with non-grad inputs build no graph
        # at all, so those activations are never allocated.
        #
        # The encoder is still trained, by the auxiliary occupancy term,
        # which re-enables gradients for its own step. This mirrors the
        # reference (arXiv 2603.07578), where the encoder is trained by
        # auxiliary supervision and the decoder discarded afterwards.
        # BOTH towers. share_features_extractor=False means the critic has
        # its own CNN over the same (2,480,640) input, so freezing only the
        # actor's left half the activations in the graph -- memory still
        # climbed 4.8 -> 16.6 GB. The critic also receives the privileged
        # channel, which is the informative part of its input anyway.
        n_frozen = 0
        for extractor_attr in ("pi_features_extractor", "vf_features_extractor"):
            extractor_module = getattr(model.policy, extractor_attr, None)
            if extractor_module is None:
                continue
            for q in extractor_module.parameters():
                q.requires_grad_(False)
                n_frozen += 1
        # Freezing alone is not enough: the parameters stay in PPO's
        # optimizer, and the auxiliary term re-enables their gradients for its
        # own step, so PPO's optimizer then sees .grad on tensors it believes
        # it owns -- allocating Adam state for them and updating them anyway.
        # Measured: ~+282 MiB retained per update from the moment the actor
        # unfroze, which is what OOMed seven runs. Rebuild the optimizer over
        # trainable parameters only.
        trainable = [q for q in model.policy.parameters() if q.requires_grad]
        model.policy.optimizer = model.policy.optimizer_class(
            trainable,
            lr=model.lr_schedule(1.0),
            **model.policy.optimizer_kwargs,
        )
        print(f"event extractors frozen for PPO ({n_frozen} tensors); "
              f"optimizer rebuilt over {len(trainable)} trainable tensors; "
              "auxiliary supervision still trains the encoders")

    if args.pretrained_encoder:
        ckpt = th.load(args.pretrained_encoder, map_location=str(exp["ppo"]["device"]))
        loaded = []
        for attr in ("pi_features_extractor", "vf_features_extractor"):
            extractor = getattr(model.policy, attr, None)
            if extractor is None or attr not in ckpt:
                continue
            extractor.load_state_dict(ckpt[attr])
            loaded.append(attr)
        print(
            f"pretrained encoder loaded from {args.pretrained_encoder} "
            f"({ckpt.get('env_steps', '?')} pretraining env-steps): {loaded}"
        )

    if args.bc_warm_start:
        from bc_warm_start import load_clone_into_policy

        clone = th.load(args.bc_warm_start, map_location=str(exp["ppo"]["device"]))
        report = load_clone_into_policy(model.policy, clone["model"])
        print(
            f"warm-started actor from {args.bc_warm_start}: "
            f"{len(report['updated'])} tensors loaded, "
            f"{len(report['skipped'])} skipped"
        )
        for s_ in report["skipped"]:
            print("  skipped:", s_)

    eval_callback = EvalCallback(
        eval_vec,
        best_model_save_path=str(output),
        log_path=str(output),
        eval_freq=eval_freq_passed,
        n_eval_episodes=int(exp["eval"]["episodes"]),
        deterministic=True,
    )

    learn_callbacks = [
        eval_callback,
        TaskMetricsCallback(),
        # EvalCallback only retains the checkpoint with the highest mean
        # reward. Success can improve while mean reward moves the other way,
        # so retaining every evaluation-window policy is necessary for
        # success-gated video review and avoids losing an informative policy
        # in --no-wandb runs.
        CheckpointCallback(
            save_freq=eval_freq_passed,
            save_path=str(output / "checkpoints"),
            name_prefix="policy",
            save_vecnormalize=isinstance(train_vec, VecNormalize),
            verbose=1,
        ),
        # Written at the same cadence as eval so every checkpoint on
        # disk has a matching normaliser and is actually loadable.
        SaveVecNormalizeCallback(
            train_vec if isinstance(train_vec, VecNormalize) else None,
            output,
            eval_freq_passed,
        ),
    ]
    gate_w = float(exp["ppo"].get("gate_supervision_weight", 0.0))
    if gate_w > 0.0:
        from neurosim.rl.tasks.velocity_dodge import PRIVILEGED_THREAT_INDEX

        learn_callbacks.append(
            GateSupervisionCallback(
                weight=gate_w,
                steps=int(exp["ppo"].get("gate_supervision_steps", 8)),
                batch_size=int(exp["ppo"].get("gate_supervision_batch", 256)),
                threat_index=PRIVILEGED_THREAT_INDEX,
            )
        )

    freeze_updates = int(exp["ppo"].get("actor_freeze_updates", 0))
    if freeze_updates > 0:
        release_ev = float(exp["ppo"].get("actor_release_explained_variance", 0.0))
        learn_callbacks.append(
            ActorFreezeWarmupCallback(
                freeze_updates,
                release_ev=release_ev,
                release_patience=int(
                    exp["ppo"].get("actor_release_patience", 3)
                ),
                max_updates=int(exp["ppo"].get("actor_freeze_max_updates", 0)),
            )
        )
        if release_ev > 0.0:
            print(
                f"actor release gated on explained_variance >= {release_ev} "
                f"(min {freeze_updates} updates, cap "
                f"{exp['ppo'].get('actor_freeze_max_updates', 0)})"
            )

    # Always installed: at coef 0.0 it only logs, so the drift stays visible
    # even on runs that do not regularise it.
    pretanh_penalty = PreTanhPenalty(
        model.policy.action_net,
        float(exp["ppo"].get("pretanh_penalty", 0.0)),
    )
    learn_callbacks.append(PreTanhLogCallback(pretanh_penalty))

    def _aux_cb(weight, near_range_m, occupancy_only, stop_after):
        cb = DistanceAuxCallback(
            weight, near_range_m=near_range_m, occupancy_only=occupancy_only
        )
        cb.stop_after = int(stop_after)
        return cb

    distance_aux = float(exp["ppo"].get("distance_aux_weight", 0.0))
    if distance_aux > 0.0:
        near_range = float(exp["ppo"].get("distance_aux_near_range_m", 0.0))
        occupancy_only = bool(exp["ppo"].get("distance_aux_occupancy_only", False))
        learn_callbacks.append(
            _aux_cb(
                distance_aux,
                near_range_m=near_range,
                occupancy_only=occupancy_only,
                stop_after=int(exp["ppo"].get("distance_aux_stop_after_steps", 0)),
            )
        )
        print(
            f"auxiliary supervision active: weight={distance_aux}"
            + (", target=occupancy" if occupancy_only else ", target=distance")
            + (f", only within {near_range} m" if near_range > 0 else "")
        )
    # Last, so its post-rollout reading is taken AFTER the auxiliary update.
    # With the probe first, train_delta bundled the aux step together with
    # PPO's train(); ordering it last splits the two.
    learn_callbacks.append(CudaMemoryProbe())
    if pretanh_penalty.coef > 0.0:
        print(f"pre-tanh mean penalty active: coef={pretanh_penalty.coef}")

    ent_coef_final = exp["ppo"].get("ent_coef_final")
    if ent_coef_final is not None:
        learn_callbacks.append(
            EntCoefScheduleCallback(
                initial=float(exp["ppo"]["ent_coef"]),
                final=float(ent_coef_final),
            )
        )
    if not args.no_wandb:
        learn_callbacks.append(
            WandbCallback(
                gradient_save_freq=0,
                model_save_path=str(output / "wandb_models"),
                model_save_freq=eval_freq_passed,
                verbose=2,
            )
        )

    model.learn(
        total_timesteps=int(exp["total_timesteps"]),
        callback=learn_callbacks,
    )
    train_disk_logger.info(
        "training_complete total_timesteps=%s", int(exp["total_timesteps"])
    )

    model.save(str(output / "final_model"))
    if isinstance(train_vec, VecNormalize):
        train_vec.save(str(output / "vecnormalize.pkl"))

    print(f"Training complete. Artifacts saved to {output}")

    eval_vec.close()
    train_vec.close()

    if run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
