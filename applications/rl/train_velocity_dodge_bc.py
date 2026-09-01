"""Behavior-clone the successful scripted velocity-dodge oracle.

PPO on this task plateaus at roughly 13% of the achievable headroom
(control 15% / oracle 50% success). Ablations show the actor does read the
event frames, and exploration variance stays healthy, but it never learns
*which way* to go for a specific obstacle -- it settles on a generic
standing displacement that beats passivity slightly. The scripted expert
already encodes that direction mapping, so cloning it supplies exactly
what RL is failing to extract.

Two augmentations here exist to attack failure modes measured on the RL
runs, not as generic regularisation:

- **Zero-event counterfactuals.** Every trained policy converged on a
  permanent ~0.6 m offset held even with obstacles *disabled* -- it never
  learned that an empty visual field means "return to the path". Pairing
  each threat sample with a copy whose event frames are blank and whose
  label is the zero action supervises that directly.
- **Virtual onsets.** The state observation includes the previous action,
  which is strongly autocorrelated during a dodge. A policy can score well
  on imitation by continuing whatever it was already doing, learning
  action continuity instead of perception. Onset copies blank the
  previous-action channel so the reaction has to come from what is seen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn


# tanh(2.65) ~ 0.99, so the expert's saturated +-1 commands stay reachable
# at a pre-tanh mean well inside the region where the gradient survives.
SQUASH_TARGET_LIMIT = 0.99

# VelocityDodgeTask's flat state is
# [v_body, v_ref_body, pos_err_body, offset_body, omega_body, prev_action,
#  obstacle slots...]. Unlike the event/dict layout, prev_action is therefore
# not at the end once privileged obstacle features are appended.
FLAT_PREVIOUS_ACTION_START = 15


DISCRETE_DEADBAND = 0.15

# Following Messikommer et al. (arXiv:2603.07578), whose event encoder is
# supervised with a forward-facing 1-D angular distance map rather than by
# actions alone. That detail is the one our setup was missing: a distance map
# cannot be predicted from vehicle state, so unlike action regression -- which
# state predicts at R2 = 0.99 -- it forces the encoder to localise obstacles
# in the event stream or fail outright.
DISTANCE_BINS = 12
DISTANCE_FOV_DEG = 120.0
DISTANCE_MAX_M = 8.0


def angular_distance_map(
    privileged,
    bins: int = DISTANCE_BINS,
    fov_deg: float = DISTANCE_FOV_DEG,
    max_range: float = DISTANCE_MAX_M,
    obstacle_radius_m: float = 0.15,
):
    """Body-frame obstacle geometry -> normalised 1-D distance map.

    ``privileged`` is [present, rel_pos(3), rel_vel(3), clearance, tca] in the
    body frame, where component 0 of rel_pos is forward and 1 is lateral.
    Returns 1.0 for "clear to max_range" down to 0.0 for "touching", so an
    empty scene maps to all ones.
    """
    array = np.atleast_2d(np.asarray(privileged, dtype=np.float32))
    out = np.ones((len(array), bins), dtype=np.float32)
    half = np.deg2rad(fov_deg) / 2.0
    edges = np.linspace(-half, half, bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    for i, row in enumerate(array):
        if row[0] < 0.5:
            continue
        forward, lateral = float(row[1]), float(row[2])
        distance = float(np.linalg.norm(row[1:4]))
        if distance <= 1e-6 or distance > max_range:
            continue
        bearing = np.arctan2(lateral, forward)
        # Angular half-width the sphere subtends at this range.
        half_width = float(np.arcsin(np.clip(obstacle_radius_m / distance, 0.0, 1.0)))
        covered = np.abs(centres - bearing) <= max(half_width, np.deg2rad(fov_deg) / bins / 2)
        out[i, covered] = distance / max_range

    return out[0] if np.ndim(privileged) == 1 else out



def discretize_lateral(actions, deadband: float = DISCRETE_DEADBAND):
    """Map a lateral command to {0: left, 1: none, 2: right}.

    Regression asks the network for a magnitude; what actually decides
    whether the vehicle survives is which side it goes. On a weak input a
    categorical choice is far easier to learn than a continuous target --
    cross-entropy keeps a strong gradient on confidently-wrong samples where
    MSE's gradient shrinks with the residual, and the class boundary sits
    exactly at the decision that matters. With lateral_axis_only there is
    only one axis to decide about, so three classes cover the space.
    """
    array = np.asarray(actions, dtype=np.float32)
    # Component 1 is the lateral axis for both a single (3,) action and a
    # batched (N, 3) block. Testing ndim > 1 mistook a single action's three
    # components for three separate samples and produced three labels.
    lateral = array[..., 1] if array.ndim >= 1 and array.shape[-1] >= 2 else array
    labels = np.ones(lateral.shape, dtype=np.int64)
    labels[lateral < -deadband] = 0
    labels[lateral > deadband] = 2
    return labels


def pool_events(events, factor: int):
    """Average-pool an event time surface by ``factor``; identity at 1."""
    if factor <= 1:
        return events
    c, h, w = events.shape[-3:]
    trimmed = events[..., : h // factor * factor, : w // factor * factor]
    return trimmed.reshape(
        *trimmed.shape[:-3], c, h // factor, factor, w // factor, factor
    ).mean(axis=(-3, -1))


def gated_imitation_loss(
    predicted: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Binary threat gate plus direction regression on threat states only.

    The policy's first Gaussian mean is treated as a gate logit. At
    inference SB3 clips it to the action space and the environment clips it
    to [0, 1]. Quiet samples therefore supervise only whether to act, not
    an arbitrary unused direction vector.
    """
    if predicted.shape[-1] != 4 or target.shape[-1] != 4:
        raise ValueError("gated imitation expects [gate, dx, dy, dz] actions")

    gate_target = target[:, 0].clamp(0.0, 1.0)
    gate_loss = nn.functional.binary_cross_entropy_with_logits(
        predicted[:, 0], gate_target
    )

    active = gate_target > 0.5
    if bool(active.any()):
        direction_loss = nn.functional.mse_loss(
            predicted[active, 1:], target[active, 1:]
        )
    else:
        direction_loss = predicted.sum() * 0.0

    return gate_loss + direction_loss, gate_loss, direction_loss


def imitation_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Plain regression for the ungated (3-dim) residual action space.

    ``velocity_delta_integrated`` has no gate channel: the action *is* the
    body-frame velocity delta, and the expert already emits exactly zero
    when it has no plan. So "do nothing when nothing is inbound" is
    expressible as an ordinary regression target rather than needing a
    separate gate head -- quiet samples supervise the zero action directly.

    Reported split by threat/quiet so the two can be watched separately;
    they behave very differently and a single averaged number hides which
    one is being learned.
    """
    if predicted.shape[-1] != target.shape[-1]:
        raise ValueError("predicted/target action dimension mismatch")

    per_sample = ((predicted - target) ** 2).mean(dim=-1)
    if weights is not None:
        loss = (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)
    else:
        loss = per_sample.mean()

    acting = target.norm(dim=-1) > 0.1
    quiet = ~acting
    threat_loss = (
        per_sample[acting].mean() if bool(acting.any()) else predicted.sum() * 0.0
    )
    quiet_loss = (
        per_sample[quiet].mean() if bool(quiet.any()) else predicted.sum() * 0.0
    )
    return loss, threat_loss, quiet_loss


def balanced_sample_weights(
    threat: np.ndarray, counterfactual: np.ndarray
) -> np.ndarray:
    """Equalise threat / quiet / counterfactual mass.

    Left to its natural frequency the dodging behaviour is a minority of
    samples, and a regressor minimises loss fastest by predicting the quiet
    action everywhere -- which is precisely the passive policy we already
    have. Half the weight goes to genuine threat samples, the rest split
    between quiet and counterfactual.
    """
    threat = np.asarray(threat, dtype=bool)
    counterfactual = np.asarray(counterfactual, dtype=bool)
    real_threat = threat & ~counterfactual
    quiet = ~threat & ~counterfactual

    if not real_threat.any() or not quiet.any():
        raise ValueError("dataset must contain both threat and non-threat samples")

    weights = np.empty(len(threat), dtype=np.float64)
    # State-only privileged datasets have no event image to blank, so a
    # "zero-event" counterfactual would pair unchanged obstacle geometry with
    # an incorrect zero label. In that case balance genuine threat and quiet
    # states equally. Event datasets retain the historical 50/25/25 split.
    groups = (
        ((real_threat, 0.5), (quiet, 0.25), (counterfactual, 0.25))
        if counterfactual.any()
        else ((real_threat, 0.5), (quiet, 0.5))
    )
    for mask, share in groups:
        n = int(mask.sum())
        weights[mask] = (share / n) if n else 0.0
    return weights


class ObservationDataset(torch.utils.data.Dataset):
    """In-memory dict-observation dataset, for small collections and tests."""

    def __init__(self, observations: list[dict[str, np.ndarray]], actions: np.ndarray):
        self._observations = observations
        self._actions = np.asarray(actions, dtype=np.float32)

    def __len__(self) -> int:
        return len(self._actions)

    def __getitem__(self, index: int):
        return self._observations[index], self._actions[index]


class HDF5ObservationDataset(torch.utils.data.Dataset):
    """Lazily-read imitation samples plus their derived augmentations.

    Nothing is decompressed until ``__getitem__``; the expert collections
    are far larger than RAM at full event resolution. The augmented samples
    are stored as index maps rather than materialised copies, so the memory
    cost of doubling the dataset is a couple of integer arrays.
    """

    def __init__(
        self,
        paths: Sequence[str],
        *,
        gated_onset_window: int = 0,
        zero_event_counterfactuals: bool = False,
        gated_action_loss: bool = False,
        blank_previous_action: bool = False,
        events_only_state: bool = False,
        blank_tracking_error: bool = False,
        discrete_actions: bool = False,
        downsample_events: int = 1,
        holdout_fraction: float = 0.0,
        holdout_side: str = "train",
        holdout_seed: int = 0,
    ):
        import h5py

        self._files = [h5py.File(p, "r") for p in paths]
        self._gated_action_loss = bool(gated_action_loss)
        self._blank_previous_action = bool(blank_previous_action)
        self._events_only_state = bool(events_only_state)
        self._blank_tracking_error = bool(blank_tracking_error)
        self._downsample_events = max(int(downsample_events), 1)
        self._discrete_actions = bool(discrete_actions)

        # Flatten (file, row) addressing across every collection file.
        self._index: list[tuple[int, int]] = []
        threat_all: list[np.ndarray] = []
        for fi, f in enumerate(self._files):
            n = len(f["actions"])
            self._index.extend((fi, i) for i in range(n))
            threat_all.append(np.asarray(f["threat"][:], dtype=bool))
        self._threat = np.concatenate(threat_all) if threat_all else np.zeros(0, bool)

        # Split by *episode*, never by frame. Frames inside one episode are
        # a few milliseconds apart and near-identical, so a frame-level split
        # leaks almost the whole test set into training and any held-out
        # number it produces is meaningless.
        if holdout_fraction > 0.0:
            keys = []
            for fi, f in enumerate(self._files):
                keys.append(
                    np.array(
                        [f"{fi}:{e}" for e in np.asarray(f["episode_index"][:])]
                    )
                )
            keys = np.concatenate(keys)
            unique = np.unique(keys)
            rng = np.random.default_rng(holdout_seed)
            shuffled = rng.permutation(unique)
            n_test = max(int(round(len(unique) * holdout_fraction)), 1)
            test_eps = set(shuffled[:n_test].tolist())
            in_test = np.array([k in test_eps for k in keys])
            keep = in_test if holdout_side == "test" else ~in_test
            # _find_onsets addresses the *unfiltered* flat array, so keep a
            # map from original position to filtered position (-1 = dropped)
            # and remap through it. Without this the onset list points past
            # the end of the shrunken index.
            self._keep_map = np.full(len(keep), -1, dtype=np.int64)
            self._keep_map[np.flatnonzero(keep)] = np.arange(int(keep.sum()))
            self._index = [ix for ix, k in zip(self._index, keep) if k]
            self._threat = self._threat[keep]
            self._split_episodes = (len(unique) - n_test, n_test)
        else:
            self._keep_map = None
            self._split_episodes = None

        self._action_dim = int(self._files[0]["actions"].shape[1])
        self._event_key = (
            "observations_events"
            if "observations_events" in self._files[0]
            else None
        )
        self._state_key = (
            "observations_state" if "observations_state" in self._files[0] else None
        )
        # Carried so batches match the actor's observation space; the value
        # is zeroed before use, mirroring AsymmetricActorCriticPolicy.
        self._privileged_key = (
            "observations_privileged"
            if "observations_privileged" in self._files[0]
            else None
        )
        self._flat_key = "observations" if "observations" in self._files[0] else None
        if not any(
            (self._event_key, self._state_key, self._privileged_key, self._flat_key)
        ):
            raise ValueError("dataset contains no supported observation array")

        # Augmentations, addressed by base index rather than copied.
        self._onset: list[int] = []
        if gated_onset_window > 0:
            self._onset = self._find_onsets(gated_onset_window)
        self._counterfactual: list[int] = []
        if zero_event_counterfactuals and self._event_key:
            self._counterfactual = list(np.flatnonzero(self._threat))

    def _find_onsets(self, window: int) -> list[int]:
        """Threat samples within ``window`` steps of a rising threat edge."""
        onsets: list[int] = []
        offset = 0
        for f in self._files:
            threat = np.asarray(f["threat"][:], dtype=bool)
            episode = np.asarray(f["episode_index"][:])
            for ep in np.unique(episode):
                rows = np.flatnonzero(episode == ep)
                t = threat[rows]
                rising = np.flatnonzero(t & ~np.concatenate(([False], t[:-1])))
                for r in rising:
                    for k in range(window):
                        if r + k < len(rows) and t[r + k]:
                            onsets.append(offset + int(rows[r + k]))
            offset += len(threat)
        if self._keep_map is not None:
            onsets = [
                int(self._keep_map[o])
                for o in onsets
                if self._keep_map[o] >= 0
            ]
        return onsets

    def __len__(self) -> int:
        return len(self._index) + len(self._onset) + len(self._counterfactual)

    @property
    def sample_kinds(self) -> tuple[np.ndarray, np.ndarray]:
        """(threat, counterfactual) masks aligned with ``__getitem__`` order."""
        base_threat = self._threat
        onset_threat = np.ones(len(self._onset), dtype=bool)
        cf_threat = np.ones(len(self._counterfactual), dtype=bool)
        threat = np.concatenate([base_threat, onset_threat, cf_threat])
        counterfactual = np.concatenate(
            [
                np.zeros(len(base_threat), bool),
                np.zeros(len(onset_threat), bool),
                np.ones(len(cf_threat), bool),
            ]
        )
        return threat, counterfactual

    def _read(self, flat_index: int):
        fi, row = self._index[flat_index]
        f = self._files[fi]
        if self._flat_key:
            observation = np.asarray(
                f[self._flat_key][row], dtype=np.float32
            ).copy()
            if self._blank_previous_action and self._action_dim:
                start = FLAT_PREVIOUS_ACTION_START
                observation[start : start + self._action_dim] = 0.0
            action = np.asarray(f["actions"][row], dtype=np.float32)
            if self._discrete_actions:
                action = np.asarray(discretize_lateral(action), dtype=np.int64)
            return (
                observation,
                action,
            )
        obs: dict[str, np.ndarray] = {}
        if self._event_key:
            obs["events"] = np.asarray(f[self._event_key][row], dtype=np.float32)
        if self._state_key:
            obs["state"] = np.asarray(f[self._state_key][row], dtype=np.float32)
        if self._privileged_key:
            # The model must stay blind to this channel, so what it receives
            # is zeros -- but the auxiliary distance target is *derived* from
            # the true values, which therefore have to be read separately and
            # carried under a key the network never consumes. Reusing the
            # blinded key made every target read "all clear".
            true_privileged = np.asarray(
                f[self._privileged_key][row], dtype=np.float32
            )
            obs["privileged"] = np.zeros_like(true_privileged)
            obs["distance_target"] = angular_distance_map(true_privileged)
        if self._downsample_events > 1 and "events" in obs:
            # Average-pool the stored time surface. 2x480x640 is 614k inputs
            # against a few thousand samples; pooling trades spatial detail
            # the obstacle does not have at this range (17 px across at 3.3 m)
            # for a far better parameter-to-sample ratio.
            obs["events"] = pool_events(obs["events"], self._downsample_events)
        action = np.asarray(f["actions"][row], dtype=np.float32)
        if self._discrete_actions:
            action = np.asarray(discretize_lateral(action), dtype=np.int64)
        if self._blank_tracking_error and "state" in obs:
            # pos_err and vel_err only. The action ~ pos_err map is
            # self-sustaining at *any* displacement, not just at zero, so a
            # clone that learns it holds whatever offset a dodge produced:
            # measured 0.457 m of standing offset with the nearest obstacle
            # 1.5-3 m away, against the expert's 0.003 m. Blanking the whole
            # state removes that but also removes the feedback needed to fly
            # (0% success); attitude and the lookahead errors carry tracking
            # information without encoding current displacement.
            obs["state"][0:6] = 0.0
        if self._events_only_state and "state" in obs:
            # Every state channel is expressed relative to the nominal path --
            # pos_err, vel_err and the lookahead errors all encode the
            # drone's displacement from it -- and under desired_body_offset
            # the action *is* a nominal-relative displacement. So the
            # observation contains the answer up to a scale factor: measured
            # R2 = 0.9997 for a linear fit on state alone under an
            # episode-level holdout, and still 0.9929 after removing
            # prev_action, pos_err and vel_err. Only yaw_err carries nothing
            # (R2 = -0.015). Zeroing all of it leaves events as the sole route
            # to the expert's action, which is the point of the test.
            keep = obs["state"][6:7].copy()
            obs["state"] = np.zeros_like(obs["state"])
            obs["state"][6:7] = keep
        if self._blank_previous_action and "state" in obs and self._action_dim:
            # The observation carries the previous action as its last
            # action_dim components, and the expert's command follows a
            # smooth quintic -- so consecutive labels are nearly identical
            # and BC can satisfy its objective by copying that channel
            # instead of reading the events. Measured on the first run:
            # threat loss fell 0.00834 -> 0.00062 in one epoch (RMS 0.025 on
            # a +-1 action) while the threat/quiet ratio *shrank* 7.0x ->
            # 4.6x, which is backwards if the model were perceiving.
            #
            # It is a shortcut only BC can take: PPO must generate the
            # action, not predict a recorded one. At rollout the clone
            # bootstraps from its own zero initial action and "copy the
            # previous action" then emits zero forever -- excellent loss, no
            # dodging. Blanking the channel for every sample forces the
            # prediction through events and state alone.
            obs["state"][-self._action_dim :] = 0.0
        return obs, action

    def __getitem__(self, index: int):
        n_base = len(self._index)
        n_onset = len(self._onset)

        if index < n_base:
            return self._read(index)

        if index < n_base + n_onset:
            # Virtual onset: same frame and label, but the previous-action
            # channel blanked so the reaction cannot be copied from it.
            obs, action = self._read(self._onset[index - n_base])
            if self._action_dim:
                if isinstance(obs, dict) and "state" in obs:
                    obs["state"][-self._action_dim :] = 0.0
                elif not isinstance(obs, dict):
                    start = FLAT_PREVIOUS_ACTION_START
                    obs[start : start + self._action_dim] = 0.0
            return obs, action

        # Zero-event counterfactual: nothing visible, so do nothing.
        obs, _ = self._read(self._counterfactual[index - n_base - n_onset])
        if isinstance(obs, dict) and "events" in obs:
            obs["events"] = np.zeros_like(obs["events"])
        if isinstance(obs, dict) and "distance_target" in obs:
            # Blank events must not be paired with a target saying an obstacle
            # is present -- that trains the encoder to hallucinate.
            obs["distance_target"] = np.ones_like(obs["distance_target"])
        if isinstance(obs, dict) and "state" in obs and self._action_dim:
            obs["state"][-self._action_dim :] = 0.0
        if self._discrete_actions:
            return obs, np.asarray(1, dtype=np.int64)  # "none"
        return obs, np.zeros(self._action_dim, dtype=np.float32)

    def close(self) -> None:
        for f in self._files:
            f.close()
        self._files = []


def _collate(batch):
    """Stack dict or flat observations into batched tensors."""
    observations, actions = zip(*batch)
    if not isinstance(observations[0], dict):
        return (
            torch.from_numpy(np.stack(observations)),
            torch.from_numpy(np.stack(actions)),
        )
    keys = observations[0].keys()
    obs = {
        k: torch.from_numpy(np.stack([o[k] for o in observations])) for k in keys
    }
    return obs, torch.from_numpy(np.stack(actions))


def _observation_to_device(obs, device):
    if isinstance(obs, dict):
        return {k: v.to(device) for k, v in obs.items()}
    return obs.to(device)


def evaluate(model, env, episodes: int, seed0: int, downsample_events: int = 1, discrete_actions: bool = False) -> dict[str, float]:
    """Roll the cloned policy out on the real env.

    Imitation loss is a poor proxy for competence here: the expert is quiet
    most of the time, so a policy that always outputs zero scores well on
    loss while being exactly the passive baseline. Only closed-loop
    success and survival separate them.
    """
    successes, lengths, returns, terminations = 0, [], [], {}
    peak_accelerations = []
    peak_command_accelerations = []
    device = next(model.parameters()).device
    for i in range(episodes):
        obs, _ = env.reset(seed=seed0 + i)
        steps = 0
        episode_return = 0.0
        policy_dt = float(
            env.policy_decimation
            * env.steps_per_action
            / env.sim.config.world_rate
        )
        previous_velocity = np.asarray(
            env.sim.dynamics.state["v"], dtype=np.float64
        ).copy()
        previous_delta_velocity = np.asarray(
            getattr(env, "_last_delta_velocity", np.zeros(3)), dtype=np.float64
        ).copy()
        peak_acceleration = 0.0
        peak_command_acceleration = 0.0
        while True:
            with torch.no_grad():
                # Keep every key: forward() blinds the privileged channel
                # itself, exactly as the SB3 actor does.
                if isinstance(obs, dict):
                    batch = {}
                    for k, v in obs.items():
                        arr = np.asarray(v)
                        if k == "events":
                            arr = pool_events(arr, downsample_events)
                        batch[k] = torch.from_numpy(arr[None]).to(device)
                else:
                    batch = torch.from_numpy(np.asarray(obs)[None]).to(device)
                out = model(batch).cpu().numpy()[0]
                if discrete_actions:
                    # argmax -> full-authority lateral command. With
                    # lateral_axis_only the task zeroes x and z anyway, so
                    # only the middle component is read.
                    action = np.zeros(3, dtype=np.float32)
                    action[1] = float(int(np.argmax(out)) - 1)
                else:
                    action = out
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            velocity = np.asarray(
                env.sim.dynamics.state["v"], dtype=np.float64
            ).copy()
            peak_acceleration = max(
                peak_acceleration,
                float(np.linalg.norm(velocity - previous_velocity) / policy_dt),
            )
            previous_velocity = velocity
            delta_velocity = np.asarray(
                getattr(env, "_last_delta_velocity", np.zeros(3)),
                dtype=np.float64,
            ).copy()
            peak_command_acceleration = max(
                peak_command_acceleration,
                float(
                    np.linalg.norm(delta_velocity - previous_delta_velocity)
                    / policy_dt
                ),
            )
            previous_delta_velocity = delta_velocity
            steps += 1
            if terminated or truncated:
                reason = info.get("termination_reason", "truncated")
                terminations[reason] = terminations.get(reason, 0) + 1
                successes += int(bool(info.get("is_success", False)))
                lengths.append(steps)
                returns.append(episode_return)
                peak_accelerations.append(peak_acceleration)
                peak_command_accelerations.append(peak_command_acceleration)
                break
    return {
        "success_rate": successes / max(episodes, 1),
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "mean_steps": float(np.mean(lengths)) if lengths else 0.0,
        "terminations": terminations,
        "mean_peak_acceleration_mps2": float(np.mean(peak_accelerations)),
        "max_peak_acceleration_mps2": float(max(peak_accelerations, default=0.0)),
        "mean_peak_residual_command_acceleration_mps2": float(
            np.mean(peak_command_accelerations)
        ),
        "max_peak_residual_command_acceleration_mps2": float(
            max(peak_command_accelerations, default=0.0)
        ),
    }


def train_and_evaluate_dataset(args, cfg, dataset, weights, test_dataset=None) -> dict[str, Any]:
    """Fit the SB3 feature extractor + action head to the expert."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from train_sb3 import build_policy_config

    from gymnasium import spaces

    from neurosim.rl import env_class_for_task
    from neurosim.rl.sb3_features import PRIVILEGED_KEY

    device = torch.device(args.device)
    env_cfg = dict(cfg["env"])
    env_cfg["enable_visualization"] = False
    env = env_class_for_task(env_cfg["task"]["name"])(env_config=env_cfg, train=False)
    model, action_dim = build_clone_net(cfg, env, args.downsample_events, args.discrete_actions)
    if args.init_checkpoint:
        payload = torch.load(args.init_checkpoint, map_location="cpu")
        model.load_state_dict(payload["model"])
        print(f"initialized clone from {args.init_checkpoint}", flush=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    sampler = torch.utils.data.WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=min(args.samples_per_epoch, len(dataset)),
        replacement=True,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=_collate,
        num_workers=args.num_workers,
    )

    history = []
    for epoch in range(args.epochs):
        model.train()
        totals = np.zeros(3)
        batches = 0
        for obs, target in loader:
            obs = _observation_to_device(obs, device)
            target = target.to(device)
            target_map = obs.pop("distance_target", None) if isinstance(obs, dict) else None
            aux = torch.zeros((), device=device)
            if args.distance_aux_weight > 0.0 and target_map is not None:
                predicted_map = model.predict_distance(obs)
                if predicted_map is not None:
                    aux = nn.functional.mse_loss(predicted_map, target_map)
            if args.discrete_actions:
                logits = model.pre_tanh(obs)
                loss = nn.functional.cross_entropy(logits, target)
                loss = loss + args.distance_aux_weight * aux
                predicted_class = logits.argmax(dim=1)
                actionable = target != 1
                a = float(
                    (predicted_class[actionable] == target[actionable]).float().mean()
                ) if actionable.any() else 0.0
                b = float((predicted_class == target).float().mean())
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                totals += [loss.item(), a, b]
                batches += 1
                continue
            raw = model.pre_tanh(obs)
            predicted = torch.tanh(raw) if model.squash else raw
            if model.squash:
                # The expert's action is clip(residual / offset_max, -1, 1),
                # so it saturates at exactly +-1 -- which tanh can only
                # approach. Regressing onto the rail drives the pre-tanh mean
                # outward without bound, which is the same drift that killed
                # v23 and v24 (mean |mu| 0.002 -> 0.567, success -> 0.0%).
                # Clamping just inside the range keeps every target
                # reachable at a finite mean.
                target = target.clamp(-SQUASH_TARGET_LIMIT, SQUASH_TARGET_LIMIT)
            if action_dim == 4 and args.gated_action_loss:
                loss, a, b = gated_imitation_loss(predicted, target)
            else:
                loss, a, b = imitation_loss(predicted, target)
            if model.squash and args.pretanh_penalty > 0.0:
                loss = loss + args.pretanh_penalty * raw.pow(2).mean()
            loss = loss + args.distance_aux_weight * aux
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals += [loss.item(), float(a), float(b)]
            batches += 1
        row = {
            "epoch": epoch,
            "loss": totals[0] / max(batches, 1),
            **(
                {
                    "dodge_accuracy": totals[1] / max(batches, 1),
                    "overall_accuracy": totals[2] / max(batches, 1),
                }
                if args.discrete_actions
                else {
                    "threat_loss": totals[1] / max(batches, 1),
                    "quiet_loss": totals[2] / max(batches, 1),
                }
            ),
        }
        if args.eval_episodes and (epoch + 1) % args.eval_every == 0:
            model.eval()
            row.update(evaluate(model, env, args.eval_episodes, args.eval_seed0, args.downsample_events, args.discrete_actions))
        if test_dataset is not None:
            row.update(holdout_losses(model, test_dataset, device, discrete=args.discrete_actions))
        history.append(row)
        print(json.dumps(row), flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "action_dim": action_dim}, out.with_suffix(".pt")
    )
    out.write_text(json.dumps({"history": history}, indent=2))
    env.close()
    return {"history": history}


@torch.no_grad()
def holdout_losses(model, dataset, device, batch_size: int = 64, discrete: bool = False) -> dict[str, float]:
    """Threat/quiet MSE on episodes the model never trained on.

    Without this every loss number is a training loss, and on a shard of a
    few thousand samples seen ~25 times over it cannot be distinguished from
    memorisation -- which makes it useless for comparing against the
    held-out R2 = 0.09 that a linear fit on the state vector achieves.
    """
    from neurosim.rl.sb3_features import PRIVILEGED_KEY  # noqa: PLC0415

    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate
    )

    def distance_stats():
        """Held-out distance-map error against the all-clear baseline.

        This is the perception question stated directly: can the encoder
        recover where the obstacle is, on episodes it never trained on? The
        baseline is predicting "clear everywhere" -- an R2 above 0 means the
        events carry obstacle geometry that generalises, and below 0 means
        they do not, independent of any action or reward.
        """
        se = n = 0.0
        base_se = 0.0
        for obs, _ in loader:
            obs = _observation_to_device(obs, device)
            target = obs.pop("distance_target", None) if isinstance(obs, dict) else None
            if target is None:
                return {}
            predicted = model.predict_distance(obs)
            if predicted is None:
                return {}
            se += float(((predicted - target) ** 2).sum())
            base_se += float(((torch.ones_like(target) - target) ** 2).sum())
            n += float(target.numel())
        mse = se / max(n, 1.0)
        base = base_se / max(n, 1.0)
        return {
            "holdout_distance_mse": mse,
            "holdout_distance_baseline_mse": base,
            "holdout_distance_r2": 1.0 - mse / base if base > 0 else float("nan"),
        }

    if discrete:
        correct_dodge = total_dodge = correct_all = total_all = 0
        for obs, target in loader:
            obs = _observation_to_device(obs, device)
            target = target.to(device)
            pred = model(obs).argmax(dim=1)
            actionable = target != 1
            correct_dodge += int((pred[actionable] == target[actionable]).sum())
            total_dodge += int(actionable.sum())
            correct_all += int((pred == target).sum())
            total_all += int(target.numel())
        model.train()
        # Majority-class rate is the bar: always predicting "none" scores
        # this without looking at anything.
        return {
            "holdout_dodge_accuracy": correct_dodge / max(total_dodge, 1),
            "holdout_overall_accuracy": correct_all / max(total_all, 1),
            "holdout_dodge_n": total_dodge,
            "holdout_majority_rate": 1.0 - total_dodge / max(total_all, 1),
            **distance_stats(),
        }
    sq_threat, n_threat, sq_quiet, n_quiet = 0.0, 0, 0.0, 0
    targets = []
    for obs, target in loader:
        obs = _observation_to_device(obs, device)
        target = target.to(device)
        predicted = model(obs)
        per = ((predicted - target) ** 2).mean(dim=1)
        threat = target.norm(dim=1) > 0.1
        sq_threat += float(per[threat].sum()); n_threat += int(threat.sum())
        sq_quiet += float(per[~threat].sum()); n_quiet += int((~threat).sum())
        targets.append(target.cpu().numpy())
    model.train()
    all_t = np.concatenate(targets) if targets else np.zeros((0, 1))
    thr_mask = np.linalg.norm(all_t, axis=1) > 0.1
    variance = (
        float(((all_t[thr_mask] - all_t[thr_mask].mean(0)) ** 2).mean())
        if thr_mask.any() else float("nan")
    )
    threat_mse = sq_threat / max(n_threat, 1)
    return {
        **distance_stats(),
        "holdout_threat_loss": threat_mse,
        "holdout_quiet_loss": sq_quiet / max(n_quiet, 1),
        "holdout_threat_variance": variance,
        "holdout_threat_r2": 1.0 - threat_mse / variance if variance else float("nan"),
        "holdout_threat_n": n_threat,
    }


def build_clone_net(cfg, env, downsample_events: int = 1, discrete_actions: bool = False):
    """Build the clone actor for ``cfg``/``env``; used by training and DAgger.

    Lives at module scope so the DAgger collector can reconstruct the same
    network to roll out, rather than duplicating the architecture.
    """
    from neurosim.rl.sb3_features import PRIVILEGED_KEY  # noqa: PLC0415
    from train_sb3 import build_policy_config  # noqa: PLC0415
    from stable_baselines3.common.torch_layers import FlattenExtractor  # noqa: PLC0415

    action_dim = int(np.prod(env.action_space.shape))
    _, policy_kwargs = build_policy_config(
        str(cfg["env"]["obs_mode"]),
        float(cfg["ppo"]["log_std_init"]),
        privileged=False,
        event_presence_features=bool(cfg["ppo"].get("event_presence_features", False)),
        event_high_resolution=bool(cfg["ppo"].get("event_high_resolution", False)),
    )
    extractor_cls = policy_kwargs.get("features_extractor_class", FlattenExtractor)
    extractor_kwargs = dict(policy_kwargs.get("features_extractor_kwargs", {}))
    squash = bool(policy_kwargs.get("squash_output", False))
    actor_observation_space = env.observation_space
    if (
        downsample_events > 1
        and hasattr(actor_observation_space, "spaces")
        and "events" in actor_observation_space.spaces
    ):
        import gymnasium as gym  # noqa: PLC0415

        old = actor_observation_space.spaces["events"]
        c, h, w = old.shape
        spaces = dict(actor_observation_space.spaces)
        spaces["events"] = gym.spaces.Box(
            float(old.low.min()), float(old.high.max()),
            (c, h // downsample_events, w // downsample_events), old.dtype,
        )
        actor_observation_space = gym.spaces.Dict(spaces)

    class CloneNet(nn.Module):
        """Same perception stack PPO uses, with a deterministic action head.

        Reusing the extractor matters: the point of the warm start is to
        hand PPO weights it can keep fine-tuning, which requires the same
        architecture it would have built itself.
        """

        def __init__(self):
            super().__init__()
            # The env exposes a critic-only 'privileged' channel. Cloning the
            # *actor* means reproducing what it can actually perceive, so the
            # extractor is built over the actor's subspace only -- otherwise
            # it grows a privileged head and the clone would depend on
            # ground-truth obstacle geometry it will not have at deployment.
            self.features_extractor = extractor_cls(
                actor_observation_space, **extractor_kwargs
            )
            # Same trunk SB3 builds from net_arch=[64, 64] with Tanh.
            self.policy_net = nn.Sequential(
                nn.Linear(self.features_extractor.features_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
            )
            # Produces the pre-tanh mean, matching SB3's action_net under
            # squash_output=True. build_policy_config sets that flag, so the
            # PPO policy these weights load into computes tanh(action_net(x)).
            # Leaving the clone unbounded would train action_net to emit the
            # command itself, and the warm start would then squash it a
            # second time -- an oracle command of 1.0 arriving as 0.76,
            # worst precisely in the full-authority dodges.
            self.action_net = nn.Linear(64, 3 if discrete_actions else action_dim)
            # Reads the event branch only -- see forward_events. Feeding it the
            # concatenated features would let proprioception satisfy the
            # auxiliary loss, recreating the shortcut it exists to remove.
            self.distance_decoder = nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, DISTANCE_BINS)
            )
            # Logits are unbounded by construction; squashing them would be
            # a category error.
            self.squash = squash and not discrete_actions

        def pre_tanh(self, obs):
            # Blind the actor exactly as AsymmetricActorCriticPolicy does.
            if isinstance(obs, dict):
                obs = dict(obs)
                if PRIVILEGED_KEY in obs:
                    obs[PRIVILEGED_KEY] = torch.zeros_like(obs[PRIVILEGED_KEY])
            return self.action_net(self.policy_net(self.features_extractor(obs)))

        def predict_distance(self, obs):
            """Predicted 1-D distance map from events alone, or None."""
            if not isinstance(obs, dict):
                return None
            blinded = dict(obs)
            if PRIVILEGED_KEY in blinded:
                blinded[PRIVILEGED_KEY] = torch.zeros_like(blinded[PRIVILEGED_KEY])
            features = self.features_extractor.forward_events(blinded)
            if features is None:
                return None
            return self.distance_decoder(features)

        def forward(self, obs):
            raw = self.pre_tanh(obs)
            return torch.tanh(raw) if self.squash else raw

    return CloneNet(), action_dim


def main() -> None:
    from train_sb3 import load_experiment_config

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-config", required=True)
    p.add_argument("--dataset", nargs="+", required=True)
    p.add_argument("--output", default="outputs/rl/bc/clone.json")
    p.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional clone .pt to fine-tune instead of starting from scratch",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--samples-per-epoch", type=int, default=20000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gated-onset-window", type=int, default=4)
    p.add_argument("--zero-event-counterfactuals", action="store_true", default=True)
    p.add_argument("--gated-action-loss", action="store_true", default=False)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-seed0", type=int, default=9001)
    p.add_argument("--downsample-events", type=int, default=1)
    p.add_argument("--discrete-actions", action="store_true")
    p.add_argument("--distance-aux-weight", type=float, default=0.0)
    p.add_argument("--holdout-fraction", type=float, default=0.0)
    p.add_argument("--holdout-seed", type=int, default=0)
    p.add_argument(
        "--blank-tracking-error",
        action="store_true",
        help=(
            "Zero pos_err and vel_err only, keeping attitude and lookahead. "
            "Removes the self-sustaining action ~ pos_err shortcut that makes "
            "a clone park at a standing offset, without removing the feedback "
            "it needs to track a path."
        ),
    )
    p.add_argument(
        "--events-only-state",
        action="store_true",
        help=(
            "Zero every nominal-relative state channel. pos_err, vel_err and "
            "the lookahead errors all encode displacement from the nominal "
            "path, and the action IS a nominal-relative displacement, so a "
            "linear fit on state alone predicts the expert at R2 = 0.9997 "
            "under an episode-level holdout. That shortcut has a fixed point "
            "at zero -- a clone that learns it never initiates a dodge."
        ),
    )
    p.add_argument(
        "--blank-previous-action",
        action="store_true",
        help=(
            "Zero the previous-action channel on every sample, not just the "
            "onset/counterfactual augmentations, to block the copycat shortcut"
        ),
    )
    p.add_argument(
        "--pretanh-penalty",
        type=float,
        default=0.01,
        help="L2 on the pre-tanh mean; matches the PPO-side pretanh_penalty",
    )
    args = p.parse_args()

    cfg = load_experiment_config(args.experiment_config)
    dataset = HDF5ObservationDataset(
        args.dataset,
        gated_onset_window=args.gated_onset_window,
        zero_event_counterfactuals=args.zero_event_counterfactuals,
        gated_action_loss=args.gated_action_loss,
        blank_previous_action=args.blank_previous_action,
        events_only_state=args.events_only_state,
        blank_tracking_error=args.blank_tracking_error,
        discrete_actions=args.discrete_actions,
        downsample_events=args.downsample_events,
        holdout_fraction=args.holdout_fraction,
        holdout_side="train",
        holdout_seed=args.holdout_seed,
    )
    threat, counterfactual = dataset.sample_kinds
    weights = balanced_sample_weights(threat, counterfactual)
    print(
        json.dumps(
            {
                "samples": len(dataset),
                "threat": int(threat.sum()),
                "counterfactual": int(counterfactual.sum()),
            }
        ),
        flush=True,
    )
    test_dataset = None
    if args.holdout_fraction > 0.0:
        test_dataset = HDF5ObservationDataset(
            args.dataset,
            gated_onset_window=0,
            zero_event_counterfactuals=False,
            gated_action_loss=args.gated_action_loss,
            blank_previous_action=args.blank_previous_action,
            events_only_state=args.events_only_state,
            blank_tracking_error=args.blank_tracking_error,
            discrete_actions=args.discrete_actions,
            downsample_events=args.downsample_events,
            holdout_fraction=args.holdout_fraction,
            holdout_side="test",
            holdout_seed=args.holdout_seed,
        )
        print(
            f"holdout: {len(test_dataset)} test samples "
            f"(episode-level split, seed {args.holdout_seed})",
            flush=True,
        )
    try:
        train_and_evaluate_dataset(args, cfg, dataset, weights, test_dataset)
    finally:
        dataset.close()
        if test_dataset is not None:
            test_dataset.close()


if __name__ == "__main__":
    main()
