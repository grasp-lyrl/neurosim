"""Sequence behavior cloning for event-driven velocity dodging.

The feed-forward clone predicts every 20 Hz command independently and can
alternate left/right when small event changes cross its decision boundary.
This trainer keeps the same anti-shortcut observation contract but learns on
contiguous, episode-bounded sequences with a GRU and temporal supervision.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_velocity_dodge_bc import (  # noqa: E402
    DISCRETE_DEADBAND,
    FLAT_PREVIOUS_ACTION_START,
    SQUASH_TARGET_LIMIT,
    pool_events,
)


class HDF5SequenceDataset(torch.utils.data.Dataset):
    """Lazy fixed-length sequences that never cross an episode boundary."""

    def __init__(
        self,
        paths: Sequence[str],
        *,
        sequence_length: int = 32,
        stride: int = 8,
        downsample_events: int = 1,
        holdout_fraction: float = 0.0,
        holdout_side: str = "train",
        holdout_seed: int = 0,
        zero_event_counterfactuals: bool = False,
        preload_events: bool = False,
    ):
        import h5py

        self._files = [h5py.File(path, "r") for path in paths]
        self.sequence_length = int(sequence_length)
        self.stride = int(stride)
        self.downsample_events = max(int(downsample_events), 1)
        first = self._files[0]
        self._flat_key = "observations" if "observations" in first else None
        self._event_key = (
            "observations_events" if "observations_events" in first else None
        )
        self._state_key = (
            "observations_state" if "observations_state" in first else None
        )
        if self._flat_key is None and (self._event_key is None or self._state_key is None):
            raise ValueError(
                "dataset needs either observations or observations_events/observations_state"
            )
        if zero_event_counterfactuals and self._event_key is None:
            raise ValueError("zero-event counterfactuals require event observations")
        self._event_arrays = (
            [np.asarray(file[self._event_key][:]) for file in self._files]
            if preload_events and self._event_key is not None
            else None
        )
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")

        episode_keys: list[str] = []
        per_file_episodes: list[np.ndarray] = []
        for fi, file in enumerate(self._files):
            episodes = np.asarray(file["episode_index"][:])
            unique = np.unique(episodes)
            per_file_episodes.append(unique)
            episode_keys.extend(f"{fi}:{int(ep)}" for ep in unique)

        selected = set(episode_keys)
        if holdout_fraction > 0.0:
            shuffled = np.random.default_rng(holdout_seed).permutation(episode_keys)
            n_test = max(int(round(len(episode_keys) * holdout_fraction)), 1)
            test = set(shuffled[:n_test].tolist())
            selected = test if holdout_side == "test" else set(episode_keys) - test

        # entry = (file index, first row, counterfactual, group)
        # group: 0 genuine threat sequence, 1 genuine quiet, 2 blank-event.
        self._entries: list[tuple[int, int, bool, int]] = []
        self.selected_episodes = 0
        for fi, (file, episodes) in enumerate(zip(self._files, per_file_episodes)):
            episode_index = np.asarray(file["episode_index"][:])
            actions = file["actions"]
            for episode in episodes:
                if f"{fi}:{int(episode)}" not in selected:
                    continue
                rows = np.flatnonzero(episode_index == episode)
                if len(rows) < self.sequence_length:
                    continue
                if len(rows) > 1 and not np.all(np.diff(rows) == 1):
                    raise ValueError("episode rows must be contiguous in HDF5")
                self.selected_episodes += 1
                for offset in range(
                    0, len(rows) - self.sequence_length + 1, self.stride
                ):
                    start = int(rows[offset])
                    block = np.asarray(
                        actions[start : start + self.sequence_length],
                        dtype=np.float32,
                    )
                    has_threat = bool(np.any(np.linalg.norm(block, axis=1) > 0.1))
                    self._entries.append((fi, start, False, 0 if has_threat else 1))
                    if zero_event_counterfactuals and has_threat:
                        self._entries.append((fi, start, True, 2))

        if not self._entries:
            raise ValueError("no complete sequences found")

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def sequence_weights(self) -> np.ndarray:
        """Balance genuine threat / quiet / blank-event sequence mass."""
        groups = np.asarray([entry[3] for entry in self._entries], dtype=np.int64)
        present = [value for value in (0, 1, 2) if np.any(groups == value)]
        shares = {0: 0.5, 1: 0.25, 2: 0.25}
        if 2 not in present:
            shares = {0: 0.5, 1: 0.5}
        weights = np.zeros(len(groups), dtype=np.float64)
        for value in present:
            mask = groups == value
            weights[mask] = shares[value] / int(mask.sum())
        return weights

    @property
    def group_counts(self) -> list[int]:
        return np.bincount(
            np.asarray([entry[3] for entry in self._entries]), minlength=3
        ).tolist()

    def __getitem__(self, index: int):
        fi, start, counterfactual, _ = self._entries[index]
        file = self._files[fi]
        stop = start + self.sequence_length
        action = np.asarray(file["actions"][start:stop], dtype=np.float32)
        if self._flat_key is not None:
            obs = np.asarray(
                file[self._flat_key][start:stop], dtype=np.float32
            ).copy()
            return obs, action
        event_source = (
            file[self._event_key]
            if self._event_arrays is None
            else self._event_arrays[fi]
        )
        obs = {
            # Keep the stored float16 representation through DataLoader IPC.
            # Casting a native-resolution sequence on CPU is both slower and
            # twice as large; the model still receives float32 after the GPU
            # transfer below.
            "events": np.asarray(event_source[start:stop]),
            "state": np.asarray(
                file[self._state_key][start:stop], dtype=np.float32
            ),
        }
        if self.downsample_events > 1:
            obs["events"] = pool_events(obs["events"], self.downsample_events)
        if counterfactual:
            obs["events"] = np.zeros_like(obs["events"])
            action = np.zeros_like(action)
        return obs, action

    def close(self) -> None:
        for file in self._files:
            file.close()
        self._files = []


def build_gru_clone_net(
    cfg,
    env,
    downsample_events: int = 1,
    hidden_size: int = 128,
    *,
    blank_previous_action: bool = False,
    events_only_state: bool = False,
    blank_tracking_error: bool = False,
):
    """Build a stateful GRU actor over the same event encoder as PPO/BC."""
    import gymnasium as gym

    from train_sb3 import build_policy_config
    from stable_baselines3.common.torch_layers import FlattenExtractor

    action_dim = int(np.prod(env.action_space.shape))
    _, policy_kwargs = build_policy_config(
        str(cfg["env"]["obs_mode"]),
        float(cfg["ppo"]["log_std_init"]),
        privileged=False,
        event_presence_features=bool(
            cfg["ppo"].get("event_presence_features", False)
        ),
        event_high_resolution=bool(cfg["ppo"].get("event_high_resolution", False)),
    )
    extractor_cls = policy_kwargs.get("features_extractor_class", FlattenExtractor)
    extractor_kwargs = dict(policy_kwargs.get("features_extractor_kwargs", {}))
    observation_space = env.observation_space
    if (
        downsample_events > 1
        and hasattr(observation_space, "spaces")
        and "events" in observation_space.spaces
    ):
        spaces = dict(observation_space.spaces)
        old = spaces["events"]
        channels, height, width = old.shape
        spaces["events"] = gym.spaces.Box(
            float(old.low.min()),
            float(old.high.max()),
            (channels, height // downsample_events, width // downsample_events),
            old.dtype,
        )
        observation_space = gym.spaces.Dict(spaces)

    class GRUClone(nn.Module):
        recurrent_type = "gru"

        def __init__(self):
            super().__init__()
            self.downsample_events = int(downsample_events)
            self.hidden_size = int(hidden_size)
            self.blank_previous_action = bool(blank_previous_action)
            self.events_only_state = bool(events_only_state)
            self.blank_tracking_error = bool(blank_tracking_error)
            self.features_extractor = extractor_cls(
                observation_space, **extractor_kwargs
            )
            self.gru = nn.GRU(
                self.features_extractor.features_dim,
                self.hidden_size,
                batch_first=True,
            )
            self.policy_net = nn.Sequential(
                nn.Linear(self.hidden_size, 64), nn.Tanh()
            )
            self.action_net = nn.Linear(64, action_dim)
            self.side_head = nn.Linear(self.hidden_size, 3)
            self.squash = bool(policy_kwargs.get("squash_output", False))

        def preprocess(self, obs):
            if not isinstance(obs, dict):
                state = obs.clone()
                if self.blank_previous_action and action_dim:
                    start = FLAT_PREVIOUS_ACTION_START
                    state[..., start : start + action_dim] = 0.0
                return state
            obs = dict(obs)
            state = obs["state"].clone()
            if self.blank_tracking_error:
                state[..., 0:6] = 0.0
            if self.events_only_state:
                yaw_error = state[..., 6:7].clone()
                state.zero_()
                state[..., 6:7] = yaw_error
            if self.blank_previous_action and action_dim:
                state[..., -action_dim:] = 0.0
            obs["state"] = state
            return obs

        def forward_sequence(self, obs, hidden=None):
            obs = self.preprocess(obs)
            if isinstance(obs, dict):
                batch, steps = obs["state"].shape[:2]
                flat = {
                    key: value.reshape(batch * steps, *value.shape[2:])
                    for key, value in obs.items()
                }
            else:
                batch, steps = obs.shape[:2]
                flat = obs.reshape(batch * steps, *obs.shape[2:])
            features = self.features_extractor(flat).reshape(batch, steps, -1)
            recurrent, hidden = self.gru(features, hidden)
            raw = self.action_net(self.policy_net(recurrent))
            action = torch.tanh(raw) if self.squash else raw
            return action, raw, self.side_head(recurrent), hidden

        def recurrent_step(self, obs, hidden=None):
            sequence = (
                {key: value.unsqueeze(1) for key, value in obs.items()}
                if isinstance(obs, dict)
                else obs.unsqueeze(1)
            )
            action, _, _, hidden = self.forward_sequence(sequence, hidden)
            return action[:, 0], hidden

        def forward(self, obs):
            action, _ = self.recurrent_step(obs, None)
            return action

    return GRUClone(), action_dim


def side_labels(action: torch.Tensor) -> torch.Tensor:
    lateral = action[..., 1]
    labels = torch.ones_like(lateral, dtype=torch.long)
    labels[lateral < -DISCRETE_DEADBAND] = 0
    labels[lateral > DISCRETE_DEADBAND] = 2
    return labels


def observation_to_device(obs, device):
    """Move either a flat state tensor or a dict observation to a device."""
    if not isinstance(obs, dict):
        return obs.to(device)
    moved = {key: value.to(device) for key, value in obs.items()}
    if "events" in moved:
        moved["events"] = moved["events"].float()
    return moved


@torch.no_grad()
def holdout_metrics(model, dataset, device, batch_size: int = 8):
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    squared = count = 0.0
    targets = []
    side_correct = side_total = 0
    for obs, target in loader:
        obs = observation_to_device(obs, device)
        target = target.to(device)
        predicted, _, side_logits, _ = model.forward_sequence(obs)
        threat = target.norm(dim=-1) > 0.1
        squared += float(((predicted[threat] - target[threat]) ** 2).sum())
        count += float(threat.sum() * target.shape[-1])
        targets.append(target[threat].cpu().numpy())
        labels = side_labels(target)
        actionable = labels != 1
        side_correct += int(
            (side_logits.argmax(dim=-1)[actionable] == labels[actionable]).sum()
        )
        side_total += int(actionable.sum())
    all_targets = np.concatenate(targets) if targets else np.zeros((0, 3))
    variance = (
        float(((all_targets - all_targets.mean(axis=0)) ** 2).mean())
        if len(all_targets)
        else float("nan")
    )
    mse = squared / max(count, 1.0)
    return {
        "holdout_threat_loss": mse,
        "holdout_threat_variance": variance,
        "holdout_threat_r2": 1.0 - mse / variance if variance > 0 else float("nan"),
        "holdout_side_accuracy": side_correct / max(side_total, 1),
        "holdout_threat_n": int(count / 3),
    }


@torch.no_grad()
def evaluate_gru(
    model,
    env,
    episodes: int,
    seed0: int,
    downsample_events: int,
    *,
    zero_events: bool = False,
    zero_privileged: bool = False,
):
    device = next(model.parameters()).device
    successes = 0
    returns, lengths, peak_accelerations = [], [], []
    terminations: dict[str, int] = {}
    total_flips = total_committed_transitions = 0
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed0 + episode)
        hidden = None
        previous_sign = 0
        episode_return = 0.0
        peak_acceleration = 0.0
        previous_velocity = np.asarray(env.sim.dynamics.state["v"], float).copy()
        policy_dt = float(
            env.policy_decimation * env.steps_per_action / env.sim.config.world_rate
        )
        steps = 0
        while True:
            if isinstance(obs, dict):
                batch = {}
                for key, value in obs.items():
                    array = np.asarray(value)
                    if key == "events":
                        array = pool_events(array, downsample_events)
                        if zero_events:
                            array = np.zeros_like(array)
                    batch[key] = torch.from_numpy(array[None]).to(device)
            else:
                array = np.asarray(obs).copy()
                if zero_privileged:
                    array[FLAT_PREVIOUS_ACTION_START + 3 :] = 0.0
                batch = torch.from_numpy(array[None]).to(device)
            action_tensor, hidden = model.recurrent_step(batch, hidden)
            action = action_tensor.cpu().numpy()[0]
            lateral = float(action[1])
            sign = int(np.sign(lateral)) if abs(lateral) > DISCRETE_DEADBAND else 0
            if sign and previous_sign:
                total_committed_transitions += 1
                total_flips += int(sign != previous_sign)
            if sign:
                previous_sign = sign
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            velocity = np.asarray(env.sim.dynamics.state["v"], float).copy()
            peak_acceleration = max(
                peak_acceleration,
                float(np.linalg.norm(velocity - previous_velocity) / policy_dt),
            )
            previous_velocity = velocity
            steps += 1
            if terminated or truncated:
                reason = info.get("termination_reason", "truncated")
                terminations[reason] = terminations.get(reason, 0) + 1
                successes += int(bool(info.get("is_success", False)))
                returns.append(episode_return)
                lengths.append(steps)
                peak_accelerations.append(peak_acceleration)
                break
    return {
        "success_rate": successes / max(episodes, 1),
        "mean_return": float(np.mean(returns)),
        "mean_steps": float(np.mean(lengths)),
        "terminations": terminations,
        "mean_peak_acceleration_mps2": float(np.mean(peak_accelerations)),
        "max_peak_acceleration_mps2": float(max(peak_accelerations, default=0.0)),
        "committed_sign_flips": total_flips,
        "committed_sign_transitions": total_committed_transitions,
        "committed_sign_flip_rate": total_flips / max(total_committed_transitions, 1),
    }


def checkpoint_payload(model, action_dim, args, epoch, metric):
    return {
        "model": model.state_dict(),
        "action_dim": int(action_dim),
        "epoch": int(epoch),
        "validation_metric": float(metric),
        "preprocessing": {
            "recurrent_type": "gru",
            "hidden_size": int(args.hidden_size),
            "downsample_events": int(args.downsample_events),
            "blank_previous_action": bool(args.blank_previous_action),
            "events_only_state": bool(args.events_only_state),
            "blank_tracking_error": bool(args.blank_tracking_error),
            "sequence_length": int(args.sequence_length),
        },
    }


def main():
    from neurosim.rl import env_class_for_task
    from train_sb3 import load_experiment_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--dataset", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--init-encoder",
        default=None,
        help="Feed-forward BC checkpoint whose event/state encoder seeds the GRU",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples-per-epoch", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--preload-events",
        action="store_true",
        help="Cache training event arrays in RAM instead of repeatedly decoding HDF5 chunks.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--sequence-stride", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--downsample-events", type=int, default=2)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--holdout-seed", type=int, default=6262)
    parser.add_argument("--blank-previous-action", action="store_true")
    parser.add_argument("--events-only-state", action="store_true")
    parser.add_argument("--blank-tracking-error", action="store_true")
    parser.add_argument("--zero-event-counterfactuals", action="store_true")
    parser.add_argument("--temporal-weight", type=float, default=0.25)
    parser.add_argument("--side-aux-weight", type=float, default=0.10)
    parser.add_argument("--flip-penalty", type=float, default=0.10)
    parser.add_argument("--pretanh-penalty", type=float, default=0.01)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seed0", type=int, default=15001)
    parser.add_argument("--eval-zero-events", action="store_true")
    parser.add_argument(
        "--eval-zero-privileged",
        action="store_true",
        help="Also evaluate flat state with obstacle-geometry slots zeroed.",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    cfg = load_experiment_config(args.experiment_config)
    env_cfg = copy.deepcopy(cfg["env"])
    env_cfg["enable_visualization"] = False
    env = env_class_for_task(env_cfg["task"]["name"])(env_config=env_cfg, train=False)

    train_dataset = HDF5SequenceDataset(
        args.dataset,
        sequence_length=args.sequence_length,
        stride=args.sequence_stride,
        downsample_events=args.downsample_events,
        holdout_fraction=args.holdout_fraction,
        holdout_side="train",
        holdout_seed=args.holdout_seed,
        zero_event_counterfactuals=args.zero_event_counterfactuals,
        preload_events=args.preload_events,
    )
    test_dataset = HDF5SequenceDataset(
        args.dataset,
        sequence_length=args.sequence_length,
        stride=args.sequence_length,
        downsample_events=args.downsample_events,
        holdout_fraction=args.holdout_fraction,
        holdout_side="test",
        holdout_seed=args.holdout_seed,
        zero_event_counterfactuals=False,
    )
    if args.preload_events:
        # Both splits refer to rows in the same source files. Reuse the
        # training cache rather than decompressing the holdout rows again.
        test_dataset._event_arrays = train_dataset._event_arrays
    print(
        json.dumps(
            {
                "train_sequences": len(train_dataset),
                "holdout_sequences": len(test_dataset),
                "train_episodes": train_dataset.selected_episodes,
                "holdout_episodes": test_dataset.selected_episodes,
                "sequence_groups": train_dataset.group_counts,
            }
        ),
        flush=True,
    )

    model, action_dim = build_gru_clone_net(
        cfg,
        env,
        args.downsample_events,
        args.hidden_size,
        blank_previous_action=args.blank_previous_action,
        events_only_state=args.events_only_state,
        blank_tracking_error=args.blank_tracking_error,
    )
    if args.init_encoder:
        initial = torch.load(args.init_encoder, map_location="cpu")["model"]
        encoder = {
            key: value
            for key, value in initial.items()
            if key.startswith("features_extractor.")
        }
        incompatible = model.load_state_dict(encoder, strict=False)
        unexpected = [key for key in incompatible.unexpected_keys if key in encoder]
        if unexpected:
            raise ValueError(f"unexpected encoder keys: {unexpected}")
        print(
            f"initialized {len(encoder)} encoder tensors from {args.init_encoder}",
            flush=True,
        )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.as_tensor(train_dataset.sequence_weights, dtype=torch.double),
        num_samples=min(args.samples_per_epoch, len(train_dataset)),
        replacement=True,
    )
    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best_metric, best_epoch = -float("inf"), -1
    try:
        for epoch in range(args.epochs):
            model.train()
            totals = np.zeros(5, dtype=np.float64)
            batches = 0
            for obs, target in loader:
                obs = observation_to_device(obs, device)
                target = target.to(device).clamp(
                    -SQUASH_TARGET_LIMIT, SQUASH_TARGET_LIMIT
                )
                predicted, raw, side_logits, _ = model.forward_sequence(obs)
                action_loss = nn.functional.mse_loss(predicted, target)
                temporal_loss = nn.functional.mse_loss(
                    predicted[:, 1:] - predicted[:, :-1],
                    target[:, 1:] - target[:, :-1],
                )
                labels = side_labels(target)
                side_loss = nn.functional.cross_entropy(
                    side_logits.reshape(-1, 3), labels.reshape(-1)
                )
                target_stable = (
                    target[:, 1:, 1] * target[:, :-1, 1]
                    > DISCRETE_DEADBAND**2
                )
                flip_values = torch.relu(
                    -predicted[:, 1:, 1] * predicted[:, :-1, 1]
                )
                flip_loss = (
                    flip_values[target_stable].mean()
                    if bool(target_stable.any())
                    else predicted.sum() * 0.0
                )
                loss = (
                    action_loss
                    + args.temporal_weight * temporal_loss
                    + args.side_aux_weight * side_loss
                    + args.flip_penalty * flip_loss
                    + args.pretanh_penalty * raw.pow(2).mean()
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                totals += [
                    float(loss.detach()),
                    float(action_loss.detach()),
                    float(temporal_loss.detach()),
                    float(side_loss.detach()),
                    float(flip_loss.detach()),
                ]
                batches += 1
            model.eval()
            row = {
                "epoch": epoch,
                "loss": totals[0] / max(batches, 1),
                "action_loss": totals[1] / max(batches, 1),
                "temporal_loss": totals[2] / max(batches, 1),
                "side_loss": totals[3] / max(batches, 1),
                "flip_loss": totals[4] / max(batches, 1),
                **holdout_metrics(model, test_dataset, device, args.batch_size),
            }
            metric = float(row["holdout_threat_r2"])
            if np.isfinite(metric) and metric > best_metric:
                best_metric, best_epoch = metric, epoch
                torch.save(
                    checkpoint_payload(model, action_dim, args, epoch, metric),
                    output.with_name(f"{output.stem}.best.pt"),
                )
            history.append(row)
            print(json.dumps(row), flush=True)

        torch.save(
            checkpoint_payload(
                model, action_dim, args, args.epochs - 1, history[-1]["holdout_threat_r2"]
            ),
            output.with_suffix(".pt"),
        )
        best_path = output.with_name(f"{output.stem}.best.pt")
        best = torch.load(best_path, map_location=device)
        model.load_state_dict(best["model"])
        model.eval()
        evaluation = evaluate_gru(
            model,
            env,
            args.eval_episodes,
            args.eval_seed0,
            args.downsample_events,
        )
        if args.eval_zero_events:
            ablation = evaluate_gru(
                model,
                env,
                args.eval_episodes,
                args.eval_seed0,
                args.downsample_events,
                zero_events=True,
            )
            evaluation.update(
                {f"zero_event_{key}": value for key, value in ablation.items()}
            )
        if args.eval_zero_privileged:
            ablation = evaluate_gru(
                model,
                env,
                args.eval_episodes,
                args.eval_seed0,
                args.downsample_events,
                zero_privileged=True,
            )
            evaluation.update(
                {f"zero_privileged_{key}": value for key, value in ablation.items()}
            )
        output.write_text(
            json.dumps(
                {
                    "history": history,
                    "best_epoch": best_epoch,
                    "best_validation_metric": best_metric,
                    "best_checkpoint": str(best_path),
                    "best_evaluation": evaluation,
                },
                indent=2,
            )
        )
    finally:
        train_dataset.close()
        test_dataset.close()
        env.close()


if __name__ == "__main__":
    main()
