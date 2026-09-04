"""Train a compact event-to-dodge GRU on paired privileged supervision."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_velocity_dodge_bc import DISCRETE_DEADBAND, pool_events  # noqa: E402
from train_velocity_dodge_gru_bc import build_gru_clone_net  # noqa: E402


TCA_SCALE_S = 1.5
MISS_SCALE_M = 2.5


class PairedCompactSequenceDataset(torch.utils.data.Dataset):
    """Episode-bounded event sequences paired with exact obstacle geometry."""

    def __init__(
        self,
        event_paths: Sequence[str],
        privileged_paths: Sequence[str],
        *,
        sequence_length: int = 32,
        stride: int = 8,
        downsample_events: int = 2,
        holdout_fraction: float = 0.2,
        holdout_side: str = "train",
        holdout_seed: int = 6262,
        counterfactuals: bool = True,
        preload_events: bool = False,
        shared_event_arrays=None,
    ):
        import h5py

        if len(event_paths) != len(privileged_paths):
            raise ValueError("event and privileged shard counts differ")
        self.event_files = [h5py.File(path, "r") for path in event_paths]
        self.privileged_files = [h5py.File(path, "r") for path in privileged_paths]
        self.sequence_length = int(sequence_length)
        self.downsample_events = max(int(downsample_events), 1)
        self.event_arrays = shared_event_arrays
        if preload_events and self.event_arrays is None:
            self.event_arrays = [
                np.asarray(file["observations_events"][:])
                for file in self.event_files
            ]

        episode_keys = []
        per_file_episodes = []
        for fi, (event_file, privileged_file) in enumerate(
            zip(self.event_files, self.privileged_files)
        ):
            for key in ("episode_index", "episode_seed", "actions", "threat"):
                if not np.array_equal(event_file[key][:], privileged_file[key][:]):
                    raise ValueError(f"paired shard mismatch for {key} at index {fi}")
            episodes = np.unique(event_file["episode_index"][:])
            per_file_episodes.append(episodes)
            episode_keys.extend(f"{fi}:{int(episode)}" for episode in episodes)

        shuffled = np.random.default_rng(holdout_seed).permutation(episode_keys)
        n_test = max(int(round(len(episode_keys) * holdout_fraction)), 1)
        test = set(shuffled[:n_test].tolist())
        selected = test if holdout_side == "test" else set(episode_keys) - test

        # (file index, first row, blank-event counterfactual, group)
        self.entries = []
        self.selected_episodes = 0
        for fi, (file, episodes) in enumerate(zip(self.event_files, per_file_episodes)):
            episode_index = np.asarray(file["episode_index"][:])
            for episode in episodes:
                if f"{fi}:{int(episode)}" not in selected:
                    continue
                rows = np.flatnonzero(episode_index == episode)
                if len(rows) < self.sequence_length:
                    continue
                self.selected_episodes += 1
                for offset in range(0, len(rows) - self.sequence_length + 1, stride):
                    start = int(rows[offset])
                    action = np.asarray(
                        file["actions"][start : start + self.sequence_length]
                    )
                    threat = bool(np.any(np.abs(action[:, 1]) > DISCRETE_DEADBAND))
                    self.entries.append((fi, start, False, 0 if threat else 1))
                    if counterfactuals and threat:
                        self.entries.append((fi, start, True, 2))

    def __len__(self):
        return len(self.entries)

    @property
    def group_counts(self):
        return np.bincount([entry[3] for entry in self.entries], minlength=3).tolist()

    @property
    def sequence_weights(self):
        groups = np.asarray([entry[3] for entry in self.entries])
        shares = {0: 0.5, 1: 0.25, 2: 0.25}
        weights = np.zeros(len(groups), dtype=np.float64)
        for group in np.unique(groups):
            mask = groups == group
            weights[mask] = shares[int(group)] / int(mask.sum())
        return weights

    def __getitem__(self, index):
        fi, start, counterfactual, _ = self.entries[index]
        stop = start + self.sequence_length
        event_file = self.event_files[fi]
        source = (
            event_file["observations_events"]
            if self.event_arrays is None
            else self.event_arrays[fi]
        )
        events = np.asarray(source[start:stop])
        if self.downsample_events > 1:
            events = pool_events(events, self.downsample_events)
        state = np.asarray(
            event_file["observations_state"][start:stop], dtype=np.float32
        )
        action = np.asarray(event_file["actions"][start:stop], dtype=np.float32)
        privileged = np.asarray(
            (
                self.privileged_files[fi]["observations_privileged"][start:stop]
                if "observations_privileged" in self.privileged_files[fi]
                else self.privileged_files[fi]["observations"][start:stop, 18:30]
            ),
            dtype=np.float32,
        )
        if counterfactual:
            events = np.zeros_like(events)
            action = np.zeros_like(action)
            privileged = np.zeros_like(privileged)
        return {"events": events, "state": state}, privileged, action

    def close(self):
        for file in self.event_files + self.privileged_files:
            file.close()


class CompactEventGRU(nn.Module):
    """Causal event encoder for trigger, TTC, miss vector, side and magnitude."""

    recurrent_type = "compact_gru"

    def __init__(self, base):
        super().__init__()
        self.features_extractor = base.features_extractor
        self.gru = base.gru
        hidden = int(base.hidden_size)
        self.trigger_head = nn.Linear(hidden, 1)
        self.tca_head = nn.Linear(hidden, 1)
        self.miss_head = nn.Linear(hidden, 2)
        self.side_head = nn.Linear(hidden, 3)
        self.magnitude_head = nn.Linear(hidden, 1)

    def preprocess(self, obs):
        obs = dict(obs)
        state = torch.zeros_like(obs["state"])
        # Ego velocity and angular rate are legitimate motion-compensation
        # inputs. Tracking error, commanded offset and previous action are not.
        state[..., 0:3] = obs["state"][..., 0:3]
        state[..., 12:15] = obs["state"][..., 12:15]
        obs["state"] = state
        obs.pop("privileged", None)
        return obs

    def forward_sequence(self, obs, hidden=None):
        obs = self.preprocess(obs)
        batch, steps = obs["state"].shape[:2]
        flat = {
            key: value.reshape(batch * steps, *value.shape[2:])
            for key, value in obs.items()
        }
        features = self.features_extractor(flat).reshape(batch, steps, -1)
        recurrent, hidden = self.gru(features, hidden)
        return {
            "trigger_logit": self.trigger_head(recurrent).squeeze(-1),
            "tca": torch.sigmoid(self.tca_head(recurrent).squeeze(-1)),
            "miss": torch.tanh(self.miss_head(recurrent)),
            "side_logits": self.side_head(recurrent),
            "magnitude": torch.sigmoid(self.magnitude_head(recurrent).squeeze(-1)),
        }, hidden

    def recurrent_step(self, obs, hidden=None):
        sequence = {key: value.unsqueeze(1) for key, value in obs.items()}
        output, hidden = self.forward_sequence(sequence, hidden)
        return {key: value[:, 0] for key, value in output.items()}, hidden


def build_compact_model(cfg, env, downsample_events=2, hidden_size=128):
    base, _ = build_gru_clone_net(
        cfg, env, downsample_events, hidden_size, blank_previous_action=True
    )
    return CompactEventGRU(base)


def compact_targets(privileged, action):
    tca = privileged[..., 11].clamp(0.0, TCA_SCALE_S)
    position = privileged[..., 1:4]
    velocity = privileged[..., 4:7]
    acceleration = privileged[..., 7:10]
    closest = (
        position
        + velocity * tca.unsqueeze(-1)
        + 0.5 * acceleration * tca.square().unsqueeze(-1)
    )
    lateral = action[..., 1]
    trigger = (lateral.abs() > DISCRETE_DEADBAND).float()
    side = torch.ones_like(lateral, dtype=torch.long)
    side[lateral < -DISCRETE_DEADBAND] = 0
    side[lateral > DISCRETE_DEADBAND] = 2
    approach = (privileged[..., 0] > 0.5) & (tca > 0.0) & (tca < 1.2)
    return {
        "trigger": trigger,
        "tca": tca / TCA_SCALE_S,
        "miss": (closest[..., 1:3] / MISS_SCALE_M).clamp(-1.0, 1.0),
        "side": side,
        "magnitude": lateral.abs().clamp(0.0, 1.0),
        "approach": approach,
    }


def structured_loss(output, target):
    trigger = target["trigger"]
    actionable = trigger > 0.5
    approach = target["approach"]
    trigger_loss = nn.functional.binary_cross_entropy_with_logits(
        output["trigger_logit"], trigger
    )
    side_loss = nn.functional.cross_entropy(
        output["side_logits"].reshape(-1, 3), target["side"].reshape(-1)
    )
    zero = output["trigger_logit"].sum() * 0.0
    tca_loss = (
        nn.functional.smooth_l1_loss(output["tca"][approach], target["tca"][approach])
        if bool(approach.any()) else zero
    )
    miss_loss = (
        nn.functional.smooth_l1_loss(output["miss"][approach], target["miss"][approach])
        if bool(approach.any()) else zero
    )
    magnitude_loss = (
        nn.functional.smooth_l1_loss(
            output["magnitude"][actionable], target["magnitude"][actionable]
        ) if bool(actionable.any()) else zero
    )
    temporal_loss = nn.functional.smooth_l1_loss(
        output["magnitude"][:, 1:] - output["magnitude"][:, :-1],
        target["magnitude"][:, 1:] - target["magnitude"][:, :-1],
    )
    total = (
        trigger_loss + side_loss + 0.5 * tca_loss + 0.5 * miss_loss
        + 0.5 * magnitude_loss + 0.1 * temporal_loss
    )
    return total, {
        "trigger_loss": trigger_loss,
        "side_loss": side_loss,
        "tca_loss": tca_loss,
        "miss_loss": miss_loss,
        "magnitude_loss": magnitude_loss,
        "temporal_loss": temporal_loss,
    }


@torch.no_grad()
def validation_metrics(model, dataset, device, batch_size):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    totals = {"trigger_correct": 0, "frames": 0, "side_correct": 0, "actionable": 0}
    squared = {"tca": 0.0, "miss": 0.0, "magnitude": 0.0}
    counts = {"tca": 0, "miss": 0, "magnitude": 0}
    for obs, privileged, action in loader:
        obs = {key: value.to(device) for key, value in obs.items()}
        obs["events"] = obs["events"].float()
        target = compact_targets(privileged.to(device), action.to(device))
        output, _ = model.forward_sequence(obs)
        predicted_trigger = torch.sigmoid(output["trigger_logit"]) >= 0.5
        totals["trigger_correct"] += int(
            (predicted_trigger == (target["trigger"] > 0.5)).sum()
        )
        totals["frames"] += int(target["trigger"].numel())
        actionable = target["trigger"] > 0.5
        totals["side_correct"] += int(
            (
                output["side_logits"].argmax(-1)[actionable]
                == target["side"][actionable]
            ).sum()
        )
        totals["actionable"] += int(actionable.sum())
        approach = target["approach"]
        squared["tca"] += float(((output["tca"][approach] - target["tca"][approach]) ** 2).sum())
        counts["tca"] += int(approach.sum())
        squared["miss"] += float(((output["miss"][approach] - target["miss"][approach]) ** 2).sum())
        counts["miss"] += int(approach.sum() * 2)
        squared["magnitude"] += float(
            (
                (output["magnitude"][actionable] - target["magnitude"][actionable])
                ** 2
            ).sum()
        )
        counts["magnitude"] += int(actionable.sum())
    return {
        "trigger_accuracy": totals["trigger_correct"] / max(totals["frames"], 1),
        "side_accuracy": totals["side_correct"] / max(totals["actionable"], 1),
        "tca_rmse_s": TCA_SCALE_S * np.sqrt(squared["tca"] / max(counts["tca"], 1)),
        "miss_rmse_m": MISS_SCALE_M * np.sqrt(squared["miss"] / max(counts["miss"], 1)),
        "magnitude_rmse": np.sqrt(squared["magnitude"] / max(counts["magnitude"], 1)),
    }


class CommitmentController:
    def __init__(
        self,
        hold_steps=7,
        on_threshold=0.5,
        off_threshold=0.3,
        alpha=0.35,
        min_magnitude=0.2,
        conditional_side=True,
        side_source="head",
    ):
        self.hold_steps = int(hold_steps)
        self.on_threshold = float(on_threshold)
        self.off_threshold = float(off_threshold)
        self.alpha = float(alpha)
        self.min_magnitude = float(min_magnitude)
        self.conditional_side = bool(conditional_side)
        if side_source not in {"head", "miss"}:
            raise ValueError(f"unknown side source: {side_source!r}")
        self.side_source = side_source
        self.reset()

    def reset(self):
        self.sign = 0
        self.remaining = 0
        self.magnitude = 0.0

    def action(self, output):
        trigger = float(torch.sigmoid(output["trigger_logit"])[0])
        if self.side_source == "miss":
            # Positive predicted lateral miss means the projectile passes on
            # the positive body-y side, so move in the opposite direction.
            # The sign is sampled only when a new commitment begins; later
            # frame-level noise therefore cannot reverse an active dodge.
            candidate = -1 if float(output["miss"][0, 0]) >= 0.0 else 1
        else:
            side_prob = torch.softmax(output["side_logits"][0], dim=-1)
            if self.conditional_side:
                # The trigger head already decides whether to act. Once it
                # fires, condition direction on the two dodge classes rather
                # than requiring either to beat the redundant "none" class.
                candidate = -1 if side_prob[0] >= side_prob[2] else 1
            else:
                side_class = int(side_prob.argmax())
                candidate = -1 if side_class == 0 else 1 if side_class == 2 else 0
        if self.sign == 0 and trigger >= self.on_threshold and candidate:
            self.sign = candidate
            self.remaining = self.hold_steps
        elif self.sign:
            self.remaining = max(self.remaining - 1, 0)
            if self.remaining == 0 and trigger < self.off_threshold:
                self.sign = 0
        desired = (
            max(float(output["magnitude"][0]), self.min_magnitude)
            if self.sign else 0.0
        )
        self.magnitude += self.alpha * (desired - self.magnitude)
        action = np.zeros(3, dtype=np.float32)
        action[1] = float(self.sign) * self.magnitude
        return action


def observation_batch(obs, device, downsample_events, zero_events=False):
    batch = {}
    for key, value in obs.items():
        if key == "privileged":
            continue
        array = np.asarray(value)
        if key == "events":
            array = pool_events(array, downsample_events)
            if zero_events:
                array = np.zeros_like(array)
        batch[key] = torch.from_numpy(array[None]).to(device)
    return batch


@torch.no_grad()
def evaluate(
    model,
    env,
    episodes,
    seed0,
    downsample_events,
    zero_events=False,
    controller_kwargs=None,
):
    device = next(model.parameters()).device
    successes = 0
    returns, lengths, accelerations = [], [], []
    terminations = {}
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed0 + episode)
        hidden = None
        controller = CommitmentController(**(controller_kwargs or {}))
        previous_velocity = np.asarray(env.sim.dynamics.state["v"], float).copy()
        policy_dt = env.policy_decimation * env.steps_per_action / env.sim.config.world_rate
        total_return = 0.0
        peak_acceleration = 0.0
        steps = 0
        while True:
            batch = observation_batch(obs, device, downsample_events, zero_events)
            output, hidden = model.recurrent_step(batch, hidden)
            action = controller.action(output)
            obs, reward, terminated, truncated, info = env.step(action)
            total_return += float(reward)
            velocity = np.asarray(env.sim.dynamics.state["v"], float).copy()
            peak_acceleration = max(peak_acceleration, float(np.linalg.norm(velocity - previous_velocity) / policy_dt))
            previous_velocity = velocity
            steps += 1
            if terminated or truncated:
                successes += int(bool(info.get("is_success", False)))
                reason = info.get("termination_reason", "truncated")
                terminations[reason] = terminations.get(reason, 0) + 1
                returns.append(total_return)
                lengths.append(steps)
                accelerations.append(peak_acceleration)
                break
    return {
        "success_rate": successes / max(episodes, 1),
        "mean_return": float(np.mean(returns)),
        "mean_steps": float(np.mean(lengths)),
        "terminations": terminations,
        "mean_peak_acceleration_mps2": float(np.mean(accelerations)),
        "max_peak_acceleration_mps2": float(max(accelerations, default=0.0)),
    }


def main():
    from neurosim.rl import env_class_for_task
    from train_sb3 import load_experiment_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--event-dataset", nargs="+", required=True)
    parser.add_argument("--privileged-dataset", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples-per-epoch", type=int, default=800)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--sequence-stride", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--downsample-events", type=int, default=2)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--holdout-seed", type=int, default=6262)
    parser.add_argument("--preload-events", action="store_true")
    parser.add_argument(
        "--warmstart",
        help="Optional GRU checkpoint providing encoder and recurrent weights.",
    )
    parser.add_argument(
        "--initialize-from",
        help="Optional compact-GRU checkpoint used to initialize every weight.",
    )
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seed0", type=int, default=16001)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load the existing best checkpoint and only run closed-loop eval.",
    )
    parser.add_argument(
        "--checkpoint",
        help=(
            "Checkpoint to load with --eval-only. Defaults to the .best.pt "
            "path derived from --output."
        ),
    )
    parser.add_argument(
        "--controller-json",
        default="{}",
        help=(
            "JSON object overriding CommitmentController arguments for "
            "closed-loop evaluation (for example, "
            "'{\"side_source\":\"miss\",\"on_threshold\":0.3}')."
        ),
    )
    parser.add_argument(
        "--skip-zero-event-eval",
        action="store_true",
        help="Skip the blank-event intervention during controller-only sweeps.",
    )
    args = parser.parse_args()

    controller_kwargs = json.loads(args.controller_json)
    if not isinstance(controller_kwargs, dict):
        raise ValueError("--controller-json must decode to an object")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    cfg = load_experiment_config(args.experiment_config)
    env_cfg = copy.deepcopy(cfg["env"])
    env_cfg["enable_visualization"] = False
    env = env_class_for_task(env_cfg["task"]["name"])(env_config=env_cfg, train=False)
    common = dict(
        sequence_length=args.sequence_length,
        downsample_events=args.downsample_events,
        holdout_fraction=args.holdout_fraction,
        holdout_seed=args.holdout_seed,
    )
    train = PairedCompactSequenceDataset(
        args.event_dataset, args.privileged_dataset,
        stride=args.sequence_stride, holdout_side="train", counterfactuals=True,
        preload_events=args.preload_events, **common,
    )
    test = PairedCompactSequenceDataset(
        args.event_dataset, args.privileged_dataset,
        stride=args.sequence_length, holdout_side="test", counterfactuals=False,
        shared_event_arrays=train.event_arrays, **common,
    )
    print(json.dumps({
        "train_sequences": len(train), "holdout_sequences": len(test),
        "train_episodes": train.selected_episodes, "holdout_episodes": test.selected_episodes,
        "sequence_groups": train.group_counts,
    }), flush=True)
    model = build_compact_model(cfg, env, args.downsample_events, args.hidden_size).to(device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_path = output_path.with_name(f"{output_path.stem}.best.pt")
    if args.eval_only:
        checkpoint_path = Path(args.checkpoint) if args.checkpoint else best_path
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        model.eval()
        real = evaluate(
            model, env, args.eval_episodes, args.eval_seed0,
            args.downsample_events, controller_kwargs=controller_kwargs,
        )
        blank = None
        if not args.skip_zero_event_eval:
            blank = evaluate(
                model, env, args.eval_episodes, args.eval_seed0,
                args.downsample_events, True, controller_kwargs=controller_kwargs,
            )
        result = {
            "best_epoch": int(payload["epoch"]),
            "best_validation_metric": float(payload["validation_metric"]),
            "best_checkpoint": str(checkpoint_path),
            "controller": controller_kwargs,
            "evaluation": real,
            "zero_event_evaluation": blank,
        }
        output_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        train.close()
        test.close()
        env.close()
        return
    if args.initialize_from:
        source = torch.load(args.initialize_from, map_location=device, weights_only=False)
        model.load_state_dict(source["model"])
        print(json.dumps({"initialize_from": args.initialize_from}), flush=True)
    elif args.warmstart:
        source = torch.load(args.warmstart, map_location=device, weights_only=False)
        current = model.state_dict()
        compatible = {
            key: value
            for key, value in source["model"].items()
            if key in current
            and current[key].shape == value.shape
            and (key.startswith("features_extractor.") or key.startswith("gru."))
        }
        if not compatible:
            raise ValueError(f"no compatible encoder weights in {args.warmstart}")
        model.load_state_dict(compatible, strict=False)
        print(
            json.dumps(
                {"warmstart": args.warmstart, "loaded_tensors": len(compatible)}
            ),
            flush=True,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.as_tensor(train.sequence_weights, dtype=torch.double),
        min(args.samples_per_epoch, len(train)), replacement=True,
    )
    loader = torch.utils.data.DataLoader(
        train, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
    )
    history = []
    best_metric, best_epoch = -float("inf"), -1
    try:
        for epoch in range(args.epochs):
            model.train()
            sums = {}
            batches = 0
            for obs, privileged, action in loader:
                obs = {key: value.to(device) for key, value in obs.items()}
                obs["events"] = obs["events"].float()
                target = compact_targets(privileged.to(device), action.to(device))
                prediction, _ = model.forward_sequence(obs)
                loss, parts = structured_loss(prediction, target)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                for key, value in {"loss": loss, **parts}.items():
                    sums[key] = sums.get(key, 0.0) + float(value.detach())
                batches += 1
            model.eval()
            row = {"epoch": epoch, **{key: value / batches for key, value in sums.items()},
                   **validation_metrics(model, test, device, args.batch_size)}
            metric = (
                0.5 * row["trigger_accuracy"]
                + 0.5 * row["side_accuracy"]
                - row["magnitude_rmse"]
            )
            if metric > best_metric:
                best_metric, best_epoch = metric, epoch
                torch.save({
                    "model": model.state_dict(), "epoch": int(epoch),
                    "validation_metric": float(metric),
                    "preprocessing": {"recurrent_type": "compact_gru", "hidden_size": args.hidden_size,
                                      "downsample_events": args.downsample_events,
                                      "controller": {"hold_steps": 7,
                                                     "on_threshold": 0.5,
                                                     "off_threshold": 0.3,
                                                     "alpha": 0.35}},
                }, output_path.with_name(f"{output_path.stem}.best.pt"))
            history.append(row)
            print(json.dumps(row), flush=True)
        payload = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        model.eval()
        real = evaluate(
            model, env, args.eval_episodes, args.eval_seed0,
            args.downsample_events, controller_kwargs=controller_kwargs,
        )
        blank = None
        if not args.skip_zero_event_eval:
            blank = evaluate(
                model, env, args.eval_episodes, args.eval_seed0,
                args.downsample_events, True, controller_kwargs=controller_kwargs,
            )
        output_path.write_text(json.dumps({
            "history": history, "best_epoch": best_epoch,
            "best_validation_metric": best_metric, "best_checkpoint": str(best_path),
            "controller": controller_kwargs,
            "evaluation": real, "zero_event_evaluation": blank,
        }, indent=2))
    finally:
        train.close()
        test.close()
        env.close()


if __name__ == "__main__":
    main()
