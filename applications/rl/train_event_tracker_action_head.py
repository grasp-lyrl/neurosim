"""Train a deployable GRU action head on frozen full-resolution event tracks.

The oracle may begin moving before the learned event tracker has acquired the
projectile.  Such labels are impossible for an event policy to imitate.  This
script causally gates each expert action sequence: oracle commands remain zero
until the tracker has observed the obstacle for consecutive frames, then the
rest of that observable encounter is retained.  Privileged image labels are
never model inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_event_geometry_controller import EventObstacleTracker  # noqa: E402


EGO_INDICES = np.asarray([0, 1, 2, 3, 4, 5, 12, 13, 14], dtype=np.int64)
TRACKER_FEATURE_DIM = 128
EXPLICIT_TRACK_DIM = 6
INPUT_DIM = TRACKER_FEATURE_DIM + EXPLICIT_TRACK_DIM + len(EGO_INDICES)


@dataclass
class CachedEpisode:
    key: str
    inputs: np.ndarray
    actions: np.ndarray
    original_actions: np.ndarray
    probabilities: np.ndarray
    label_valid: np.ndarray
    action_valid: np.ndarray | None = None


class EventTrackerActionHead(nn.Module):
    """Second recurrent stage mapping causal event-track features to actions."""

    def __init__(self, input_dim: int = INPUT_DIM, hidden_size: int = 128):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.gru = nn.GRU(self.input_dim, self.hidden_size, batch_first=True)
        self.trunk = nn.Sequential(
            nn.Linear(self.hidden_size, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU()
        )
        self.gate_head = nn.Linear(64, 1)
        # Only body-y/body-z are learned. Forward/back motion created the
        # blind braking shortcut in earlier policies and carries little oracle
        # energy here (5.3% of active frames are x-dominant).
        self.direction_head = nn.Linear(64, 2)

    def forward(self, inputs, hidden=None):
        recurrent, hidden = self.gru(inputs, hidden)
        latent = self.trunk(recurrent)
        gate_logit = self.gate_head(latent)[..., 0]
        direction = torch.tanh(self.direction_head(latent))
        yz = torch.sigmoid(gate_logit)[..., None] * direction
        action = torch.cat([torch.zeros_like(yz[..., :1]), yz], dim=-1)
        return {
            "gate_logit": gate_logit,
            "direction": direction,
            "action": action,
            "hidden": hidden,
        }


class LearnedPulseController:
    """Latch one learned y/z direction into a smooth non-reversing pulse."""

    def __init__(
        self,
        *,
        gate_threshold: float = 0.5,
        confirm_steps: int = 2,
        hold_steps: int = 20,
        ramp_steps: int = 3,
        refractory_absence_steps: int = 8,
        magnitude: float = 1.0,
        minimum_direction_norm: float = 0.1,
        minimum_hold_steps: int | None = None,
        release_inbound_threshold: float | None = None,
        release_confirm_steps: int = 3,
        renewal_extension_steps: int = 0,
        renewal_confirm_steps: int = 2,
        maximum_hold_steps: int | None = None,
        renewal_direction_cosine: float = 0.0,
    ):
        self.gate_threshold = float(gate_threshold)
        self.confirm_steps = max(int(confirm_steps), 1)
        self.hold_steps = max(int(hold_steps), 1)
        self.ramp_steps = max(int(ramp_steps), 0)
        self.refractory_absence_steps = max(int(refractory_absence_steps), 1)
        self.magnitude = float(magnitude)
        self.minimum_direction_norm = float(minimum_direction_norm)
        self.minimum_hold_steps = (
            self.hold_steps
            if minimum_hold_steps is None
            else max(min(int(minimum_hold_steps), self.hold_steps), 1)
        )
        self.release_inbound_threshold = (
            None
            if release_inbound_threshold is None
            else float(release_inbound_threshold)
        )
        self.release_confirm_steps = max(int(release_confirm_steps), 1)
        self.renewal_extension_steps = max(int(renewal_extension_steps), 0)
        self.renewal_confirm_steps = max(int(renewal_confirm_steps), 1)
        if maximum_hold_steps is None:
            maximum_hold_steps = self.hold_steps + self.renewal_extension_steps
        self.maximum_hold_steps = max(int(maximum_hold_steps), self.hold_steps)
        self.renewal_direction_cosine = float(renewal_direction_cosine)
        if not -1.0 <= self.renewal_direction_cosine <= 1.0:
            raise ValueError("renewal_direction_cosine must be in [-1, 1]")
        self.reset()

    def reset(self):
        self.confirmed = 0
        self.pulse_step = None
        self.direction = np.zeros(2, dtype=np.float32)
        self.refractory = False
        self.absent_steps = 0
        self.release_confirmed = 0
        self.release_step = None
        self.early_releases = 0
        self.early_release_pulse_steps = []
        self.pulse_end_step = self.hold_steps
        self.renewal_confirmed = 0
        self.renewals = 0
        self.renewal_pulse_steps = []
        self.last_renewed = False
        self.commitments = 0
        self.last_committed = False

    def _envelope(self, step: int) -> float:
        if self.ramp_steps == 0:
            return 1.0
        attack = min((step + 1) / self.ramp_steps, 1.0)
        release = min((self.pulse_end_step - 1 - step) / self.ramp_steps, 1.0)
        return float(max(min(attack, release), 0.0))

    def _finish_pulse(self) -> None:
        self.pulse_step = None
        self.release_step = None
        self.release_confirmed = 0
        self.renewal_confirmed = 0
        self.refractory = True
        self.absent_steps = 0

    def _maybe_renew(
        self,
        gate_probability: float,
        direction: np.ndarray,
        *,
        track_present: bool,
    ) -> None:
        """Selectively postpone release without changing the latched direction."""
        if (
            self.renewal_extension_steps <= 0
            or self.pulse_end_step >= self.maximum_hold_steps
            or self.release_step is not None
        ):
            self.renewal_confirmed = 0
            return
        # Only recent evidence near the release ramp may renew a pulse. This
        # avoids turning an early detection into an unconditional long hold.
        window_start = (
            self.pulse_end_step
            - max(self.ramp_steps, 1)
            - self.renewal_confirm_steps
        )
        if int(self.pulse_step) < window_start:
            self.renewal_confirmed = 0
            return

        vector = np.asarray(direction, dtype=np.float32).reshape(2)
        norm = float(np.linalg.norm(vector))
        compatible = bool(
            norm >= self.minimum_direction_norm
            and float(np.dot(vector / norm, self.direction))
            >= self.renewal_direction_cosine
        )
        detected = (
            track_present
            and float(gate_probability) >= self.gate_threshold
            and compatible
        )
        self.renewal_confirmed = self.renewal_confirmed + 1 if detected else 0
        if self.renewal_confirmed < self.renewal_confirm_steps:
            return

        old_end = self.pulse_end_step
        self.pulse_end_step = min(
            self.maximum_hold_steps,
            self.pulse_end_step + self.renewal_extension_steps,
        )
        self.renewal_confirmed = 0
        if self.pulse_end_step > old_end:
            self.renewals += 1
            self.renewal_pulse_steps.append(int(self.pulse_step))
            self.last_renewed = True

    def _pulse_action(self, *, threat_passed: bool = False) -> np.ndarray:
        step = int(self.pulse_step)
        if step >= self.minimum_hold_steps and self.release_inbound_threshold is not None:
            self.release_confirmed = (
                self.release_confirmed + 1 if threat_passed else 0
            )
            if (
                self.release_step is None
                and self.release_confirmed >= self.release_confirm_steps
            ):
                self.release_step = 0
                self.early_releases += 1
                self.early_release_pulse_steps.append(step)

        action = np.zeros(3, dtype=np.float32)
        if self.release_step is None:
            envelope = self._envelope(step)
        elif self.ramp_steps == 0:
            envelope = 0.0
        else:
            envelope = max(
                (self.ramp_steps - 1 - int(self.release_step)) / self.ramp_steps,
                0.0,
            )
        action[1:] = self.direction * (self.magnitude * envelope)
        step += 1
        if self.release_step is not None:
            self.release_step += 1
        if (
            step >= self.pulse_end_step
            or self.release_step is not None
            and self.release_step >= max(self.ramp_steps, 1)
        ):
            self._finish_pulse()
        else:
            self.pulse_step = step
        return action

    def step(
        self,
        gate_probability,
        direction,
        *,
        track_present: bool,
        threat_passed: bool = False,
    ):
        self.last_committed = False
        self.last_renewed = False
        if self.pulse_step is not None:
            self._maybe_renew(
                gate_probability,
                direction,
                track_present=track_present,
            )
            return self._pulse_action(threat_passed=threat_passed)
        if self.refractory:
            self.absent_steps = 0 if track_present else self.absent_steps + 1
            if self.absent_steps >= self.refractory_absence_steps:
                self.refractory = False
                self.confirmed = 0
            return np.zeros(3, dtype=np.float32)

        detected = track_present and float(gate_probability) >= self.gate_threshold
        self.confirmed = self.confirmed + 1 if detected else 0
        if self.confirmed < self.confirm_steps:
            return np.zeros(3, dtype=np.float32)
        vector = np.asarray(direction, dtype=np.float32).reshape(2)
        norm = float(np.linalg.norm(vector))
        if norm < self.minimum_direction_norm:
            self.confirmed = 0
            return np.zeros(3, dtype=np.float32)
        self.direction = vector / norm
        self.pulse_step = 0
        self.pulse_end_step = self.hold_steps
        self.renewal_confirmed = 0
        self.release_confirmed = 0
        self.release_step = None
        self.confirmed = 0
        self.commitments += 1
        self.last_committed = True
        return self._pulse_action()


def commitment_targets(actions: np.ndarray, label_valid: np.ndarray) -> np.ndarray:
    """Replace each active oracle run by its one normalized pulse direction."""
    source = np.asarray(actions, dtype=np.float32)
    valid = np.asarray(label_valid, dtype=bool)
    output = np.zeros_like(source)
    active = (np.linalg.norm(source[:, 1:], axis=1) > 0.1) & valid
    padded = np.r_[False, active, False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    stops = np.flatnonzero(padded[:-1] & ~padded[1:])
    for start, stop in zip(starts, stops, strict=True):
        vector = source[start:stop, 1:].sum(axis=0)
        norm = float(np.linalg.norm(vector))
        if norm > 1e-6:
            output[start:stop, 1:] = vector / norm
    return output


@torch.no_grad()
def rebase_input_normalization(model, old_mean, old_std, new_mean, new_std):
    """Preserve the GRU function when dataset normalization statistics change."""
    old_mean = torch.as_tensor(old_mean, device=model.gru.weight_ih_l0.device)
    old_std = torch.as_tensor(old_std, device=model.gru.weight_ih_l0.device)
    new_mean = torch.as_tensor(new_mean, device=model.gru.weight_ih_l0.device)
    new_std = torch.as_tensor(new_std, device=model.gru.weight_ih_l0.device)
    weight = model.gru.weight_ih_l0.detach().clone()
    scale = new_std / old_std
    offset = (new_mean - old_mean) / old_std
    model.gru.weight_ih_l0.copy_(weight * scale[None, :])
    model.gru.bias_ih_l0.add_(weight @ offset)


def tracker_input(prediction, state: np.ndarray) -> np.ndarray:
    features = np.asarray(prediction.features, dtype=np.float32).reshape(-1)
    if features.shape != (TRACKER_FEATURE_DIM,):
        raise ValueError(f"unexpected tracker feature shape {features.shape}")
    explicit = np.asarray(
        [
            prediction.probability,
            *(2.0 * (np.asarray(prediction.centre) - 0.5)),
            *(2.0 * np.asarray(prediction.geometry) - 1.0),
            prediction.inbound_probability,
        ],
        dtype=np.float32,
    )
    ego = np.asarray(state, dtype=np.float32).reshape(-1)[EGO_INDICES]
    return np.concatenate([features, explicit, ego]).astype(np.float32)


def causal_visibility_mask(
    probabilities: np.ndarray,
    event_activity: np.ndarray,
    *,
    threshold: float,
    confirm_steps: int,
    reset_absence_steps: int,
) -> np.ndarray:
    """Latch causal visibility, resetting only after sustained absence."""
    visible = (probabilities >= threshold) & (event_activity > 0.0)
    output = np.zeros(len(visible), dtype=bool)
    confirmed = 0
    absent = 0
    latched = False
    for index, detected in enumerate(visible):
        if detected:
            confirmed += 1
            absent = 0
            if confirmed >= confirm_steps:
                latched = True
        else:
            confirmed = 0
            absent += 1
            if absent >= reset_absence_steps:
                latched = False
        output[index] = latched
    return output


def _episode_slices(file: h5py.File):
    episode_index = np.asarray(file["episode_index"])
    starts = np.flatnonzero(np.r_[True, episode_index[1:] != episode_index[:-1]])
    stops = np.r_[starts[1:], len(episode_index)]
    return zip(starts, stops, strict=True)


@torch.no_grad()
def cache_episodes(
    paths: list[str],
    tracker: EventObstacleTracker,
    *,
    threshold: float,
    confirm_steps: int,
    reset_absence_steps: int,
    action_labels_valid: bool = True,
) -> list[CachedEpisode]:
    episodes: list[CachedEpisode] = []
    for path in paths:
        with h5py.File(path, "r") as file:
            for start, stop in _episode_slices(file):
                tracker.reset()
                inputs, probabilities, activity = [], [], []
                for row in range(int(start), int(stop)):
                    events = np.asarray(file["observations_events"][row])
                    prediction = tracker.step(events)
                    inputs.append(
                        tracker_input(prediction, file["observations_state"][row])
                    )
                    probabilities.append(prediction.probability)
                    activity.append(float(np.any(np.abs(events) > 1e-6)))
                original = np.asarray(file["actions"][start:stop], dtype=np.float32)
                visible = causal_visibility_mask(
                    np.asarray(probabilities),
                    np.asarray(activity),
                    threshold=threshold,
                    confirm_steps=confirm_steps,
                    reset_absence_steps=reset_absence_steps,
                )
                actions = original.copy()
                actions[~visible] = 0.0
                actions[:, 0] = 0.0
                episode = int(file["episode_index"][start])
                seed = int(file["episode_seed"][start])
                episodes.append(
                    CachedEpisode(
                        key=f"{Path(path).name}:episode{episode}:seed{seed}",
                        inputs=np.asarray(inputs, dtype=np.float32),
                        actions=actions,
                        original_actions=original,
                        probabilities=np.asarray(probabilities, dtype=np.float32),
                        label_valid=np.ones(len(inputs), dtype=bool),
                        action_valid=np.full(
                            len(inputs), action_labels_valid, dtype=bool
                        ),
                    )
                )
    return episodes


def load_feature_episodes(
    paths: list[str], *, action_labels_valid: bool = True
) -> list[CachedEpisode]:
    """Load compact feature shards written by the online oracle collector."""
    episodes: list[CachedEpisode] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload.get("input_dim", -1)) != INPUT_DIM:
            raise ValueError(f"feature shard {path} has incompatible input width")
        for row in payload["episodes"]:
            label_valid = np.asarray(
                row.get("label_valid", np.ones(len(row["inputs"]), dtype=bool)),
                dtype=bool,
            )
            default_action_valid = (
                label_valid.copy()
                if action_labels_valid
                else np.zeros(len(row["inputs"]), dtype=bool)
            )
            episodes.append(
                CachedEpisode(
                    key=str(row["key"]),
                    inputs=np.asarray(row["inputs"], dtype=np.float32),
                    actions=np.asarray(row["actions"], dtype=np.float32),
                    original_actions=np.asarray(
                        row["original_actions"], dtype=np.float32
                    ),
                    probabilities=np.asarray(row["probabilities"], dtype=np.float32),
                    label_valid=label_valid,
                    action_valid=np.asarray(
                        row.get("action_valid", default_action_valid),
                        dtype=bool,
                    ),
                )
            )
    return episodes


def padded_batch(episodes: list[CachedEpisode], indices: np.ndarray, mean, std, device):
    selected = [episodes[int(index)] for index in indices]
    length = max(len(episode.inputs) for episode in selected)
    batch = len(selected)
    inputs = np.zeros((batch, length, INPUT_DIM), dtype=np.float32)
    actions = np.zeros((batch, length, 3), dtype=np.float32)
    valid = np.zeros((batch, length), dtype=bool)
    action_valid = np.zeros((batch, length), dtype=bool)
    for row, episode in enumerate(selected):
        count = len(episode.inputs)
        inputs[row, :count] = (episode.inputs - mean) / std
        actions[row, :count] = episode.actions
        valid[row, :count] = episode.label_valid
        if episode.action_valid is None:
            action_valid[row, :count] = episode.label_valid
        else:
            action_valid[row, :count] = episode.action_valid
    return (
        torch.from_numpy(inputs).to(device),
        torch.from_numpy(actions).to(device),
        torch.from_numpy(valid).to(device),
        torch.from_numpy(action_valid).to(device),
    )


def action_loss(output, target, valid, action_valid=None):
    if action_valid is None:
        action_valid = valid
    active = torch.linalg.vector_norm(target[..., 1:], dim=-1) > 0.1
    gate_target = active.float()
    valid_active = active & action_valid
    valid_flat = valid
    positives = max(int(valid_active.sum()), 1)
    negatives = max(int((valid & ~active).sum()), 1)
    pos_weight = torch.as_tensor(
        min(negatives / positives, 5.0), device=target.device
    )
    gate = nn.functional.binary_cross_entropy_with_logits(
        output["gate_logit"][valid_flat],
        gate_target[valid_flat],
        pos_weight=pos_weight,
    )
    weights = 1.0 + 4.0 * active.float()
    per_frame = nn.functional.smooth_l1_loss(
        output["action"][..., 1:], target[..., 1:], reduction="none"
    ).mean(-1)
    if bool(action_valid.any()):
        action = (
            per_frame[action_valid] * weights[action_valid]
        ).sum() / weights[action_valid].sum()
    else:
        action = output["action"].sum() * 0.0
    transition_valid = action_valid[:, 1:] & action_valid[:, :-1]
    predicted_delta = output["action"][:, 1:, 1:] - output["action"][:, :-1, 1:]
    target_delta = target[:, 1:, 1:] - target[:, :-1, 1:]
    if bool(transition_valid.any()):
        temporal = nn.functional.smooth_l1_loss(
            predicted_delta[transition_valid], target_delta[transition_valid]
        )
    else:
        temporal = output["action"].sum() * 0.0
    total = action + 0.5 * gate + 0.1 * temporal
    return total, {"action": action, "gate": gate, "temporal": temporal}


def gate_classification_loss(output, target, valid, *, positive_weight_cap=5.0):
    """Train threat/no-threat classification without altering dodge direction.

    This is useful for real-event negative controls: their zero action is a
    trustworthy *gate* label, but it is not a direction-regression target.
    """
    active = torch.linalg.vector_norm(target[..., 1:], dim=-1) > 0.1
    positives = max(int((active & valid).sum()), 1)
    negatives = max(int((~active & valid).sum()), 1)
    pos_weight = torch.as_tensor(
        min(negatives / positives, float(positive_weight_cap)),
        device=target.device,
    )
    return nn.functional.binary_cross_entropy_with_logits(
        output["gate_logit"][valid],
        active.float()[valid],
        pos_weight=pos_weight,
    )


def checkpoint_selection_score(values, *, gate_only=False):
    if gate_only:
        # A structured pulse normalizes the direction vector at commitment, so
        # gate-induced action magnitude is not a useful selection criterion.
        # Prefer precision while retaining a meaningful recall contribution.
        return 2.0 * values["trigger_precision"] + values["trigger_recall"]
    return (
        values["trigger_f1"]
        + values["active_direction_agreement"]
        - 0.5 * values["active_yz_mae"]
    )


@torch.no_grad()
def metrics(model, episodes, mean, std, device):
    inputs, target, valid, action_valid = padded_batch(
        episodes, np.arange(len(episodes)), mean, std, device
    )
    output = model(inputs)
    predicted_gate = torch.sigmoid(output["gate_logit"]) >= 0.5
    actual_gate = torch.linalg.vector_norm(target[..., 1:], dim=-1) > 0.1
    tp = int((predicted_gate & actual_gate & valid).sum())
    fp = int((predicted_gate & ~actual_gate & valid).sum())
    fn = int((~predicted_gate & actual_gate & valid).sum())
    active = actual_gate & valid & action_valid
    predicted_yz = output["action"][..., 1:]
    target_yz = target[..., 1:]
    mae = float(torch.abs(predicted_yz[active] - target_yz[active]).mean())
    cosine = nn.functional.cosine_similarity(predicted_yz[active], target_yz[active])
    return {
        "frames": int(valid.sum()),
        "active_frames": int(active.sum()),
        "trigger_precision": tp / max(tp + fp, 1),
        "trigger_recall": tp / max(tp + fn, 1),
        "trigger_f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "active_yz_mae": mae,
        "active_direction_cosine": float(cosine.mean()),
        "active_direction_agreement": float((cosine > 0.0).float().mean()),
    }


class EventActionPolicy:
    """Deployment adapter: raw event observation to recurrent body-y/z action."""

    def __init__(
        self,
        checkpoint: str | Path,
        device="cuda:0",
        intervention="real",
        structured_pulse: bool | None = None,
        pulse_kwargs: dict[str, Any] | None = None,
        track_presence_threshold: float = 0.7,
        minimum_inbound_probability: float | None = None,
    ):
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        prep = payload["preprocessing"]
        self.device = torch.device(device)
        self.tracker = EventObstacleTracker(
            prep["spatial_checkpoint"], prep["temporal_checkpoint"], device=device
        )
        self.model = EventTrackerActionHead(
            input_dim=int(prep["input_dim"]), hidden_size=int(prep["hidden_size"])
        ).to(self.device)
        self.model.load_state_dict(payload["model"])
        self.model.eval()
        self.mean = np.asarray(prep["input_mean"], dtype=np.float32)
        self.std = np.asarray(prep["input_std"], dtype=np.float32)
        self.gate_threshold = float(prep.get("deployment_gate_threshold", 0.5))
        self.track_presence_threshold = float(track_presence_threshold)
        self.minimum_inbound_probability = (
            None
            if minimum_inbound_probability is None
            else float(minimum_inbound_probability)
        )
        if structured_pulse is None:
            structured_pulse = bool(prep.get("structured_commitment", False))
        self.structured_pulse = bool(structured_pulse)
        default_pulse = dict(prep.get("pulse_controller", {}))
        default_pulse.update(pulse_kwargs or {})
        default_pulse["gate_threshold"] = self.gate_threshold
        self.pulse_controller = (
            LearnedPulseController(**default_pulse) if self.structured_pulse else None
        )
        if intervention not in {"real", "blank", "reverse_history"}:
            raise ValueError(f"unknown intervention {intervention!r}")
        self.intervention = intervention
        self.hidden = None
        self.last_gate_probability = 0.0
        self.last_direction = np.zeros(2, dtype=np.float32)
        self.last_event_active = False
        self.last_input = None
        self.last_prediction = None
        self.last_committed = False

    def reset(self):
        self.tracker.reset()
        self.hidden = None
        self.last_gate_probability = 0.0
        self.last_direction.fill(0.0)
        self.last_event_active = False
        self.last_input = None
        self.last_prediction = None
        self.last_committed = False
        if self.pulse_controller is not None:
            self.pulse_controller.reset()

    @torch.no_grad()
    def infer(self, observation):
        """Advance the frozen tracker/GRU once, without advancing control."""
        events = np.asarray(observation["events"])
        blank = self.intervention == "blank"
        if self.intervention == "reverse_history":
            history = events.reshape(-1, 2, *events.shape[-2:])
            events = history[::-1].copy().reshape(events.shape)
        prediction = self.tracker.step(events, blank_events=blank)
        self.last_prediction = prediction
        state = np.asarray(observation["state"], dtype=np.float32)
        self.last_input = tracker_input(prediction, state)
        vector = (self.last_input - self.mean) / self.std
        inputs = torch.from_numpy(vector[None, None]).to(self.device)
        output = self.model(inputs, self.hidden)
        self.hidden = output["hidden"]
        self.last_gate_probability = float(torch.sigmoid(output["gate_logit"])[0, 0])
        self.last_direction = (
            output["direction"][0, 0].cpu().numpy().astype(np.float32, copy=True)
        )
        self.last_event_active = (not blank) and bool(
            np.any(np.abs(events) > 1e-6)
        )
        return self.last_gate_probability, self.last_direction.copy()

    def action_from_inference(self, gate_probability=None, direction=None):
        """Advance control from the most recent frozen-network inference."""
        if self.last_prediction is None:
            raise RuntimeError("infer() must be called before action_from_inference()")
        if gate_probability is None:
            gate_probability = self.last_gate_probability
        if direction is None:
            direction = self.last_direction
        result = np.zeros(3, dtype=np.float32)
        self.last_committed = False
        if self.pulse_controller is not None:
            track_present = (
                self.last_event_active
                and self.last_prediction.probability >= self.track_presence_threshold
                and (
                    self.minimum_inbound_probability is None
                    or self.last_prediction.inbound_probability
                    >= self.minimum_inbound_probability
                )
            )
            release_threshold = self.pulse_controller.release_inbound_threshold
            result = self.pulse_controller.step(
                gate_probability,
                direction,
                track_present=track_present,
                threat_passed=(
                    track_present
                    and release_threshold is not None
                    and self.last_prediction.inbound_probability
                    <= release_threshold
                ),
            )
            self.last_committed = self.pulse_controller.last_committed
        elif float(gate_probability) >= self.gate_threshold:
            result[1:] = direction
        return result, None

    @torch.no_grad()
    def predict(self, observation, deterministic=True):
        del deterministic
        self.infer(observation)
        return self.action_from_inference()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", nargs="*", default=[])
    parser.add_argument(
        "--train-gate-only-dataset",
        nargs="*",
        default=[],
        help="Raw HDF5 datasets that supervise presence but not dodge vectors.",
    )
    parser.add_argument("--train-feature-dataset", nargs="*", default=[])
    parser.add_argument(
        "--train-gate-only-feature-dataset",
        nargs="*",
        default=[],
        help="Feature shards whose labels supervise the gate but not action regression.",
    )
    parser.add_argument("--validation-dataset", nargs="*", default=[])
    parser.add_argument(
        "--validation-gate-only-dataset", nargs="*", default=[]
    )
    parser.add_argument("--validation-feature-dataset", nargs="*", default=[])
    parser.add_argument(
        "--validation-gate-only-feature-dataset", nargs="*", default=[]
    )
    parser.add_argument("--spatial-checkpoint", required=True)
    parser.add_argument("--temporal-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--visibility-threshold", type=float, default=0.7)
    parser.add_argument("--confirm-steps", type=int, default=2)
    parser.add_argument("--reset-absence-steps", type=int, default=3)
    parser.add_argument("--deployment-gate-threshold", type=float, default=0.5)
    parser.add_argument("--structured-commitment", action="store_true")
    parser.add_argument(
        "--deploy-structured-pulse",
        action="store_true",
        help=(
            "Package the learned framewise head with the structured pulse "
            "controller without replacing training targets by commitment targets."
        ),
    )
    parser.add_argument("--pulse-confirm-steps", type=int, default=2)
    parser.add_argument("--pulse-hold-steps", type=int, default=20)
    parser.add_argument("--pulse-ramp-steps", type=int, default=3)
    parser.add_argument("--pulse-refractory-absence-steps", type=int, default=8)
    parser.add_argument("--pulse-magnitude", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=4141)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Also save epoch snapshots at this interval (0 disables snapshots).",
    )
    parser.add_argument(
        "--init-checkpoint",
        help="Optional compatible action-head checkpoint used to initialize training.",
    )
    parser.add_argument(
        "--gate-only-finetune",
        action="store_true",
        help=(
            "Freeze the recurrent representation and direction head, and train "
            "only the threat gate. Requires --init-checkpoint and preserves its "
            "input normalization."
        ),
    )
    parser.add_argument(
        "--gate-positive-weight-cap",
        type=float,
        default=5.0,
        help="Maximum positive-class weight used by gate-only fine-tuning.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tracker = EventObstacleTracker(
        args.spatial_checkpoint, args.temporal_checkpoint, device=args.device
    )
    common = dict(
        threshold=args.visibility_threshold,
        confirm_steps=args.confirm_steps,
        reset_absence_steps=args.reset_absence_steps,
    )
    if not (
        args.train_dataset
        or args.train_gate_only_dataset
        or args.train_feature_dataset
        or args.train_gate_only_feature_dataset
    ):
        raise ValueError("at least one training or compact feature dataset is required")
    print("caching frozen tracker outputs for training", flush=True)
    train_episodes = cache_episodes(args.train_dataset, tracker, **common)
    train_episodes.extend(
        cache_episodes(
            args.train_gate_only_dataset,
            tracker,
            action_labels_valid=False,
            **common,
        )
    )
    train_episodes.extend(load_feature_episodes(args.train_feature_dataset))
    train_episodes.extend(
        load_feature_episodes(
            args.train_gate_only_feature_dataset, action_labels_valid=False
        )
    )
    if not (
        args.validation_dataset
        or args.validation_gate_only_dataset
        or args.validation_feature_dataset
        or args.validation_gate_only_feature_dataset
    ):
        raise ValueError("at least one validation or compact validation dataset is required")
    print("caching frozen tracker outputs for validation", flush=True)
    validation_episodes = cache_episodes(args.validation_dataset, tracker, **common)
    validation_episodes.extend(
        cache_episodes(
            args.validation_gate_only_dataset,
            tracker,
            action_labels_valid=False,
            **common,
        )
    )
    validation_episodes.extend(load_feature_episodes(args.validation_feature_dataset))
    validation_episodes.extend(
        load_feature_episodes(
            args.validation_gate_only_feature_dataset, action_labels_valid=False
        )
    )
    if args.structured_commitment:
        for episode in [*train_episodes, *validation_episodes]:
            episode.actions = commitment_targets(episode.actions, episode.label_valid)
    model = EventTrackerActionHead(hidden_size=args.hidden_size).to(device)
    initial = None
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(initial["model"])
        print(f"initialized action head from {args.init_checkpoint}", flush=True)
    if args.gate_only_finetune and initial is None:
        raise ValueError("--gate-only-finetune requires --init-checkpoint")
    if args.gate_positive_weight_cap <= 0.0:
        raise ValueError("--gate-positive-weight-cap must be positive")

    if args.gate_only_finetune:
        # Keep the exact feature coordinate system in which the frozen GRU was
        # trained. Only the small linear gate is allowed to learn the new
        # real-event negatives; the maneuver representation cannot regress.
        initial_prep = initial["preprocessing"]
        mean = np.asarray(initial_prep["input_mean"], dtype=np.float32)
        std = np.asarray(initial_prep["input_std"], dtype=np.float32)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.gate_head.parameters():
            parameter.requires_grad_(True)
        optimized_parameters = model.gate_head.parameters()
    else:
        all_train = np.concatenate([episode.inputs for episode in train_episodes])
        mean = all_train.mean(0).astype(np.float32)
        std = np.maximum(all_train.std(0), 1e-4).astype(np.float32)
        if initial is not None:
            initial_prep = initial["preprocessing"]
            rebase_input_normalization(
                model,
                initial_prep["input_mean"],
                initial_prep["input_std"],
                mean,
                std,
            )
        optimized_parameters = model.parameters()
    optimizer = torch.optim.AdamW(
        optimized_parameters, lr=args.learning_rate, weight_decay=1e-5
    )
    rng = np.random.default_rng(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_score = -float("inf")
    history: list[dict[str, Any]] = []
    preprocessing = {
        "recurrent_type": "event_tracker_action_gru",
        "input_dim": INPUT_DIM,
        "hidden_size": args.hidden_size,
        "input_mean": mean.tolist(),
        "input_std": std.tolist(),
        "spatial_checkpoint": args.spatial_checkpoint,
        "temporal_checkpoint": args.temporal_checkpoint,
        "visibility_threshold": args.visibility_threshold,
        "confirm_steps": args.confirm_steps,
        "reset_absence_steps": args.reset_absence_steps,
        "deployment_gate_threshold": args.deployment_gate_threshold,
        "ego_indices": EGO_INDICES.tolist(),
        "learned_action_axes": [1, 2],
        "structured_commitment": bool(
            args.structured_commitment or args.deploy_structured_pulse
        ),
        "pulse_controller": {
            "confirm_steps": args.pulse_confirm_steps,
            "hold_steps": args.pulse_hold_steps,
            "ramp_steps": args.pulse_ramp_steps,
            "refractory_absence_steps": args.pulse_refractory_absence_steps,
            "magnitude": args.pulse_magnitude,
        },
    }
    model.eval()
    initial_train_metrics = metrics(model, train_episodes, mean, std, device)
    initial_validation_metrics = metrics(
        model, validation_episodes, mean, std, device
    )
    best_score = checkpoint_selection_score(
        initial_validation_metrics, gate_only=args.gate_only_finetune
    )
    initial_row = {
        "epoch": 0,
        "loss": None,
        "train": initial_train_metrics,
        "validation": initial_validation_metrics,
        "selection_score": best_score,
    }
    history.append(initial_row)
    print(json.dumps(initial_row), flush=True)
    torch.save(
        {"model": model.state_dict(), "preprocessing": preprocessing, "epoch": 0},
        output,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train_episodes))
        losses = []
        for start in range(0, len(order), args.batch_size):
            inputs, target, valid, action_valid = padded_batch(
                train_episodes, order[start : start + args.batch_size], mean, std, device
            )
            prediction = model(inputs)
            if args.gate_only_finetune:
                loss = gate_classification_loss(
                    prediction,
                    target,
                    valid,
                    positive_weight_cap=args.gate_positive_weight_cap,
                )
            else:
                loss, _ = action_loss(
                    prediction, target, valid, action_valid=action_valid
                )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        train_metrics = metrics(model, train_episodes, mean, std, device)
        validation_metrics = metrics(model, validation_episodes, mean, std, device)
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(row)
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps(row), flush=True)
        selection_score = checkpoint_selection_score(
            validation_metrics, gate_only=args.gate_only_finetune
        )
        row["selection_score"] = selection_score
        if selection_score > best_score:
            best_score = selection_score
            torch.save(
                {"model": model.state_dict(), "preprocessing": preprocessing, "epoch": epoch},
                output,
            )
        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            snapshot = output.with_name(
                f"{output.stem}.epoch{epoch:03d}{output.suffix}"
            )
            torch.save(
                {"model": model.state_dict(), "preprocessing": preprocessing, "epoch": epoch},
                snapshot,
            )

    last_output = output.with_name(f"{output.stem}.last{output.suffix}")
    torch.save(
        {"model": model.state_dict(), "preprocessing": preprocessing, "epoch": args.epochs},
        last_output,
    )

    payload = {
        "train_datasets": args.train_dataset,
        "train_gate_only_datasets": args.train_gate_only_dataset,
        "train_feature_datasets": args.train_feature_dataset,
        "train_gate_only_feature_datasets": args.train_gate_only_feature_dataset,
        "validation_datasets": args.validation_dataset,
        "validation_gate_only_datasets": args.validation_gate_only_dataset,
        "validation_feature_datasets": args.validation_feature_dataset,
        "validation_gate_only_feature_datasets": (
            args.validation_gate_only_feature_dataset
        ),
        "init_checkpoint": args.init_checkpoint,
        "gate_only_finetune": args.gate_only_finetune,
        "gate_positive_weight_cap": args.gate_positive_weight_cap,
        "train_episodes": len(train_episodes),
        "validation_episodes": len(validation_episodes),
        "train_retained_actions": int(
            sum((np.linalg.norm(e.actions[:, 1:], axis=1) > 0.1).sum() for e in train_episodes)
        ),
        "train_original_actions": int(
            sum((np.linalg.norm(e.original_actions, axis=1) > 0.1).sum() for e in train_episodes)
        ),
        "best_selection_score": best_score,
        "last_checkpoint": str(last_output),
        "checkpoint_every": args.checkpoint_every,
        "history": history,
        "preprocessing": preprocessing,
    }
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2))
    print(f"saved {output} and {output.with_suffix('.json')}", flush=True)


if __name__ == "__main__":
    main()
