"""Evaluate a causal event-tracker-driven geometric dodge controller.

The learned models in this script are perception only: a frozen spatial
event localizer followed by a recurrent heatmap tracker.  Control is an
explicit state machine so a noisy sequence of frame-level centres cannot
produce the left/right switching seen in the BC policies.

The controller waits for consecutive high-confidence detections, estimates a
short image-plane motion vector, chooses a direction away from the predicted
centre, and commits to that direction for a fixed pulse.  It cannot recommit
until the tracked object has been absent for a refractory interval.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_event_obstacle_localizer import SpatialEventLocalizer  # noqa: E402
from train_event_obstacle_tracker import TemporalHeatmapTracker  # noqa: E402
from train_sb3 import load_experiment_config  # noqa: E402
from neurosim.rl import env_class_for_task  # noqa: E402
from neurosim.core.coord_trans import rotate_vector_by_quat  # noqa: E402


@dataclass(frozen=True)
class TrackerPrediction:
    probability: float
    centre: np.ndarray
    geometry: np.ndarray
    inbound_probability: float = 0.0
    features: np.ndarray | None = None


class EventObstacleTracker:
    """One-step causal inference wrapper for the two-stage event tracker."""

    def __init__(
        self,
        spatial_checkpoint: str | Path,
        temporal_checkpoint: str | Path,
        device: str = "cuda:0",
    ):
        self.device = torch.device(device)
        spatial_payload = torch.load(
            spatial_checkpoint, map_location=self.device, weights_only=False
        )
        input_shape = tuple(int(value) for value in spatial_payload["input_shape"])
        self.input_shape = input_shape
        self.spatial = SpatialEventLocalizer(in_channels=input_shape[0]).to(self.device)
        self.spatial.load_state_dict(spatial_payload["model"])
        self.spatial.eval()

        temporal_payload = torch.load(
            temporal_checkpoint, map_location=self.device, weights_only=False
        )
        self.temporal = TemporalHeatmapTracker().to(self.device)
        temporal_state = temporal_payload["model"]
        self.has_inbound_head = "inbound_head.weight" in temporal_state
        missing, unexpected = self.temporal.load_state_dict(
            temporal_state, strict=False
        )
        allowed_missing = {
            "inbound_head.weight",
            "inbound_head.bias",
            *{
                f"inbound_gru.{name}"
                for name in self.temporal.inbound_gru.state_dict()
            },
        }
        if set(missing) - allowed_missing or unexpected:
            raise ValueError(
                f"incompatible temporal checkpoint: missing={missing}, "
                f"unexpected={unexpected}"
            )
        if not any(key.startswith("inbound_gru.") for key in temporal_state):
            self.temporal.inbound_gru.load_state_dict(
                self.temporal.gru.state_dict()
            )
        self.temporal.eval()
        self.hidden = None

    def reset(self) -> None:
        self.hidden = None

    @torch.no_grad()
    def step(self, events: np.ndarray, *, blank_events: bool = False) -> TrackerPrediction:
        array = np.asarray(events)
        if tuple(array.shape) != self.input_shape:
            raise ValueError(
                f"event shape {tuple(array.shape)} does not match checkpoint "
                f"input shape {self.input_shape}"
            )
        event_tensor = torch.from_numpy(array[None]).to(
            device=self.device, dtype=torch.float32
        )
        if blank_events:
            event_tensor.zero_()

        spatial = self.spatial(event_tensor)
        map_height, map_width = spatial["heatmap_logits"].shape[-2:]
        probability = torch.softmax(
            spatial["heatmap_logits"].flatten(1) / 0.25, dim=-1
        ).reshape(-1, map_height, map_width)
        coarse = F.adaptive_avg_pool2d(probability[:, None], (30, 40))[:, 0]
        coarse = coarse * (map_height * map_width / (30 * 40))
        entropy = -(probability * probability.clamp_min(1e-12).log()).sum(
            (1, 2)
        ) / np.log(map_height * map_width)
        auxiliary = torch.cat(
            [
                torch.sigmoid(spatial["presence_logit"])[:, None],
                spatial["centre"],
                spatial["geometry"],
                coarse.amax((1, 2))[:, None],
                entropy[:, None],
            ],
            dim=-1,
        )
        temporal = self.temporal(
            coarse[:, None], auxiliary[:, None], self.hidden
        )
        self.hidden = temporal["hidden"]
        return TrackerPrediction(
            probability=float(torch.sigmoid(temporal["presence_logit"])[0, 0]),
            centre=temporal["centre"][0, 0].cpu().numpy(),
            geometry=temporal["geometry"][0, 0].cpu().numpy(),
            inbound_probability=(
                float(torch.sigmoid(temporal["inbound_logit"])[0, 0])
                if self.has_inbound_head
                else 0.0
            ),
            features=temporal["features"][0, 0].cpu().numpy(),
        )


class LatchedGeometryController:
    """Convert tracked centres into one smooth, non-reversing dodge pulse."""

    def __init__(
        self,
        *,
        threshold: float = 0.99,
        track_threshold: float = 0.7,
        minimum_track_age_steps: int = 12,
        inbound_threshold: float = 1.1,
        inbound_confirm_steps: int = 2,
        track_reset_absence_steps: int = 3,
        confirm_steps: int = 2,
        hold_steps: int = 20,
        refractory_absence_steps: int = 8,
        magnitude: float = 1.0,
        motion_alpha: float = 0.5,
        motion_lead_steps: float = 3.0,
        centre_deadband: float = 0.015,
        control_mode: str = "image_plane",
        fallback_lateral_sign: int = 1,
    ):
        if control_mode not in {"lateral", "image_plane"}:
            raise ValueError(f"unknown control mode: {control_mode!r}")
        self.threshold = float(threshold)
        self.track_threshold = float(track_threshold)
        self.minimum_track_age_steps = max(int(minimum_track_age_steps), 0)
        self.inbound_threshold = float(inbound_threshold)
        self.inbound_confirm_steps = max(int(inbound_confirm_steps), 1)
        self.track_reset_absence_steps = max(int(track_reset_absence_steps), 1)
        self.confirm_steps = max(int(confirm_steps), 1)
        self.hold_steps = max(int(hold_steps), 1)
        self.refractory_absence_steps = max(int(refractory_absence_steps), 1)
        self.magnitude = float(magnitude)
        self.motion_alpha = float(motion_alpha)
        self.motion_lead_steps = float(motion_lead_steps)
        self.centre_deadband = float(centre_deadband)
        self.control_mode = control_mode
        self.fallback_lateral_sign = 1 if fallback_lateral_sign >= 0 else -1
        self.reset()

    def reset(self) -> None:
        self.confirmed = 0
        self.track_age = 0
        self.track_absent_steps = 0
        self.hold_remaining = 0
        self.absent_steps = 0
        self.refractory = False
        self.previous_centre: np.ndarray | None = None
        self.motion = np.zeros(2, dtype=np.float64)
        self.direction = np.zeros(3, dtype=np.float32)
        self.pending_direction: np.ndarray | None = None
        self.pending_impact_centre = np.full(2, np.nan, dtype=np.float64)
        self.inbound_confirmed = 0
        self.commitments = 0
        self.last_impact_centre = np.full(2, np.nan, dtype=np.float64)
        self.last_release_reason = "none"

    def _choose_direction(self, impact_centre: np.ndarray) -> np.ndarray:
        offset = np.asarray(impact_centre, dtype=np.float64) - 0.5
        # Horizontal image-right corresponds to body -Y for a forward camera;
        # moving body +Y is therefore away from an obstacle on image-right.
        # Image-down is body -Z, so body +Z similarly moves away.
        lateral = float(offset[0])
        vertical = float(offset[1])
        if abs(lateral) < self.centre_deadband:
            lateral = float(self.fallback_lateral_sign)
        if self.control_mode == "lateral":
            return np.asarray([0.0, np.sign(lateral), 0.0], dtype=np.float32)

        vector = np.asarray([lateral, vertical], dtype=np.float64)
        if float(np.linalg.norm(vector)) < self.centre_deadband:
            vector = np.asarray([float(self.fallback_lateral_sign), 0.0])
        # Normalize in command space. The environment independently enforces
        # the physical velocity and acceleration bounds on the resulting
        # body-frame command.
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        return np.asarray([0.0, vector[0], vector[1]], dtype=np.float32)

    def step(self, prediction: TrackerPrediction) -> tuple[np.ndarray, bool]:
        detected = prediction.probability >= self.threshold
        track_visible = prediction.probability >= self.track_threshold
        centre = np.asarray(prediction.centre, dtype=np.float64)
        committed_now = False

        if track_visible:
            self.track_age += 1
            self.track_absent_steps = 0
        else:
            self.track_absent_steps += 1
            if self.track_absent_steps >= self.track_reset_absence_steps:
                self.track_age = 0
                if self.hold_remaining == 0 and not self.refractory:
                    self.pending_direction = None
                    self.confirmed = 0

        if detected:
            if self.previous_centre is not None:
                delta = centre - self.previous_centre
                self.motion += self.motion_alpha * (delta - self.motion)
            self.previous_centre = centre.copy()
            self.absent_steps = 0
        else:
            self.previous_centre = None
            self.motion *= 0.5
            self.absent_steps += 1

        inbound_detected = (
            track_visible
            and prediction.inbound_probability >= self.inbound_threshold
        )
        self.inbound_confirmed = (
            self.inbound_confirmed + 1 if inbound_detected else 0
        )

        if self.hold_remaining > 0:
            self.hold_remaining -= 1
            if self.hold_remaining == 0:
                self.refractory = True
            return self.direction * self.magnitude, committed_now

        if self.refractory:
            if self.absent_steps >= self.refractory_absence_steps:
                self.refractory = False
                self.confirmed = 0
                self.inbound_confirmed = 0
            return np.zeros(3, dtype=np.float32), committed_now

        if self.pending_direction is None:
            self.confirmed = self.confirmed + 1 if detected else 0
        if self.pending_direction is None and self.confirmed >= self.confirm_steps:
            impact = np.clip(
                centre + self.motion_lead_steps * self.motion, 0.0, 1.0
            )
            self.pending_impact_centre = impact
            self.pending_direction = self._choose_direction(impact)
            self.confirmed = 0

        age_ready = self.track_age >= self.minimum_track_age_steps
        inbound_ready = self.inbound_confirmed >= self.inbound_confirm_steps
        ready = age_ready or inbound_ready
        if self.pending_direction is not None and ready:
            self.last_impact_centre = self.pending_impact_centre.copy()
            self.direction = self.pending_direction.copy()
            self.pending_direction = None
            self.hold_remaining = self.hold_steps - 1
            self.commitments += 1
            committed_now = True
            self.last_release_reason = "age" if age_ready else "inbound"
            if self.hold_remaining == 0:
                self.refractory = True
            return self.direction * self.magnitude, committed_now

        return np.zeros(3, dtype=np.float32), committed_now


class EventGeometryPolicy:
    """Predict-style adapter shared by numeric and video evaluation."""

    def __init__(
        self,
        spatial_checkpoint: str | Path,
        temporal_checkpoint: str | Path,
        *,
        device: str = "cuda:0",
        blank_events: bool = False,
        controller_kwargs: dict[str, Any] | None = None,
    ):
        self.tracker = EventObstacleTracker(
            spatial_checkpoint, temporal_checkpoint, device=device
        )
        self.controller = LatchedGeometryController(**(controller_kwargs or {}))
        self.blank_events = bool(blank_events)
        self.last_prediction: TrackerPrediction | None = None
        self.last_committed = False

    def reset(self) -> None:
        self.tracker.reset()
        self.controller.reset()
        self.last_prediction = None
        self.last_committed = False

    def predict(self, observation, deterministic: bool = True):
        del deterministic
        if not isinstance(observation, dict) or "events" not in observation:
            raise ValueError("event geometry policy requires a dict observation with events")
        self.last_prediction = self.tracker.step(
            observation["events"], blank_events=self.blank_events
        )
        action, self.last_committed = self.controller.step(self.last_prediction)
        return action, None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--spatial-checkpoint", required=True)
    parser.add_argument("--temporal-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--dataset",
        help=(
            "Optional streaming HDF5 of controller-visited event frames with "
            "privileged image labels. Labels are recorded only, never input."
        ),
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed0", type=int, default=30001)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sim-gpu-id", type=int, default=0)
    parser.add_argument("--blank-events", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.99)
    parser.add_argument("--track-threshold", type=float, default=0.7)
    parser.add_argument("--minimum-track-age-steps", type=int, default=12)
    parser.add_argument(
        "--inbound-threshold",
        type=float,
        default=1.1,
        help=(
            "Inbound emergency-release threshold. The safe default disables "
            "the rejected experimental override; pass 0.98 to reproduce it."
        ),
    )
    parser.add_argument("--inbound-confirm-steps", type=int, default=2)
    parser.add_argument("--track-reset-absence-steps", type=int, default=3)
    parser.add_argument("--confirm-steps", type=int, default=2)
    parser.add_argument("--hold-steps", type=int, default=20)
    parser.add_argument("--refractory-absence-steps", type=int, default=8)
    parser.add_argument("--magnitude", type=float, default=1.0)
    parser.add_argument("--motion-alpha", type=float, default=0.5)
    parser.add_argument("--motion-lead-steps", type=float, default=3.0)
    parser.add_argument("--centre-deadband", type=float, default=0.015)
    parser.add_argument(
        "--control-mode", choices=("lateral", "image_plane"), default="image_plane"
    )
    parser.add_argument("--fallback-lateral-sign", type=int, choices=(-1, 1), default=1)
    return parser.parse_args()


def _write_payload(path: Path, args, results: list[dict[str, Any]]) -> None:
    encounters_total = sum(int(row["encounters_total"]) for row in results)
    encounters_cleared = sum(int(row["encounters_cleared"]) for row in results)
    payload = {
        "experiment_config": args.experiment_config,
        "spatial_checkpoint": args.spatial_checkpoint,
        "temporal_checkpoint": args.temporal_checkpoint,
        "blank_events": bool(args.blank_events),
        "controller": {
            "threshold": args.threshold,
            "track_threshold": args.track_threshold,
            "minimum_track_age_steps": args.minimum_track_age_steps,
            "inbound_threshold": args.inbound_threshold,
            "inbound_confirm_steps": args.inbound_confirm_steps,
            "track_reset_absence_steps": args.track_reset_absence_steps,
            "confirm_steps": args.confirm_steps,
            "hold_steps": args.hold_steps,
            "refractory_absence_steps": args.refractory_absence_steps,
            "magnitude": args.magnitude,
            "motion_alpha": args.motion_alpha,
            "motion_lead_steps": args.motion_lead_steps,
            "centre_deadband": args.centre_deadband,
            "control_mode": args.control_mode,
            "fallback_lateral_sign": args.fallback_lateral_sign,
        },
        "episodes": len(results),
        "success_rate": float(np.mean([row["success"] for row in results]))
        if results
        else 0.0,
        "encounters_total": encounters_total,
        "encounters_cleared": encounters_cleared,
        "encounter_clear_rate": encounters_cleared / max(encounters_total, 1),
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _privileged_commitment_diagnostics(env) -> dict[str, Any]:
    """Evaluation-only truth for diagnosing a controller commitment.

    Nothing returned here is passed to either the tracker or controller.
    Keeping it next to the causal action trace lets us distinguish perception
    timing, coordinate mapping, and insufficient control authority afterward.
    """
    from evaluate_velocity_dodge_oracle import obstacle_image_label

    rows = env._obstacle_relative_states(env.sim.dynamics.state) or []
    diagnostic: dict[str, Any] = {
        "true_image_label": obstacle_image_label(env).tolist(),
    }
    if not rows:
        return diagnostic
    row = rows[0]
    quaternion = np.asarray(env.sim.dynamics.state["q"], dtype=np.float64)
    rel_pos = rotate_vector_by_quat(
        np.asarray(row["rel_pos"], dtype=np.float64), quaternion, inverse=True
    )
    rel_vel = rotate_vector_by_quat(
        np.asarray(row["rel_vel"], dtype=np.float64), quaternion, inverse=True
    )
    rel_accel = rotate_vector_by_quat(
        np.asarray(row["rel_accel"], dtype=np.float64), quaternion, inverse=True
    )
    tca = float(row["time_to_closest_approach"])
    closest = rel_pos + rel_vel * tca + 0.5 * rel_accel * tca * tca
    diagnostic.update(
        {
            "encounter_id": int(row["object_id"]),
            "centre_distance_m": float(np.linalg.norm(rel_pos)),
            "time_to_closest_approach_s": tca,
            "true_rel_pos_body": rel_pos.tolist(),
            "true_closest_body": closest.tolist(),
        }
    )
    return diagnostic


def main() -> None:
    args = _parse_args()
    cfg = load_experiment_config(args.experiment_config)
    env_cfg = copy.deepcopy(cfg["env"])
    env_cfg["enable_visualization"] = False
    env_cfg.setdefault("visual_backend", {})["gpu_id"] = args.sim_gpu_id
    env_cls = env_class_for_task(env_cfg["task"]["name"])
    env = env_cls(env_config=env_cfg, train=False)
    controller_kwargs = {
        "threshold": args.threshold,
        "track_threshold": args.track_threshold,
        "minimum_track_age_steps": args.minimum_track_age_steps,
        "inbound_threshold": args.inbound_threshold,
        "inbound_confirm_steps": args.inbound_confirm_steps,
        "track_reset_absence_steps": args.track_reset_absence_steps,
        "confirm_steps": args.confirm_steps,
        "hold_steps": args.hold_steps,
        "refractory_absence_steps": args.refractory_absence_steps,
        "magnitude": args.magnitude,
        "motion_alpha": args.motion_alpha,
        "motion_lead_steps": args.motion_lead_steps,
        "centre_deadband": args.centre_deadband,
        "control_mode": args.control_mode,
        "fallback_lateral_sign": args.fallback_lateral_sign,
    }
    policy = EventGeometryPolicy(
        args.spatial_checkpoint,
        args.temporal_checkpoint,
        device=args.device,
        blank_events=args.blank_events,
        controller_kwargs=controller_kwargs,
    )
    seeds = (
        [int(seed) for seed in args.seeds]
        if args.seeds is not None
        else [args.seed0 + episode for episode in range(args.episodes)]
    )
    output = Path(args.output)
    results: list[dict[str, Any]] = []
    dataset_writer = None
    if args.dataset:
        from evaluate_velocity_dodge_oracle import HDF5ImitationWriter

        dataset_writer = HDF5ImitationWriter(
            Path(args.dataset), args.experiment_config
        )
    try:
        for episode, seed in enumerate(seeds):
            observation, _ = env.reset(seed=seed)
            policy.reset()
            previous_velocity = np.asarray(env.sim.dynamics.state["v"], float).copy()
            policy_dt = float(
                env.policy_decimation
                * env.steps_per_action
                / env.sim.config.world_rate
            )
            total_reward = 0.0
            peak_acceleration = 0.0
            commitment_rows = []
            episode_observations = []
            episode_actions = []
            steps = 0
            while True:
                action, _ = policy.predict(observation)
                if dataset_writer is not None:
                    from evaluate_velocity_dodge_oracle import obstacle_image_label

                    episode_observations.append(
                        {
                            "events": np.asarray(
                                observation["events"], dtype=np.float16
                            ).copy(),
                            "state": np.asarray(
                                observation["state"], dtype=np.float32
                            ).copy(),
                            "obstacle_image": obstacle_image_label(env),
                        }
                    )
                    episode_actions.append(
                        np.asarray(action, dtype=np.float32).copy()
                    )
                if policy.last_committed:
                    prediction = policy.last_prediction
                    commitment_rows.append(
                        {
                            "step": steps,
                            "time_s": steps * policy_dt,
                            "probability": prediction.probability,
                            "inbound_probability": prediction.inbound_probability,
                            "release_reason": policy.controller.last_release_reason,
                            "centre": prediction.centre.tolist(),
                            "predicted_radius_px": float(
                                prediction.geometry[0] * 10.0
                            ),
                            "predicted_depth_m": float(
                                prediction.geometry[1] * 20.0
                            ),
                            "predicted_impact_centre": (
                                policy.controller.last_impact_centre.tolist()
                            ),
                            "action": action.tolist(),
                            "diagnostic_truth_not_used_by_controller": (
                                _privileged_commitment_diagnostics(env)
                            ),
                        }
                    )
                observation, reward, terminated, truncated, info = env.step(action)
                velocity = np.asarray(env.sim.dynamics.state["v"], float).copy()
                peak_acceleration = max(
                    peak_acceleration,
                    float(np.linalg.norm(velocity - previous_velocity) / policy_dt),
                )
                previous_velocity = velocity
                total_reward += float(reward)
                steps += 1
                if terminated or truncated:
                    terms = info.get("reward_terms", {}) or {}
                    row = {
                        "episode": episode,
                        "seed": seed,
                        "success": bool(info.get("is_success", False)),
                        "termination_reason": info.get(
                            "termination_reason", "truncated" if truncated else "none"
                        ),
                        "steps": steps,
                        "reward": total_reward,
                        "episode_min_clearance": float(
                            terms.get("episode_min_clearance", np.inf)
                        ),
                        "encounters_total": int(terms.get("encounters_total", 0)),
                        "encounters_cleared": int(terms.get("encounters_cleared", 0)),
                        "peak_acceleration_mps2": peak_acceleration,
                        "commitments": policy.controller.commitments,
                        "commitment_rows": commitment_rows,
                    }
                    results.append(row)
                    if dataset_writer is not None:
                        dataset_writer.append_episode(
                            episode_observations,
                            episode_actions,
                            episode=episode,
                            seed=seed,
                        )
                    _write_payload(output, args, results)
                    print(json.dumps(row), flush=True)
                    break
    finally:
        # High-resolution Habitat currently segfaults during teardown on this
        # branch. Results are flushed after every episode above so a teardown
        # fault cannot discard a completed evaluation.
        if dataset_writer is not None:
            dataset_writer.close()
        env.close()
    _write_payload(output, args, results)
    print(f"summary={output}", flush=True)


if __name__ == "__main__":
    main()
