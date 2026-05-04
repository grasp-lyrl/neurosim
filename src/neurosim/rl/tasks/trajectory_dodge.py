"""Trajectory-tracking obstacle-dodge task implementation.

The quadrotor follows a pre-planned free-space MinSnap trajectory while a
learned policy outputs small velocity corrections to dodge dynamic obstacles
thrown at the agent.  An SE3 controller in the loop handles low-level attitude
control; the policy only adjusts the *desired velocity* fed to the controller.
"""

import numpy as np

from .base import RLTask, EventRepresentationManager


class TrajectoryDodgeTask(RLTask):
    """Reward and success logic for trajectory tracking + obstacle avoidance.

    The task stores per-step desired state from the trajectory so the reward
    can measure tracking error.  The environment is responsible for calling
    :meth:`set_desired_state` each step before :meth:`compute_reward`.

    Parameters
    ----------
    w_tracking : float
        Weight for position tracking error penalty.
    w_vel_tracking : float
        Weight for velocity tracking error penalty.
    w_correction : float
        Weight for penalizing large policy corrections (encourages
        minimal intervention).
    w_events : float
        Weight for event-activity penalty (optional, 0 to disable).
    w_survival : float
        Per-step survival bonus.
    crash_penalty : float
        Penalty applied when the episode terminates due to collision.
    success_position_threshold : float
        L2 distance to final waypoint required for success (meters).
    max_tracking_error : float
        If tracking error exceeds this value the episode terminates.
    """

    def __init__(
        self,
        w_tracking: float,
        w_vel_tracking: float,
        w_correction: float,
        w_events: float,
        w_survival: float,
        crash_penalty: float,
        success_position_threshold: float,
        max_tracking_error: float,
    ):
        self.w_tracking = float(w_tracking)
        self.w_vel_tracking = float(w_vel_tracking)
        self.w_correction = float(w_correction)
        self.w_events = float(w_events)
        self.w_survival = float(w_survival)
        self._crash_penalty = float(crash_penalty)
        self.success_position_threshold = float(success_position_threshold)
        self.max_tracking_error = float(max_tracking_error)

        # Per-step desired state, set by the environment before compute_reward.
        self._desired_position: np.ndarray = np.zeros(3, dtype=np.float32)
        self._desired_velocity: np.ndarray = np.zeros(3, dtype=np.float32)

        # Last correction applied by the policy (set by env).
        self._last_correction: np.ndarray = np.zeros(3, dtype=np.float32)

        # Whether the trajectory has been fully traversed.
        self._trajectory_complete: bool = False

    @property
    def crash_penalty(self) -> float:
        return self._crash_penalty

    def on_reset(self) -> None:
        self._desired_position = np.zeros(3, dtype=np.float32)
        self._desired_velocity = np.zeros(3, dtype=np.float32)
        self._last_correction = np.zeros(3, dtype=np.float32)
        self._trajectory_complete = False

    def set_desired_state(
        self,
        desired_position: np.ndarray,
        desired_velocity: np.ndarray,
    ) -> None:
        """Called by the environment each step to update the trajectory target."""
        self._desired_position = np.asarray(desired_position, dtype=np.float32)
        self._desired_velocity = np.asarray(desired_velocity, dtype=np.float32)

    def set_last_correction(self, correction: np.ndarray) -> None:
        """Called by the environment after scaling the policy action."""
        self._last_correction = np.asarray(correction, dtype=np.float32)

    def set_trajectory_complete(self, complete: bool) -> None:
        """Called by the environment when the trajectory time is exceeded."""
        self._trajectory_complete = complete

    def compute_reward(
        self,
        *,
        state: dict[str, np.ndarray],
        action: np.ndarray,
        prev_action: np.ndarray | None,
        event_manager: EventRepresentationManager,
        obs_mode: str,
    ) -> tuple[float, dict[str, float]]:
        x = np.asarray(state["x"], dtype=np.float32)
        v = np.asarray(state["v"], dtype=np.float32)

        # Position tracking error
        tracking_error = float(np.linalg.norm(x - self._desired_position))

        # Velocity tracking error
        vel_error = float(np.linalg.norm(v - self._desired_velocity))

        # Correction magnitude
        correction_norm = float(np.linalg.norm(self._last_correction))

        # Event activity (optional)
        if obs_mode == "state":
            event_activity_density = 0.0
        else:
            event_activity = event_manager.step_event_count
            event_activity_density = event_activity / (
                event_manager.raw_height * event_manager.raw_width
            )

        # Reward terms
        r_tracking = -self.w_tracking * tracking_error
        r_vel_tracking = -self.w_vel_tracking * vel_error
        r_correction = -self.w_correction * correction_norm
        r_events = -self.w_events * event_activity_density
        r_survival = self.w_survival

        reward = r_tracking + r_vel_tracking + r_correction + r_events + r_survival

        return reward, {
            "tracking_error": tracking_error,
            "vel_error": vel_error,
            "correction_norm": correction_norm,
            "event_activity_density": event_activity_density,
            "r_tracking": r_tracking,
            "r_vel_tracking": r_vel_tracking,
            "r_correction": r_correction,
            "r_events": r_events,
            "r_survival": r_survival,
        }

    def check_success(self, *, state: dict[str, np.ndarray]) -> bool:
        if not self._trajectory_complete:
            return False
        x = np.asarray(state["x"], dtype=np.float32)
        dist = float(np.linalg.norm(x - self._desired_position))
        return dist < self.success_position_threshold

    def check_terminated(self, *, state: dict[str, np.ndarray]) -> tuple[bool, str]:
        x = np.asarray(state["x"], dtype=np.float32)
        tracking_error = float(np.linalg.norm(x - self._desired_position))
        if tracking_error > self.max_tracking_error:
            return True, "max_tracking_error_exceeded"
        return False, ""
