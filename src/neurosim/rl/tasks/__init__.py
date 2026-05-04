"""Task registry for Neurosim RL."""

from .base import EventRepresentationManager, RLTask
from .hover_stop import HoverStopTask
from .trajectory_dodge import TrajectoryDodgeTask


def build_task(task_name: str, **kwargs) -> RLTask:
    name = task_name.strip().lower()
    if name == "hover_stop":
        return HoverStopTask(**kwargs)
    if name == "trajectory_dodge":
        return TrajectoryDodgeTask(**kwargs)
    raise ValueError(f"Unsupported task_name: {task_name}")


__all__ = [
    "EventRepresentationManager",
    "RLTask",
    "HoverStopTask",
    "TrajectoryDodgeTask",
    "build_task",
]
