"""Task registry for Neurosim RL."""

from .base import RewardOutcome, RLTask, TaskStep
from .hover_stop import HoverStopTask
from .reactive_dodge import ReactiveDodgeTask
from .velocity_dodge import VelocityDodgeTask


def build_task(task_name: str, **kwargs) -> RLTask:
    name = task_name.strip().lower()
    if name == "hover_stop":
        return HoverStopTask(**kwargs)
    if name == "reactive_dodge":
        return ReactiveDodgeTask(**kwargs)
    if name == "velocity_dodge":
        return VelocityDodgeTask(**kwargs)
    raise ValueError(f"Unsupported task_name: {task_name}")


__all__ = [
    "HoverStopTask",
    "ReactiveDodgeTask",
    "VelocityDodgeTask",
    "RewardOutcome",
    "RLTask",
    "TaskStep",
    "build_task",
]
