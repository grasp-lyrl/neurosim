"""Task registry for Neurosim RL."""

from .base import RLTask
from .hover_stop import HoverStopTask


def build_task(task_name: str, **kwargs) -> RLTask:
    name = task_name.strip().lower()
    if name == "hover_stop":
        return HoverStopTask(**kwargs)
    raise ValueError(f"Unsupported task_name: {task_name}")


__all__ = ["RLTask", "HoverStopTask", "build_task"]
