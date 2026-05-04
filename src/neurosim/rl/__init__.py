from .env import BaseNeurosimRLEnv, HoverStopEnv
from .env_reactive_dodge import ReactiveDodgeEnv
from .representations import EventRepresentationManager
from .sb3_features import CombinedEventStateExtractor, EventCnnExtractor
from .tasks import (
    HoverStopTask,
    ReactiveDodgeTask,
    RewardOutcome,
    RLTask,
    TaskStep,
    build_task,
)
from .vehicles import RLVehicle, RotorpyCtbrVehicle, build_vehicle


ENV_BY_TASK = {
    "hover_stop": HoverStopEnv,
    "reactive_dodge": ReactiveDodgeEnv,
}


def env_class_for_task(task_name: str) -> type[BaseNeurosimRLEnv]:
    """Return the concrete env class registered for ``task_name``.

    Library-style helper for entrypoints (``train_sb3.py``, ``run_policy.py``)
    that build an env from a YAML config: pass in ``env_config["task"]["name"]``
    and instantiate the returned class with ``env_config=...``.
    """
    if task_name not in ENV_BY_TASK:
        known = ", ".join(sorted(ENV_BY_TASK))
        raise ValueError(
            f"No env class registered for task {task_name!r}. Known: {known}."
        )
    return ENV_BY_TASK[task_name]


__all__ = [
    "BaseNeurosimRLEnv",
    "HoverStopEnv",
    "ReactiveDodgeEnv",
    "EventCnnExtractor",
    "EventRepresentationManager",
    "CombinedEventStateExtractor",
    "RLTask",
    "TaskStep",
    "RewardOutcome",
    "HoverStopTask",
    "ReactiveDodgeTask",
    "build_task",
    "RLVehicle",
    "RotorpyCtbrVehicle",
    "build_vehicle",
    "ENV_BY_TASK",
    "env_class_for_task",
]
