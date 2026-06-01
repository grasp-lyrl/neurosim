"""GPU placement helpers for multi-worker (multi-process) workloads.

Shared by RL training (vectorized Habitat envs) and the online data-generation
pipeline (parallel simulator producers). Keeps worker→GPU assignment logic in
one place so both stacks place workers identically.
"""

import logging

logger = logging.getLogger(__name__)


def sim_gpu_assignments(num_envs: int, n_gpus: int) -> list[int]:
    """Map each worker index to a GPU id in ``0 .. n_gpus - 1``.

    For ``n`` parallel workers and ``g`` GPUs with ``g >= 2``:

    - ``max(0, n // g - 2)`` workers are assigned to GPU ``0`` (roughly two fewer
      than an even ``n / g`` share, so a training/policy process can keep GPU 0
      busier).
    - The remaining ``n - count0`` workers are round-robined over GPUs
      ``1 .. g - 1``.

    For ``g <= 1`` or ``n <= 0``, every worker uses GPU ``0``.
    """
    n, g = num_envs, int(n_gpus)
    if g == 1 or n <= 0:
        return [0] * n

    count0 = max(0, n // g - 2)
    assign: list[int] = [0] * count0

    rest = n - count0
    for i in range(rest):
        assign.append(1 + (i % (g - 1)))
    return assign


def explicit_gpu_map(num_workers: int, gpu_ids: list[int] | None) -> list[int]:
    """Resolve an explicit per-worker GPU list, cycling ``gpu_ids`` as needed.

    Used by the data-gen config's ``data.gpu_map`` (e.g. ``[0]*8`` to place 8
    simulator producers on ``gpu0``). When ``gpu_ids`` is ``None`` every worker
    uses GPU ``0``.

    Args:
        num_workers: Number of workers to place.
        gpu_ids: Explicit GPU ids; cycled if shorter than ``num_workers``.
    """
    if not gpu_ids:
        return [0] * max(0, num_workers)
    return [int(gpu_ids[i % len(gpu_ids)]) for i in range(max(0, num_workers))]
