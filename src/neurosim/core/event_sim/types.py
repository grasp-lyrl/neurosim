import torch
from typing import NamedTuple


class Events(NamedTuple):
    """Container for a batch of events returned by event simulators."""

    x: torch.Tensor
    y: torch.Tensor
    t: torch.Tensor
    p: torch.Tensor
