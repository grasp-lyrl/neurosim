"""Random square crop parameters for training."""

import torch
from torch import Tensor


def get_random_crop_params(
    input_size: tuple[int, int], output_size: tuple[int, int], batch_size: int = 1
) -> Tensor:
    """One random [y0, x0, y1, x1] crop box per batch element, as an int tensor [B, 4]."""
    w, h = input_size
    tw, th = output_size
    if w == tw and h == th:
        return torch.tensor([[0, 0, w, h]] * batch_size)

    i = torch.randint(0, w - tw + 1, size=(batch_size,))
    j = torch.randint(0, h - th + 1, size=(batch_size,))
    return torch.stack([i, j, i + tw, j + th], dim=1)
