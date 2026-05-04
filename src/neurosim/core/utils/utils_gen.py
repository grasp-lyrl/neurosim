import copy
import yaml
from typing import Any
from pathlib import Path

import torch
import numpy as np


MAP_BORDER_INDICATOR = 2
RECOLOR_MAP = np.array(
    [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
)  # free, occupied, unknown


def outline_border(top_down_map):
    left_right_block_nav = (top_down_map[:, :-1] == 1) & (
        top_down_map[:, :-1] != top_down_map[:, 1:]
    )
    left_right_nav_block = (top_down_map[:, 1:] == 1) & (
        top_down_map[:, :-1] != top_down_map[:, 1:]
    )

    up_down_block_nav = (top_down_map[:-1] == 1) & (
        top_down_map[:-1] != top_down_map[1:]
    )
    up_down_nav_block = (top_down_map[1:] == 1) & (
        top_down_map[:-1] != top_down_map[1:]
    )

    top_down_map[:, :-1][left_right_block_nav] = MAP_BORDER_INDICATOR
    top_down_map[:, 1:][left_right_nav_block] = MAP_BORDER_INDICATOR

    top_down_map[:-1][up_down_block_nav] = MAP_BORDER_INDICATOR
    top_down_map[1:][up_down_nav_block] = MAP_BORDER_INDICATOR


@torch.compile(mode="reduce-overhead")
def color2intensity(color_im: torch.Tensor) -> torch.Tensor:
    """Convert a color image to an intensity image.

    Args:
        color_im (torch.Tensor): Input color image of shape (H, W, 3) with RGB channels.

    Returns:
        torch.Tensor: Intensity image of shape (H, W).
    """
    intensity_im = (
        0.2989 * color_im[:, :, 0]
        + 0.5870 * color_im[:, :, 1]
        + 0.1140 * color_im[:, :, 2]
    )
    return intensity_im


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and require the top-level node to be a mapping."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    """Recursively update ``dst`` with values from ``src``.

    Nested dicts are merged in place; non-dict values from ``src`` are
    deep-copied so callers can safely mutate either side afterward.
    """
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)
    return dst
