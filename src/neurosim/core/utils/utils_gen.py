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
