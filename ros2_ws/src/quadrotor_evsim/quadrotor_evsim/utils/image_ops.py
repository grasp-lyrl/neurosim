import torch

@torch.compile(mode="reduce-overhead")
def color2intensity(color_im: torch.Tensor) -> torch.Tensor:
    """Convert a color image to an intensity image.

    Args:
        color_im (torch.Tensor): Input color image of shape (H, W, 3) with RGB channels.
        
    Returns:
        torch.Tensor: Intensity image of shape (H, W).
    """
    intensity_im = 0.2989 * color_im[:, :, 0] + 0.5870 * color_im[:, :, 1] + 0.1140 * color_im[:, :, 2]
    return intensity_im
