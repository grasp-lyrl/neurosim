"""
Computes edge maps from rendered color images using Kornia's Canny edge
detector, which runs entirely on GPU via PyTorch convolutions.

Supported algorithms:
    - canny:   Full Canny pipeline (Gaussian blur → Sobel gradients →
               non-maximum suppression → hysteresis thresholding).
               Returns a clean, thin, binary edge map.  **Default.**
    - sobel:   Sobel gradient magnitude.  Returns continuous-valued edge
               strengths (thicker edges, no NMS).  Faster than Canny.
    - laplacian: Laplacian of Gaussian.  Returns continuous-valued second
                 derivative magnitudes.  Sensitive to noise.

The sensor creates an internal color sensor in Habitat and computes edges on each call.
"""

import torch
import logging

from neurosim.core.utils.utils_gen import color2intensity

logger = logging.getLogger(__name__)


class EdgeDetector:
    """Computes edge maps from color images on GPU.

    Like the :class:`OpticalFlowComputer`, this class is instantiated once
    per sensor and called every time a new frame is available.  All heavy
    computation stays on GPU.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        device: CUDA device string.
        algorithm: Edge detection algorithm (``"canny"``, ``"sobel"``,
                   ``"laplacian"``).
        low_threshold: Canny low hysteresis threshold (ignored for sobel/laplacian).
        high_threshold: Canny high hysteresis threshold.
        kernel_size: Gaussian / Sobel kernel size (int or tuple).
        sigma: Gaussian sigma for Canny pre-smoothing (float or tuple).
        return_magnitude: If ``True`` (and algorithm is ``"canny"``), also
                          return the gradient magnitude alongside the binary
                          edge map.  Default ``False``.
    """

    def __init__(
        self,
        width: int,
        height: int,
        device: str = "cuda:0",
        algorithm: str = "canny",
        low_threshold: float = 0.1,
        high_threshold: float = 0.2,
        kernel_size: int | tuple[int, int] = 5,
        sigma: float | tuple[float, float] = 1.0,
        return_magnitude: bool = False,
    ):
        self.width = width
        self.height = height
        self.device = device
        self.algorithm = algorithm.lower()
        self.return_magnitude = return_magnitude

        # Store Canny-specific parameters
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        self.sigma = (sigma, sigma) if isinstance(sigma, (int, float)) else sigma

        # Lazy-import kornia so the module can be loaded even when kornia
        # is absent (e.g. in unit-test mocking scenarios).
        import kornia.filters  # noqa: F401

        self._kornia_filters = kornia.filters

        logger.info(
            f"EdgeDetector initialized: {width}x{height}, "
            f"algorithm={self.algorithm}, device={device}, "
            f"thresholds=({low_threshold}, {high_threshold})"
        )

    @torch.no_grad()
    def detect(self, color_image: torch.Tensor) -> torch.Tensor:
        """Compute an edge map from a color image.

        Args:
            color_image: (H, W, 3) uint8 or float32 RGB tensor on GPU.

        Returns:
            (H, W) float32 edge map on GPU.
              - For ``"canny"``: binary {0, 1} edge map (thin edges).
              - For ``"sobel"``: continuous gradient magnitude (0-1 range).
              - For ``"laplacian"``: continuous second-derivative magnitude.
        """
        img = self._prepare_input(color_image)  # (1, 1, H, W) float32 on GPU

        if self.algorithm == "canny":
            magnitude, edges = self._kornia_filters.canny(
                img,
                low_threshold=self.low_threshold,
                high_threshold=self.high_threshold,
                kernel_size=self.kernel_size,
                sigma=self.sigma,
                hysteresis=True,
            )
            result = edges.squeeze(0).squeeze(0)  # (H, W)
            if self.return_magnitude:
                # Return magnitude multiplied by edge mask for clean visualisation
                mag = magnitude.squeeze(0).squeeze(0)
                return torch.stack([result, mag], dim=-1)  # (H, W, 2)
            return result

        elif self.algorithm == "sobel":
            # Sobel gradient magnitude (normalised to 0-1)
            magnitude = self._kornia_filters.sobel(img)  # (1, 1, H, W)
            mag = magnitude.squeeze(0).squeeze(0)
            # Normalise to [0, 1]
            mag_max = mag.max()
            if mag_max > 0:
                mag = mag / mag_max
            return mag

        elif self.algorithm == "laplacian":
            lap = self._kornia_filters.laplacian(img, kernel_size=self.kernel_size[0])
            lap = lap.squeeze(0).squeeze(0).abs()
            lap_max = lap.max()
            if lap_max > 0:
                lap = lap / lap_max
            return lap

        else:
            raise ValueError(
                f"Unknown edge algorithm '{self.algorithm}'. "
                "Supported: canny, sobel, laplacian"
            )

    def _prepare_input(self, color_image: torch.Tensor) -> torch.Tensor:
        """Convert (H, W, 3) RGB image to (1, 1, H, W) grayscale float tensor.

        Handles both uint8 [0, 255] and float [0, 1] inputs.
        Stays on the same device as the input tensor.
        """
        img = color_image
        if img.device.type == "cpu":
            img = img.to(self.device)

        # Normalise uint8 -> float
        if img.dtype == torch.uint8:
            img = img.float() / 255.0

        # (H, W, 3) -> grayscale using shared color2intensity utility (Rec. 601)
        return color2intensity(img)[None, None]  # (1, 1, H, W)
