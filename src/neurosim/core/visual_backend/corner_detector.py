"""
Computes keypoints and descriptors from rendered color images using
classical computer-vision feature detectors and descriptor extractors.

Supported detectors:
    - ORB: Oriented FAST and Rotated BRIEF (fast, binary descriptor)
    - SIFT: Scale-Invariant Feature Transform (float descriptor, scale-invariant)
    - FAST: Features from Accelerated Segment Test (keypoints only, very fast)
    - BRISK: Binary Robust Invariant Scalable Keypoints (binary, multi-scale)
    - GFTT: Good Features to Track / Shi-Tomasi corners (keypoints only)

Supported descriptors (used when detector does not produce descriptors):
    - ORB: Oriented BRIEF binary descriptor (32 bytes)
    - BRIEF: Binary descriptor (32 bytes)  [via ORB with detected keypoints]
    - BRISK: Binary descriptor (64 bytes)
    - SIFT: 128-dim float descriptor

The sensor creates an internal color sensor in Habitat and extracts kps on each call.
"""

import cv2
import torch
import logging
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FeatureDetectionResult:
    """Container for feature detection results.

    Attributes:
        keypoints: (N, 2) float32 array of (x, y) pixel coordinates.
        scores: (N,) float32 array of keypoint response/score values.
        descriptors: (N, D) array of descriptors (float32 for SIFT, uint8 for
                     binary descriptors like ORB/BRISK).  ``None`` when no
                     descriptor extractor is configured.
        sizes: (N,) float32 array of keypoint sizes (diameters).
        angles: (N,) float32 array of keypoint orientations in degrees.
                -1 means orientation is not available.
        num_keypoints: Number of detected keypoints.
        detector_name: Name of the detector that produced these results.
        descriptor_name: Name of the descriptor extractor (may differ from
                         detector when using a separate extractor).
    """

    keypoints: np.ndarray  # (N, 2) float32
    scores: np.ndarray  # (N,) float32
    descriptors: np.ndarray | None  # (N, D) float32 or uint8
    sizes: np.ndarray  # (N,) float32
    angles: np.ndarray  # (N,) float32
    octaves: np.ndarray  # (N,) int32 – pyramid level (0 = base)
    num_keypoints: int
    detector_name: str
    descriptor_name: str | None


_DETECTOR_FACTORIES = {
    "orb": lambda cfg: cv2.ORB_create(
        nfeatures=cfg.get("max_features", 1000),
        scaleFactor=cfg.get("scale_factor", 1.2),
        nlevels=cfg.get("n_levels", 8),
        edgeThreshold=cfg.get("edge_threshold", 31),
        patchSize=cfg.get("patch_size", 31),
        fastThreshold=cfg.get("fast_threshold", 20),
    ),
    "sift": lambda cfg: cv2.SIFT_create(
        nfeatures=cfg.get("max_features", 1000),
        nOctaveLayers=cfg.get("n_octave_layers", 3),
        contrastThreshold=cfg.get("contrast_threshold", 0.04),
        edgeThreshold=cfg.get("edge_threshold", 10),
        sigma=cfg.get("sigma", 1.6),
    ),
    "fast": lambda cfg: cv2.FastFeatureDetector_create(
        threshold=cfg.get("fast_threshold", 20),
        nonmaxSuppression=cfg.get("nonmax_suppression", True),
    ),
    "brisk": lambda cfg: cv2.BRISK_create(
        thresh=cfg.get("brisk_threshold", 30),
        octaves=cfg.get("n_levels", 3),
        patternScale=cfg.get("pattern_scale", 1.0),
    ),
    "gftt": lambda cfg: cv2.GFTTDetector_create(
        maxCorners=cfg.get("max_features", 1000),
        qualityLevel=cfg.get("quality_level", 0.01),
        minDistance=cfg.get("min_distance", 10),
        blockSize=cfg.get("block_size", 3),
        useHarrisDetector=cfg.get("use_harris", False),
        k=cfg.get("harris_k", 0.04),
    ),
}

_DESCRIPTOR_FACTORIES = {
    "orb": lambda cfg: cv2.ORB_create(
        nfeatures=cfg.get("max_features", 1000),
    ),
    "sift": lambda cfg: cv2.SIFT_create(
        nfeatures=cfg.get("max_features", 1000),
    ),
    "brisk": lambda cfg: cv2.BRISK_create(),
}


def _create_detector(name: str, cfg: dict) -> cv2.Feature2D:
    """Instantiate an OpenCV feature detector by name."""
    name_lower = name.lower()
    if name_lower not in _DETECTOR_FACTORIES:
        raise ValueError(
            f"Unknown detector '{name}'. Supported: {list(_DETECTOR_FACTORIES.keys())}"
        )
    return _DETECTOR_FACTORIES[name_lower](cfg)


def _create_descriptor_extractor(name: str, cfg: dict) -> cv2.Feature2D | None:
    """Instantiate an OpenCV descriptor extractor by name.

    Returns ``None`` if *name* is ``None`` or ``"none"``.
    """
    if name is None or str(name).lower() == "none":
        return None
    name_lower = name.lower()
    if name_lower not in _DESCRIPTOR_FACTORIES:
        raise ValueError(
            f"Unknown descriptor '{name}'. "
            f"Supported: {list(_DESCRIPTOR_FACTORIES.keys())}"
        )
    return _DESCRIPTOR_FACTORIES[name_lower](cfg)


class CornerDetector:
    """Detects keypoints and computes descriptors from color images.

    This class wraps OpenCV feature detectors and descriptor extractors
    behind a simple, GPU-pipeline-friendly interface.  The caller passes
    an RGB torch.Tensor (which may live on GPU); the class handles the
    transfer to CPU / conversion to grayscale / detection / packing.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        detector: Name of the detector algorithm (orb, sift, fast, brisk, gftt).
        descriptor: Name of the descriptor algorithm.  ``None`` means the
                    detector's own descriptor is used (ORB, SIFT, BRISK) or
                    no descriptors are returned (FAST, GFTT).
        device: CUDA device string (used for future GPU-accelerated detectors).
        max_features: Maximum number of features to retain.
        detector_params: Extra parameters forwarded to the detector factory.
        descriptor_params: Extra parameters forwarded to the descriptor factory.
    """

    def __init__(
        self,
        width: int,
        height: int,
        detector: str = "orb",
        descriptor: str | None = None,
        device: str = "cuda:0",
        max_features: int = 1000,
        **extra_params,
    ):
        self.width = width
        self.height = height
        self.device = device
        self.detector_name = detector.lower()
        self.max_features = max_features

        # Merge extra_params into a config dict used by factories
        cfg = {"max_features": max_features, **extra_params}

        # --- Build the OpenCV detector ---------------------------------
        self._detector = _create_detector(self.detector_name, cfg)

        # --- Build the descriptor extractor ----------------------------
        # Some detectors already compute descriptors (ORB, SIFT, BRISK).
        # For detectors that only find keypoints (FAST, GFTT) we allow
        # the user to specify a separate descriptor extractor.
        _has_builtin_descriptor = self.detector_name in ("orb", "sift", "brisk")

        if descriptor is not None:
            # Explicit descriptor requested
            self.descriptor_name = descriptor.lower()
            if self.descriptor_name == self.detector_name and _has_builtin_descriptor:
                # Same as built-in → just use the detector's own compute()
                self._descriptor_extractor = None
                self._use_builtin_descriptor = True
            else:
                self._descriptor_extractor = _create_descriptor_extractor(
                    self.descriptor_name, cfg
                )
                self._use_builtin_descriptor = False
        elif _has_builtin_descriptor:
            # No explicit descriptor, but detector has one built in
            self.descriptor_name = self.detector_name
            self._descriptor_extractor = None
            self._use_builtin_descriptor = True
        else:
            # Keypoint-only detector, no descriptor requested
            self.descriptor_name = None
            self._descriptor_extractor = None
            self._use_builtin_descriptor = False

        logger.info(
            f"CornerDetector initialized: {width}x{height}, "
            f"detector={self.detector_name}, descriptor={self.descriptor_name}, "
            f"max_features={max_features}, device={device}"
        )

    def detect(self, color_image: torch.Tensor) -> FeatureDetectionResult:
        """Detect keypoints and compute descriptors from a color image.

        Args:
            color_image: (H, W, 3) uint8 or float32 RGB tensor.  May live on
                         GPU - will be transferred to CPU automatically.

        Returns:
            A :class:`FeatureDetectionResult` with keypoints, scores,
            descriptors, sizes, and angles.
        """
        # --- Transfer to CPU numpy if needed ---------------------------
        if isinstance(color_image, torch.Tensor):
            img_np = color_image.detach().cpu().numpy()
        else:
            img_np = np.asarray(color_image)

        # Ensure uint8
        if img_np.dtype != np.uint8:
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            else:
                img_np = img_np.clip(0, 255).astype(np.uint8)

        # Convert to grayscale (OpenCV expects BGR, but for grayscale
        # the channel order doesn't matter for the conversion formula)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # --- Detect & describe -----------------------------------------
        if self._use_builtin_descriptor:
            kps, desc = self._detector.detectAndCompute(gray, None)
        else:
            kps = self._detector.detect(gray, None)
            if self._descriptor_extractor is not None and len(kps) > 0:
                kps, desc = self._descriptor_extractor.compute(gray, kps)
            else:
                desc = None

        # --- Sort by response and keep top-N ---------------------------
        if len(kps) > self.max_features:
            kps = sorted(kps, key=lambda k: k.response, reverse=True)[
                : self.max_features
            ]
            if desc is not None:
                indices = np.argsort([-k.response for k in kps])[: self.max_features]
                desc = desc[indices]

        # --- Pack into numpy arrays ------------------------------------
        n = len(kps)
        if n == 0:
            return FeatureDetectionResult(
                keypoints=np.empty((0, 2), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
                descriptors=np.empty((0, 0), dtype=np.uint8)
                if desc is None
                else np.empty(
                    (0, desc.shape[1] if desc is not None and desc.ndim == 2 else 0),
                    dtype=np.uint8,
                ),
                sizes=np.empty((0,), dtype=np.float32),
                angles=np.empty((0,), dtype=np.float32),
                octaves=np.empty((0,), dtype=np.int32),
                num_keypoints=0,
                detector_name=self.detector_name,
                descriptor_name=self.descriptor_name,
            )

        coords = np.array([kp.pt for kp in kps], dtype=np.float32)  # (N, 2) x, y
        scores = np.array([kp.response for kp in kps], dtype=np.float32)
        sizes = np.array([kp.size for kp in kps], dtype=np.float32)
        angles = np.array([kp.angle for kp in kps], dtype=np.float32)
        # cv2.KeyPoint.octave encodes both octave and layer; extract the
        # low byte (bits 0-7) which is the pyramid octave level.
        octaves = np.array([int(kp.octave) & 0xFF for kp in kps], dtype=np.int32)
        # Treat values >= 128 as negative levels (wrap-around encoding)
        octaves = np.where(octaves >= 128, octaves - 256, octaves)

        return FeatureDetectionResult(
            keypoints=coords,
            scores=scores,
            descriptors=desc,
            sizes=sizes,
            angles=angles,
            octaves=octaves,
            num_keypoints=n,
            detector_name=self.detector_name,
            descriptor_name=self.descriptor_name,
        )
