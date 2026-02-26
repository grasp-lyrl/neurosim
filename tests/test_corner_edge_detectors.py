"""
Pytest tests for CornerDetector and EdgeDetector.

Run with:
    conda run -n neurosim pytest tests/test_corner_edge_detectors.py -v
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neurosim.core.visual_backend.corner_detector import (
    CornerDetector,
    FeatureDetectionResult,
)
from neurosim.core.visual_backend.edge_detector import EdgeDetector
from neurosim.core.utils.utils_h5 import H5Logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _checkerboard(h: int = 120, w: int = 160, tile: int = 10) -> np.ndarray:
    """Return an (H, W, 3) uint8 checkerboard image with lots of corners."""
    img = np.zeros((h, w), dtype=np.uint8)
    for r in range(0, h, tile):
        for c in range(0, w, tile):
            if ((r // tile) + (c // tile)) % 2 == 0:
                img[r : r + tile, c : c + tile] = 200
    return np.stack([img, img, img], axis=-1)


def _torch_img(h: int = 480, w: int = 640) -> torch.Tensor:
    """Return the checkerboard as a torch GPU tensor (or CPU if no CUDA)."""
    arr = _checkerboard(h, w)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.from_numpy(arr).to(device)


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ===========================================================================
# CornerDetector tests
# ===========================================================================


class TestCornerDetectorInit:
    def test_default_orb(self):
        cd = CornerDetector(width=160, height=120)
        assert cd.detector_name == "orb"
        assert cd.max_features == 1000

    def test_custom_params_forwarded(self):
        cd = CornerDetector(
            width=160, height=120, detector="orb", max_features=50, fast_threshold=10
        )
        assert cd.max_features == 50

    @pytest.mark.parametrize("det", ["orb", "sift", "fast", "brisk", "gftt"])
    def test_all_detectors_init(self, det):
        cd = CornerDetector(width=160, height=120, detector=det)
        assert cd.detector_name == det


class TestCornerDetectorDetect:
    @pytest.fixture
    def detector(self):
        return CornerDetector(width=640, height=480, detector="orb", max_features=200)

    def test_returns_feature_detection_result(self, detector):
        img = _torch_img()
        result = detector.detect(img)
        assert isinstance(result, FeatureDetectionResult)

    def test_result_fields_non_empty(self, detector):
        img = _torch_img()
        result = detector.detect(img)
        # A checkerboard should reliably produce keypoints with ORB
        assert result.num_keypoints > 0, (
            f"Expected >0 keypoints on checkerboard; got {result.num_keypoints}. "
            "Check that the test image has sufficient texture."
        )
        assert result.keypoints.shape == (result.num_keypoints, 2)
        assert result.scores.shape == (result.num_keypoints,)
        assert result.sizes.shape == (result.num_keypoints,)
        assert result.angles.shape == (result.num_keypoints,)
        assert result.octaves.shape == (result.num_keypoints,)

    def test_octaves_field_dtype(self, detector):
        img = _torch_img()
        result = detector.detect(img)
        assert result.octaves.dtype == np.int32

    def test_max_features_respected(self):
        cd = CornerDetector(width=160, height=120, detector="orb", max_features=5)
        result = cd.detect(_torch_img())
        assert result.num_keypoints <= 5

    def test_empty_image_returns_zero_keypoints(self, detector):
        blank = torch.zeros((480, 640, 3), dtype=torch.uint8)
        result = detector.detect(blank)
        assert result.num_keypoints == 0
        assert result.keypoints.shape[1] == 2
        assert result.octaves.shape == (0,)

    def test_numpy_input_accepted(self, detector):
        arr = _checkerboard()
        result = detector.detect(arr)
        assert isinstance(result, FeatureDetectionResult)

    def test_float32_input_accepted(self, detector):
        arr = _checkerboard().astype(np.float32) / 255.0
        result = detector.detect(arr)
        assert isinstance(result, FeatureDetectionResult)

    def test_detector_name_in_result(self, detector):
        result = detector.detect(_torch_img())
        assert result.detector_name == "orb"


# ===========================================================================
# EdgeDetector tests
# ===========================================================================


class TestEdgeDetectorInit:
    def test_default_canny(self):
        ed = EdgeDetector(width=160, height=120, device=DEVICE)
        assert ed.algorithm == "canny"

    def test_custom_thresholds(self):
        ed = EdgeDetector(
            width=160, height=120, low_threshold=0.05, high_threshold=0.3, device=DEVICE
        )
        assert ed.low_threshold == pytest.approx(0.05)
        assert ed.high_threshold == pytest.approx(0.3)


class TestEdgeDetectorDetect:
    @pytest.fixture
    def canny_detector(self):
        return EdgeDetector(width=160, height=120, algorithm="canny", device=DEVICE)

    def _gpu_img(self):
        arr = _checkerboard()
        return torch.from_numpy(arr).to(DEVICE)

    def test_canny_output_shape(self, canny_detector):
        out = canny_detector.detect(self._gpu_img())
        assert out.shape == (120, 160)

    def test_canny_output_binary(self, canny_detector):
        out = canny_detector.detect(self._gpu_img())
        vals = out.unique()
        assert set(vals.cpu().numpy().tolist()).issubset({0.0, 1.0})

    def test_canny_detects_edges(self, canny_detector):
        out = canny_detector.detect(self._gpu_img())
        assert out.sum() > 0

    def test_sobel_output_shape(self):
        ed = EdgeDetector(width=160, height=120, algorithm="sobel", device=DEVICE)
        out = ed.detect(self._gpu_img())
        assert out.shape == (120, 160)
        assert out.min() >= 0.0 and out.max() <= 1.0 + 1e-5

    def test_laplacian_output_shape(self):
        ed = EdgeDetector(width=160, height=120, algorithm="laplacian", device=DEVICE)
        out = ed.detect(self._gpu_img())
        assert out.shape == (120, 160)

    def test_blank_image_few_edges(self, canny_detector):
        blank = torch.zeros((120, 160, 3), dtype=torch.uint8).to(DEVICE)
        out = canny_detector.detect(blank)
        assert out.sum() == 0

    def test_unknown_algorithm_raises(self):
        with pytest.raises(ValueError, match="Unknown edge algorithm"):
            ed = EdgeDetector(width=160, height=120, algorithm="bogus", device=DEVICE)
            ed.detect(self._gpu_img())

    def test_cpu_input_moved_to_device(self, canny_detector):
        cpu_img = torch.from_numpy(_checkerboard())  # CPU
        out = canny_detector.detect(cpu_img)
        assert out.shape == (120, 160)

    def test_uint8_and_float_give_same_result(self, canny_detector):
        arr = _checkerboard()
        img_uint8 = torch.from_numpy(arr).to(DEVICE)
        img_float = img_uint8.float() / 255.0
        out_u = canny_detector.detect(img_uint8)
        out_f = canny_detector.detect(img_float)
        assert torch.allclose(out_u, out_f)

    def _gpu_img(self):
        arr = _checkerboard()
        return torch.from_numpy(arr).to(DEVICE)


# ===========================================================================
# H5Logger corner logging tests (static-method level, no subprocess)
# ===========================================================================


def _make_fdr(n: int, desc_dim: int | None = 32) -> dict:
    """Build a FeatureDetectionResult-like dict (post _torch_to_numpy conversion)."""
    if n == 0:
        desc = np.empty((0, desc_dim), dtype=np.uint8) if desc_dim else None
        return {
            "keypoints": np.empty((0, 2), dtype=np.float32),
            "scores": np.empty((0,), dtype=np.float32),
            "sizes": np.empty((0,), dtype=np.float32),
            "angles": np.empty((0,), dtype=np.float32),
            "octaves": np.empty((0,), dtype=np.int32),
            "descriptors": desc,
            "num_keypoints": np.array([0], dtype=np.int32),
        }
    rng = np.random.default_rng(n)
    desc = rng.integers(0, 255, (n, desc_dim), dtype=np.uint8) if desc_dim else None
    return {
        "keypoints": rng.random((n, 2), dtype=np.float32) * 640,
        "scores": rng.random(n, dtype=np.float32) * 100,
        "sizes": rng.random(n, dtype=np.float32) * 20 + 1,
        "angles": rng.random(n, dtype=np.float32) * 360,
        "octaves": rng.integers(-1, 4, n, dtype=np.int32),
        "descriptors": desc,
        "num_keypoints": np.array([n], dtype=np.int32),
    }


class TestH5TorchToNumpy:
    def test_converts_feature_detection_result(self):
        cd = CornerDetector(width=640, height=480, detector="orb", max_features=10)
        fdr = cd.detect(_torch_img())
        converted = H5Logger._torch_to_numpy(fdr)
        assert isinstance(converted, dict)
        assert "keypoints" in converted
        assert "num_keypoints" in converted
        assert isinstance(converted["keypoints"], np.ndarray)

    def test_num_keypoints_is_1d_array(self):
        cd = CornerDetector(width=640, height=480, detector="orb", max_features=10)
        fdr = cd.detect(_torch_img())
        converted = H5Logger._torch_to_numpy(fdr)
        nk = converted["num_keypoints"]
        assert isinstance(nk, np.ndarray)
        assert nk.shape == (1,)
        assert nk[0] == fdr.num_keypoints


class TestH5WriteCorners:
    """Test _write_corners static method directly (no subprocess overhead)."""

    def _write_and_read(self, data_list, times, steps):
        import h5py

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tf:
            fname = tf.name
        try:
            with h5py.File(fname, "w") as f:
                grp = f.create_group("corner_sensor")
                H5Logger._write_corners(
                    grp,
                    data_list,
                    times,
                    steps,
                    meta_chunk_size=100,
                    feature_chunk_size=1000,
                    compression=None,
                )
            with h5py.File(fname, "r") as f:
                return {k: f["corner_sensor"][k][:] for k in f["corner_sensor"]}
        finally:
            Path(fname).unlink(missing_ok=True)

    def test_basic_write_creates_datasets(self):
        data = [_make_fdr(5), _make_fdr(3)]
        times = np.array([0.1, 0.2])
        steps = np.array([10, 20])
        result = self._write_and_read(data, times, steps)
        assert "keypoints" in result
        assert "scores" in result
        assert "num_keypoints" in result
        assert "sim_time" in result
        assert "sim_step" in result

    def test_keypoints_shape_is_total_n(self):
        n1, n2 = 4, 7
        data = [_make_fdr(n1), _make_fdr(n2)]
        times, steps = np.array([0.0, 0.1]), np.array([0, 1])
        result = self._write_and_read(data, times, steps)
        assert result["keypoints"].shape == (n1 + n2, 2)
        assert result["scores"].shape == (n1 + n2,)

    def test_num_keypoints_per_frame(self):
        counts = [3, 0, 6]
        data = [_make_fdr(n) for n in counts]
        times = np.arange(len(counts), dtype=float)
        steps = np.arange(len(counts), dtype=int)
        result = self._write_and_read(data, times, steps)
        np.testing.assert_array_equal(result["num_keypoints"], counts)

    def test_descriptors_written_when_present(self):
        data = [_make_fdr(5, desc_dim=32), _make_fdr(3, desc_dim=32)]
        times, steps = np.array([0.0, 0.1]), np.array([0, 1])
        result = self._write_and_read(data, times, steps)
        assert "descriptors" in result
        assert result["descriptors"].shape == (8, 32)

    def test_no_descriptors_when_absent(self):
        data = [_make_fdr(5, desc_dim=None), _make_fdr(2, desc_dim=None)]
        times, steps = np.array([0.0, 0.1]), np.array([0, 1])
        result = self._write_and_read(data, times, steps)
        assert "descriptors" not in result

    def test_all_zero_frames(self):
        data = [_make_fdr(0), _make_fdr(0)]
        times, steps = np.array([0.0, 0.1]), np.array([0, 1])
        result = self._write_and_read(data, times, steps)
        # No feature datasets should be written, only metadata
        assert "keypoints" not in result
        assert "num_keypoints" in result
        np.testing.assert_array_equal(result["num_keypoints"], [0, 0])

    def test_octaves_dtype(self):
        data = [_make_fdr(4)]
        times, steps = np.array([0.0]), np.array([0])
        result = self._write_and_read(data, times, steps)
        assert result["octaves"].dtype == np.int32

    def test_slice_recovery(self):
        """Verify that per-frame keypoints can be recovered using num_keypoints."""
        frames = [_make_fdr(n) for n in [3, 5, 2]]
        times, steps = np.arange(3, dtype=float), np.arange(3)
        result = self._write_and_read(frames, times, steps)

        offsets = np.concatenate([[0], np.cumsum(result["num_keypoints"])])
        for i, frame in enumerate(frames):
            start, end = int(offsets[i]), int(offsets[i + 1])
            recovered_kps = result["keypoints"][start:end]
            np.testing.assert_array_almost_equal(
                recovered_kps, frame["keypoints"], decimal=5
            )
