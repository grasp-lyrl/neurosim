"""Unit tests for the radial-tangential camera lens."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from neurosim.core.visual_backend.lens import Radtan, create_lens

PINHOLE = {"width": 640, "height": 480, "hfov": 49.07}
CAMERA = {**PINHOLE, "principal_point": [293.18, 231.42]}
REFIT = [-0.5721, 0.2233, -0.0025, 0.0066, 0.0]
CALIB_40DEG = [-0.6045, 0.4435, -0.0025, 0.0066, -0.3308]
SETTINGS = Path("configs/apartment_1-settings.yaml")
SCENE = Path("data/scene_datasets/habitat-test-scenes/apartment_1.glb")


def test_camera_without_distortion_reads_habitat_as_is():
    lens = create_lens(PINHOLE, "cpu", "rgba")
    obs = torch.randint(0, 256, (480, 640, 4), dtype=torch.uint8)
    assert (lens.resolution, lens.hfov) == ((480, 640), PINHOLE["hfov"])
    assert lens(obs) is obs, "a pinhole lens must hand Habitat's image through"


def test_gathered_pixel_reprojects_onto_its_sensor_pixel():
    lens = Radtan({**CAMERA, "distortion": REFIT}, "cpu", "nearest")
    (hr, wr), f = lens.resolution, 320 / np.tan(np.radians(CAMERA["hfov"]) / 2)
    row, col = np.divmod(lens.index.numpy(), wr)
    x, y = (col + 0.5 - wr / 2) / f, (row + 0.5 - hr / 2) / f
    rays = np.stack([x, y, np.ones(x.shape)], -1).reshape(-1, 3)
    k = np.array([[f, 0, 293.18], [0, f, 231.42], [0, 0, 1]])
    uv = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), k, np.array(REFIT))[0]
    u, v = np.meshgrid(np.arange(640), np.arange(480))
    err = np.linalg.norm(uv.reshape(480, 640, 2) - np.stack([u, v], -1), axis=-1)
    assert err.max() < 0.75, f"a gathered pixel lands {err.max():.2f} px off"


def test_folding_distortion_is_rejected():
    with pytest.raises(ValueError):
        Radtan({**CAMERA, "distortion": CALIB_40DEG}, "cpu", "nearest")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_resample_kernel_matches_grid_sample():
    cfg = {**CAMERA, "distortion": REFIT}
    lens = Radtan(cfg, "cuda", "rgba")
    shape = (*lens.resolution, 4)
    rgba = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
    h, w, taps = lens.pos.shape[:3]
    half = torch.tensor(lens.resolution[::-1], device="cuda") / 2
    grid = ((lens.pos + 0.5) / half - 1).reshape(1, h, w * taps, 2)
    image = rgba.float().movedim(-1, 0)[None]
    ref = F.grid_sample(image, grid, padding_mode="border", align_corners=False)
    ref = ref[0].unflatten(-1, (w, taps)).mean(-1).movedim(0, -1)
    err = (lens(rgba).float() - ref).abs().max().item()
    assert err <= 0.51, f"RGBA is {err:.2f} off grid_sample, beyond rounding"
    rec601 = torch.tensor([0.2989, 0.5870, 0.1140], device="cuda")
    luma = Radtan(cfg, "cuda", "luma")(rgba)
    torch.testing.assert_close(luma, ref[..., :3] @ rec601 / 255, rtol=0, atol=1e-4)


@pytest.mark.skipif(not SCENE.exists(), reason="apartment_1 scene not available")
def test_zero_distortion_lens_sees_the_native_depth_camera():
    pytest.importorskip("habitat_sim")
    from neurosim.core.visual_backend.habitat_wrapper import HabitatWrapper

    def depth_camera(**lens):
        pose = {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0]}
        return {"type": "depth", **PINHOLE, "zfar": 100.0, **pose, **lens}

    settings = yaml.safe_load(SETTINGS.read_text())["visual_backend"]
    settings["sensors"] = {
        "native": depth_camera(),
        "lens": depth_camera(distortion=[0.0] * 5, render_scale=3),
    }
    backend = HabitatWrapper(settings)
    point = backend._sim.pathfinder.get_random_navigable_point()
    backend.update_agent_state(point, np.quaternion(1, 0, 0, 0))
    native, lensed = backend.render_depth("native"), backend.render_depth("lens")
    backend.close()
    assert native.max() > 0, "the camera sees nothing, so the comparison is empty"
    torch.testing.assert_close(lensed, native, rtol=1e-4, atol=1e-6)
