"""Camera lenses on Habitat's pinhole renders: the pinhole itself, or radial-tangential."""

import cv2
import numpy as np
import torch
import triton
import triton.language as tl

from neurosim.core.utils import color2intensity

TAPS = 3


class Pinhole:
    """Habitat's own camera: the render is the sensor image."""

    def __init__(self, cfg: dict, readout: str):
        self.resolution, self.hfov = (cfg["height"], cfg["width"]), cfg["hfov"]
        self.readout = readout

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """What the sensor reads out: luma, or Habitat's observation as it is."""
        return color2intensity(obs[..., :3] / 255.0) if self.readout == "luma" else obs


class Radtan:
    """OpenCV radial-tangential lens: each sensor pixel reads a wider pinhole render."""

    def __init__(self, cfg: dict, device: str, readout: str):
        w, h = cfg["width"], cfg["height"]
        f = w / 2 / np.tan(np.radians(cfg["hfov"]) / 2)
        cx, cy = cfg.get("principal_point", ((w - 1) / 2, (h - 1) / 2))
        k = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        dist = np.asarray(cfg["distortion"], float) * cfg.get("distortion_scale", 1.0)

        taps = 1 if readout == "nearest" else TAPS
        o = (np.arange(taps) + 0.5) / taps - 0.5
        u, v = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
        px = np.stack([np.stack([u + dx, v + dy], -1) for dy in o for dx in o], 2)
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)
        rays = cv2.undistortPointsIter(
            px.reshape(-1, 1, 2), k, dist, None, None, criteria
        )
        rays = rays.reshape(h, w, taps * taps, 2)

        # undistortPoints returns garbage, silently, where the model folds over
        centre = rays[:, :, taps * taps // 2]
        rays3d = np.dstack([centre, np.ones((h, w))]).reshape(-1, 3)
        back = cv2.projectPoints(rays3d, np.zeros(3), np.zeros(3), k, dist)[0]
        if np.abs(back.reshape(h, w, 2) - np.stack([u, v], -1)).max() > 1e-3:
            raise ValueError(
                f"distortion {dist.tolist()} is not invertible on the sensor"
            )

        fr = f * cfg.get("render_scale", 1.0)
        half = np.ceil(np.abs(rays).max((0, 1, 2)) * fr + 1)
        self.resolution = (2 * int(half[1]), 2 * int(half[0]))
        self.hfov = float(np.degrees(2 * np.arctan(half[0] / fr)))
        pos = rays * fr + half - 0.5
        self.pos = torch.from_numpy(pos).float().to(device)
        nearest = np.rint(pos[:, :, taps * taps // 2]).astype(np.int64)
        index = nearest[..., 1] * self.resolution[1] + nearest[..., 0]
        self.index = torch.from_numpy(index).to(device)
        self.readout = readout

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """What the sensor reads out: luma, RGBA, or the nearest render value."""
        if self.readout == "nearest":
            return obs.reshape(-1, *obs.shape[2:])[self.index]
        luma = self.readout == "luma"
        h, w, taps = self.pos.shape[:3]
        out = torch.empty(
            h, w, dtype=torch.float32 if luma else torch.int32, device=obs.device
        )
        with torch.cuda.device(obs.device):
            _resample[(triton.cdiv(h * w, 256),)](
                obs.view(torch.int32),
                self.pos,
                out,
                h * w,
                obs.shape[1],
                obs.shape[0],
                TAPS=taps,
                LUMA=luma,
                BLOCK=256,
            )
        return out if luma else out.view(torch.uint8).view(h, w, 4)


def create_lens(cfg: dict, device: str, readout: str) -> Pinhole | Radtan:
    """The camera's radial-tangential lens if its config gives a distortion, else its pinhole."""
    return (
        Radtan(cfg, device, readout) if "distortion" in cfg else Pinhole(cfg, readout)
    )


@triton.jit
def _resample(
    rgba,
    pos,
    out,
    n,
    wr,
    hr,
    TAPS: tl.constexpr,
    LUMA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    r = tl.zeros([BLOCK], dtype=tl.float32)
    g = tl.zeros([BLOCK], dtype=tl.float32)
    b = tl.zeros([BLOCK], dtype=tl.float32)
    a = tl.zeros([BLOCK], dtype=tl.float32)
    for t in tl.static_range(TAPS):
        x = tl.load(pos + (i * TAPS + t) * 2, mask=mask, other=0.0)
        y = tl.load(pos + (i * TAPS + t) * 2 + 1, mask=mask, other=0.0)
        x = tl.minimum(tl.maximum(x, 0.0), wr - 1.0)
        y = tl.minimum(tl.maximum(y, 0.0), hr - 1.0)
        x0, y0 = x.to(tl.int32), y.to(tl.int32)
        fx, fy = x - x0, y - y0
        for c in tl.static_range(4):
            xc = tl.minimum(x0 + c % 2, wr - 1)
            yc = tl.minimum(y0 + c // 2, hr - 1)
            wc = (fx if c % 2 else 1 - fx) * (fy if c // 2 else 1 - fy)
            p = tl.load(rgba + yc * wr + xc, mask=mask, other=0)
            r += wc * (p & 0xFF).to(tl.float32)
            g += wc * ((p >> 8) & 0xFF).to(tl.float32)
            b += wc * ((p >> 16) & 0xFF).to(tl.float32)
            a += wc * ((p >> 24) & 0xFF).to(tl.float32)
    if LUMA:
        luma = (0.2989 * r + 0.5870 * g + 0.1140 * b) / (255.0 * TAPS)
        tl.store(out + i, luma, mask=mask)
    else:
        r = (r / TAPS + 0.5).to(tl.int32)
        g = (g / TAPS + 0.5).to(tl.int32)
        b = (b / TAPS + 0.5).to(tl.int32)
        a = (a / TAPS + 0.5).to(tl.int32)
        tl.store(out + i, r | (g << 8) | (b << 16) | (a << 24), mask=mask)
