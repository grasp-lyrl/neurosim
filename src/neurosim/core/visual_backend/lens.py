"""Camera lenses on Habitat's pinhole renders: the pinhole itself, or radial-tangential."""

import cv2
import numpy as np
import torch
import triton
import triton.language as tl

from neurosim.core.coord_trans.calibration import camera_matrix, distortion_coefficients

TAPS = 3


class Pinhole:
    """Habitat's own camera: the render is the sensor image."""

    def __init__(self, cfg: dict, device: str, readout: str):
        self.resolution, self.hfov = (cfg["height"], cfg["width"]), cfg["hfov"]
        self.read = self.as_is
        if readout == "luma":
            self.levels, self.read = linear_levels(device), self.luma

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return self.read(obs)

    def luma(self, obs: torch.Tensor) -> torch.Tensor:
        """Luminance in linear light."""
        h, w = obs.shape[:2]
        out = torch.empty(h, w, dtype=torch.float32, device=obs.device)
        with torch.cuda.device(obs.device):
            _luma[(triton.cdiv(h * w, 256),)](
                obs.view(torch.int32), self.levels, out, h * w, BLOCK=256
            )
        return out

    @staticmethod
    def as_is(obs: torch.Tensor) -> torch.Tensor:
        return obs


class Radtan:
    """OpenCV radial-tangential lens: each sensor pixel reads a wider pinhole render."""

    def __init__(self, cfg: dict, device: str, readout: str):
        w, h = cfg["width"], cfg["height"]
        k, dist = camera_matrix(cfg), distortion_coefficients(cfg)
        f = k[0, 0]

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
        self.read = {"nearest": self.nearest, "luma": self.luma, "rgba": self.rgba}[
            readout
        ]
        if readout == "luma":
            self.levels = linear_levels(device)

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """What the sensor reads out: luma, RGBA, or the nearest render value."""
        return self.read(obs)

    def nearest(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.reshape(-1, *obs.shape[2:])[self.index]

    def luma(self, obs: torch.Tensor) -> torch.Tensor:
        return self.resample(obs, self.levels, torch.float32, luma=True)

    def rgba(self, obs: torch.Tensor) -> torch.Tensor:
        h, w = self.pos.shape[:2]
        # Without LUMA the kernel reads no levels, so the render stands in for them.
        out = self.resample(obs, obs, torch.int32, luma=False)
        return out.view(torch.uint8).view(h, w, 4)

    def resample(
        self, obs: torch.Tensor, levels: torch.Tensor, dtype: torch.dtype, luma: bool
    ) -> torch.Tensor:
        h, w, taps = self.pos.shape[:3]
        out = torch.empty(h, w, dtype=dtype, device=obs.device)
        with torch.cuda.device(obs.device):
            _resample[(triton.cdiv(h * w, 256),)](
                obs.view(torch.int32),
                levels,
                self.pos,
                out,
                h * w,
                obs.shape[1],
                obs.shape[0],
                TAPS=taps,
                LUMA=luma,
                BLOCK=256,
            )
        return out


def create_lens(cfg: dict, device: str, readout: str) -> Pinhole | Radtan:
    """The camera's radial-tangential lens if its config gives a distortion, else its pinhole."""
    if "principal_point" in cfg and "distortion" not in cfg:
        raise ValueError(
            "a principal_point needs a distortion: Habitat's pinhole is centred"
        )
    return (
        Radtan(cfg, device, readout)
        if "distortion" in cfg
        else Pinhole(cfg, device, readout)
    )


def linear_levels(device: str) -> torch.Tensor:
    """The 256 8-bit sRGB levels in linear light, from the kernels' own _linear."""
    levels = torch.empty(256, dtype=torch.float32, device=device)
    with torch.cuda.device(levels.device):
        _linear_levels[(1,)](levels, BLOCK=256)
    return levels


@triton.jit
def _channel(p, shift: tl.constexpr):
    """One 8-bit channel of packed RGBA pixels."""
    return ((p >> shift) & 0xFF).to(tl.float32)


@triton.jit
def _linear(channel):
    """An 8-bit sRGB channel value in linear light, in [0, 1]."""
    v = channel / 255.0
    return tl.where(v <= 0.04045, v / 12.92, tl.exp(2.4 * tl.log((v + 0.055) / 1.055)))


@triton.jit
def _linear_levels(out, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(out + i, _linear(i.to(tl.float32)))


@triton.jit
def _level(levels, p, shift: tl.constexpr):
    """One channel of packed RGBA pixels in linear light."""
    return tl.load(levels + ((p >> shift) & 0xFF))


@triton.jit
def _luminance(r, g, b):
    """Rec. 709 luminance of linear-light RGB."""
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


@triton.jit
def _luma(rgba, levels, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    p = tl.load(rgba + i, mask=i < n)
    rgb = _level(levels, p, 0), _level(levels, p, 8), _level(levels, p, 16)
    tl.store(out + i, _luminance(*rgb), mask=i < n)


@triton.jit
def _resample(
    rgba,
    levels,
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
            if LUMA:
                # Blend in linear light, the light a sensor integrates.
                pr, pg, pb = (
                    _level(levels, p, 0),
                    _level(levels, p, 8),
                    _level(levels, p, 16),
                )
            else:
                pr, pg, pb = _channel(p, 0), _channel(p, 8), _channel(p, 16)
            r += wc * pr
            g += wc * pg
            b += wc * pb
            a += wc * _channel(p, 24)
    if LUMA:
        luma = _luminance(r, g, b) / TAPS
        tl.store(out + i, luma, mask=mask)
    else:
        r = (r / TAPS + 0.5).to(tl.int32)
        g = (g / TAPS + 0.5).to(tl.int32)
        b = (b / TAPS + 0.5).to(tl.int32)
        a = (a / TAPS + 0.5).to(tl.int32)
        tl.store(out + i, r | (g << 8) | (b << 16) | (a << 24), mask=mask)
