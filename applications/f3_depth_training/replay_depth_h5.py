"""Step a trained depth model over recorded events, one window at a time.

Press ``n``/Enter for the next window and ``q`` to quit, or pass ``--video`` to write
an mp4 instead (much faster than forwarding a window over ssh). Events are
centre-cropped to the model's frame, never resampled: rescaling event coordinates
makes some target pixels collect more source pixels than others, and the feature
field reads that fixed density pattern as structure.
"""

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from matplotlib import colormaps

from .nets import EventFFDepthAnythingV2, load_depth_weights
from .utils import get_depth_image, get_disparity_image

# dataset -> (event group, ms->index dataset, sensor width, height)
LAYOUTS = {
    "m3ed": ("prophesee/left", "prophesee/left/ms_map_idx", 1280, 720),
    "tumvie": ("events", "ms_to_idx", 1280, 720),
    # NeuroFly flight bags, via the neurofly repo's scripts/mcap_to_h5.py; the DVS
    # matches the model frame, so the centre crop below is the identity.
    "neurofly": ("event_camera_1", "event_camera_1/ms_to_idx", 640, 480),
}


def load_model(run: Path, ckpt: str, device):
    """Rebuild the model from its saved config and load weights."""
    model = EventFFDepthAnythingV2.init_from_config(f"{run}/models/depth_config.yml")
    # best/last also carry optimizer+scheduler (3x the model); mmap skips what we drop.
    state = torch.load(
        f"{run}/models/{ckpt}", map_location="cpu", weights_only=False, mmap=True
    )
    load_depth_weights(model, state.get("model", state))
    return model.to(device).eval()


def undistort_lut(calib: str, width: int, height: int, alpha: float) -> np.ndarray:
    """Sensor pixel -> its place in a rectified pinhole frame, as an ``(h, w, 2)`` lookup.

    One entry per pixel, so rectifying a window is a gather instead of a per-event solve.
    ``alpha`` picks the rectified frame the way ``getOptimalNewCameraMatrix`` does: 0
    zooms in until every rectified pixel is filled (trading away the periphery), 1 keeps
    the whole field of view and leaves the corners empty.

    Pixels the model cannot place come back NaN, which fails the in-frame test and drops
    those events. A wide lens can have them: undistortion inverts ``r * f(r)``, and once
    that curve turns over, observed radii past its peak have no ideal point at all --
    for the 80 deg DVXplorer calibration here, that is the image corners. ``undistortPoints``
    does not report the failure, it just returns whatever its iteration wandered to, so
    every entry is checked by projecting it back through the forward model.
    """
    store = cv2.FileStorage(calib, cv2.FILE_STORAGE_READ)
    assert store.isOpened(), f"cannot read {calib}"
    root = store.root()
    # The intrinsics sit under a node named for the sensor serial, so find it by content.
    cameras = [
        root.getNode(key)
        for key in root.keys()
        if root.getNode(key).isMap()
        and not root.getNode(key).getNode("camera_matrix").empty()
    ]
    assert len(cameras) == 1, f"{calib}: expected one camera node, found {len(cameras)}"
    fisheye = root.getNode("use_fisheye_model")
    assert fisheye.empty() or int(fisheye.real()) == 0, (
        f"{calib} is a fisheye calibration; this maps points with the plumb-bob model"
    )

    k = cameras[0].getNode("camera_matrix").mat()
    d = cameras[0].getNode("distortion_coefficients").mat()
    calib_wh = (cameras[0].getNode("image_width"), cameras[0].getNode("image_height"))
    assert (int(calib_wh[0].real()), int(calib_wh[1].real())) == (width, height), (
        f"{calib} is for {calib_wh[0].real():.0f}x{calib_wh[1].real():.0f}, sensor is {width}x{height}"
    )

    grid = np.stack(np.meshgrid(np.arange(width), np.arange(height)), -1)
    flat = grid.reshape(-1, 1, 2).astype(np.float64)
    # undistortPoints stops after 5 iterations, which on this much barrel leaves a median
    # error of 0.13 px; iterating to convergence brings it to ~1e-10 and rectifies half
    # the sensor again (43% of pixels -> 92%).
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 200, 1e-10)
    ideal = cv2.undistortPointsIter(flat, k, d, None, None, criteria)
    ideal = ideal.reshape(height, width, 2)

    # Only the entries that survive a round trip through the forward model are real.
    rays = np.dstack([ideal, np.ones((height, width, 1))]).reshape(-1, 3)
    back = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), k, d)[0].reshape(-1, 2)
    good = (np.linalg.norm(back - flat.reshape(-1, 2), axis=1) < 0.05).reshape(
        height, width
    )
    assert good.any(), f"{calib}: no pixel could be rectified"

    x, y = ideal[..., 0], ideal[..., 1]
    # Outer box holds every rectifiable pixel; inner is the largest all-filled rectangle,
    # bounded by how far each image edge reaches inward. alpha blends the two.
    outer = (x[good].min(), x[good].max(), y[good].min(), y[good].max())
    edges = (
        x[:, 0][good[:, 0]],
        x[:, -1][good[:, -1]],
        y[0, :][good[0, :]],
        y[-1, :][good[-1, :]],
    )
    inner = (
        edges[0].max() if edges[0].size else outer[0],
        edges[1].min() if edges[1].size else outer[1],
        edges[2].max() if edges[2].size else outer[2],
        edges[3].min() if edges[3].size else outer[3],
    )
    x0, x1, y0, y1 = (i + alpha * (o - i) for i, o in zip(inner, outer))

    # Pixel centres run 0..width-1, so that is the span the box maps onto: an undistorted
    # calibration then comes back as the identity rather than a slight stretch.
    lut = np.full((height, width, 2), np.nan, np.float32)
    lut[..., 0] = np.where(good, (x - x0) / (x1 - x0) * (width - 1), np.nan)
    lut[..., 1] = np.where(good, (y - y0) / (y1 - y0) * (height - 1), np.nan)
    return lut


def read_window(grp, ms_idx, t0_ms: int, window_ms: int, crop, lut=None):
    """Events in ``[t0, t0 + window)`` inside the crop, as ``[x, y, t_rel, p]`` floats."""
    x0, y0, w, h = crop
    i0, i1 = int(ms_idx[t0_ms]), int(ms_idx[t0_ms + window_ms])
    x = grp["x"][i0:i1].astype(np.int32)
    y = grp["y"][i0:i1].astype(np.int32)
    if lut is None:
        xf, yf = x.astype(np.float32), y.astype(np.float32)
    else:
        # Rectified positions are fractional and stay that way: the model reads
        # coordinates, not a pixel grid, so rounding here would only reintroduce the
        # uneven density a remap creates.
        xf, yf = lut[y, x, 0], lut[y, x, 1]

    # Test the grid cell the model will address, not the continuous bound: it splats with
    # round(x * w), so a rectified 639.7 sits inside [0, 640) yet rounds to 640 and walks
    # off the field. Integer pixel coordinates land on themselves, so the raw path is
    # unaffected. NaN (an unrectifiable pixel) compares false and drops out here.
    px, py = np.rint(xf - x0), np.rint(yf - y0)
    inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)

    t_anchor = (t0_ms + window_ms) * 1000
    events = np.empty((int(inside.sum()), 4), np.float32)
    events[:, 0] = (xf[inside] - x0) / w
    events[:, 1] = (yf[inside] - y0) / h
    events[:, 2] = (t_anchor - grp["t"][i0:i1][inside]) / (window_ms * 1000)
    events[:, 3] = grp["p"][i0:i1][inside]
    return events


def event_frame(events, w: int, h: int):
    """Polarity frame, positive red and negative blue, in BGR."""
    frame = np.zeros((h, w, 3), np.uint8)
    x = np.rint(events[:, 0] * w).astype(np.int32).clip(0, w - 1)
    y = np.rint(events[:, 1] * h).astype(np.int32).clip(0, h - 1)
    pos = events[:, 3] > 0
    frame[y[pos], x[pos], 2] = 255
    frame[y[~pos], x[~pos], 0] = 255
    return frame


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay recorded events through a trained depth model, window by window."
    )
    parser.add_argument(
        "--run", required=True, help="Training output dir (has models/)"
    )
    parser.add_argument("--h5", required=True, help="Event sequence HDF5")
    parser.add_argument("--dataset", default="m3ed", choices=sorted(LAYOUTS))
    parser.add_argument(
        "--ckpt", default="best_d1.pth", help="Checkpoint under models/"
    )
    parser.add_argument("--start-ms", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--video", help="Write an mp4 instead of opening a window")
    parser.add_argument("--frames", type=int, default=200, help="Frames for --video")
    parser.add_argument(
        "--focal",
        type=float,
        default=1033.13,
        help="Sensor focal length in px, M3ED's by default; metric runs scale by it",
    )
    parser.add_argument(
        "--depth-range",
        default="0.5,20",
        help="near,far metres of the fixed colour scale (metric runs only)",
    )
    parser.add_argument(
        "--calib", help="OpenCV calibration XML; rectifies event coordinates"
    )
    parser.add_argument(
        "--calib-alpha",
        type=float,
        default=0.0,
        help="getOptimalNewCameraMatrix alpha: 0 crops to valid pixels, 1 keeps the full FOV",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    model = load_model(Path(args.run), args.ckpt, device)

    model_w, model_h, model_t = model.eventff.frame_sizes
    window_ms = model_t
    group, ms_key, src_w, src_h = LAYOUTS[args.dataset]
    crop = ((src_w - model_w) // 2, (src_h - model_h) // 2, model_w, model_h)

    # Whole frame: infer_image resizes the short edge to dav2's size and keeps aspect,
    # so unlike training there is no square crop and the prediction covers the input.
    cparams = torch.tensor([0, 0, model_h, model_w], dtype=torch.int32, device=device)
    cmap = colormaps["magma"]
    # A metric head emits canonical depth; the real thing scales with the focal length.
    metric = model.dav2.head == "sigmoid"
    scale = args.focal / model.dav2_config["focal_canonical"] if metric else 1.0
    near, far = (float(v) for v in args.depth_range.split(","))

    lut = None
    if args.calib:
        lut = undistort_lut(args.calib, src_w, src_h, args.calib_alpha)
        # Rectified coordinates are fractional, so the field has to splat rather than
        # round; rounding would leave cells no event can reach.
        model.eventff.bilinear_splat = True
        kept = (
            (lut[..., 0] >= crop[0])
            & (lut[..., 0] < crop[0] + model_w)
            & (lut[..., 1] >= crop[1])
            & (lut[..., 1] < crop[1] + model_h)
        )
        print(
            f"rectifying with {args.calib} (alpha={args.calib_alpha}); "
            f"{kept.mean():.1%} of sensor pixels land in the frame"
        )

    with h5py.File(args.h5, "r") as f:
        grp, ms_idx = f[group], f[ms_key]
        print(f"{args.h5}: {grp['x'].shape[0]:,} events, {len(ms_idx)} ms")
        print(
            f"crop x0={crop[0]} y0={crop[1]} {model_w}x{model_h}, window {window_ms} ms"
        )
        writer = None
        if args.video:
            writer = cv2.VideoWriter(
                args.video,
                cv2.VideoWriter_fourcc(*"mp4v"),
                1000 / window_ms,  # windows are contiguous, so this plays at real speed
                (2 * model_w, model_h),
            )
            print(f"writing {args.frames} frames to {args.video}")
        else:
            print("n/Enter: next window   q: quit")

        end_ms = args.start_ms + args.frames * window_ms if writer else len(ms_idx)
        t0_ms = args.start_ms
        while t0_ms + window_ms < min(end_ms, len(ms_idx)):
            events = read_window(grp, ms_idx, t0_ms, window_ms, crop, lut)
            ff_events = torch.from_numpy(events).to(device)
            counts = torch.tensor([len(events)], dtype=torch.int32, device=device)

            out = model.infer_image(ff_events, counts, cparams)[0]
            full = torch.ones_like(out, dtype=torch.bool)
            # get_*_image returns matplotlib RGB; cv2 wants BGR.
            pred = cv2.cvtColor(
                get_depth_image(out * scale, full, cmap, near, far)
                if metric
                else get_disparity_image(out, full, cmap),
                cv2.COLOR_RGB2BGR,
            )
            panel = np.hstack([event_frame(events, model_w, model_h), pred])
            print(f"t={t0_ms:6d} ms  {len(events):8,} events")

            if writer is not None:
                writer.write(panel)
            else:
                cv2.imshow("events | disparity", panel)
                if (cv2.waitKey(0) & 0xFF) in (ord("q"), 27):
                    break
            t0_ms += window_ms

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
