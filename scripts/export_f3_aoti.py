"""Export the depth model, or F3 alone, to an AOTI package for the Orin.

    CUDA_HOME=/usr/local/cuda-12.8 python -m scripts.export_f3_aoti \
        --conf applications/f3_depth_training/configs/depth_training_config_voltmeter_50ms.yml \
        --ckpt outputs/<run>/models/best_weights.pth --out /tmp/depth.pt2
"""

import argparse
import statistics
import time

import torch
import torch.nn as nn
import yaml

from applications.f3_depth_training.nets import F3, EventFFDepthAnythingV2
from applications.f3_depth_training.train_depth_nonrec import predict_full_frame


class Field(nn.Module):
    """F3 alone: events [N, 4] -> feature field, autocast where the tracer can see it."""

    def __init__(self, f3: F3):
        super().__init__()
        self.f3 = f3

    def forward(self, events, counts):
        with torch.autocast("cuda", torch.bfloat16):
            return self.f3(events[:, :3], counts)


class Frame(nn.Module):
    """The whole model: events [N, 4] -> disparity [1, H, W], aspect preserved."""

    def __init__(self, model: EventFFDepthAnythingV2, height: int, width: int):
        super().__init__()
        self.model, self.height, self.width = model, height, width

    def forward(self, events, counts):
        with torch.autocast("cuda", torch.bfloat16):
            return predict_full_frame(
                self.model, events, counts, self.height, self.width
            )


def build(args, device) -> tuple[nn.Module, int, int]:
    """The module to export, and the sensor it runs on."""
    conf = yaml.safe_load(open(args.conf))
    if args.module == "f3":
        f3 = F3.from_config(args.conf)
        if args.ckpt:
            from applications.f3_depth_training.nets import load_f3_weights

            load_f3_weights(f3, args.ckpt)
        w, h, _ = f3.frame_sizes
        return Field(f3).to(device).eval(), h, w

    dav2 = {k: v for k, v in conf["dav2_config"].items() if k != "ckpt"}
    model = EventFFDepthAnythingV2(conf["eventff"]["config"], dav2)
    if args.ckpt:
        from applications.f3_depth_training.nets import load_depth_weights

        state = torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)
        load_depth_weights(model, state.get("model", state))
    w, h, _ = model.eventff.frame_sizes
    return Frame(model.to(device).eval(), h, w), h, w


def example(count: int, w: int, h: int, device):
    """One tick on whole pixels and whole age buckets, as the loader delivers it."""
    events = torch.rand(count, 4, device=device)
    events[:, 0] = torch.randint(0, w, (count,), device=device) / w
    events[:, 1] = torch.randint(0, h, (count,), device=device) / h
    return events, torch.tensor([count], dtype=torch.int32, device=device)


def median_ms(fn, repeats: int) -> float:
    """Median milliseconds per call, after a warmup."""
    for _ in range(8):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e3)
    return statistics.median(times)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conf", required=True, help="training config for `depth`, F3 yaml for `f3`"
    )
    parser.add_argument("--module", choices=["depth", "f3"], default="depth")
    parser.add_argument(
        "--ckpt", help="weights; random if omitted, for a shape-only export"
    )
    parser.add_argument("--out", required=True, help="where to write the .pt2 package")
    parser.add_argument(
        "--events", type=int, default=400_000, help="event count to trace with"
    )
    parser.add_argument(
        "--check",
        type=int,
        nargs="*",
        default=[100_000, 2_000_000],
        help="event counts to verify the package against eager at",
    )
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    # The autotuner benchmarks kernels on random values for every graph input, and the
    # scatter's indices only lie in range for a real tick, so it asserts device-side.
    torch._inductor.config.triton.autotune_at_compile_time = False
    torch.set_float32_matmul_precision("high")

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    module, h, w = build(args, device)

    events, counts = example(args.events, w, h, device)
    with torch.no_grad():
        program = torch.export.export(
            module,
            (events, counts),
            dynamic_shapes=({0: torch.export.Dim("n_events", min=2)}, None),
        )
        path = torch._inductor.aoti_compile_and_package(program, package_path=args.out)
    print(f"wrote {path}")

    runner = torch._inductor.aoti_load_package(path)
    print(f"{'events':>10}{'AOTI ms':>10}{'eager ms':>10}   max |delta|")
    for count in args.check:
        events, counts = example(count, w, h, device)
        with torch.no_grad():
            gap = (runner(events, counts) - module(events, counts)).abs().max()
            aoti = median_ms(lambda: runner(events, counts), 20)
            eager = median_ms(lambda: module(events, counts), 20)
        print(f"{count:>10,}{aoti:>10.2f}{eager:>10.2f}   {gap:.2e}", flush=True)


if __name__ == "__main__":
    main()
