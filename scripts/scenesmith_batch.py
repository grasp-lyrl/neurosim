"""Convert a whole SceneSmith dataset into baked, KTX2-compressed, Habitat-ready GLBs.

    python scripts/scenesmith_batch.py \\
        --dataset /local_hdd/richeek/scenesmith-example-scenes \\
        --out /local_hdd/richeek/scenesmith-glb \\
        --blender /local_hdd/richeek/tools/blender-4.2.9-linux-x64/blender \\
        --subsets Room House --workers 4 --gpus 0 1
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

STATE_NAME = "_state.json"
LOG_DIR_NAME = "_logs"
MIN_FREE_GB = 50.0

_state_lock = threading.Lock()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, help="dir holding <subset>/scene_*.tar")
    ap.add_argument("--out", required=True, help="output dir for the GLBs")
    ap.add_argument("--blender", required=True, help="path to the blender binary")
    ap.add_argument(
        "--bake-script",
        default=str(Path(__file__).with_name("scenesmith_bake_glb.py")),
        help="the Blender-side script (defaults to the one next to this file)",
    )
    ap.add_argument(
        "--subsets",
        nargs="*",
        default=["Room", "House"],
        help="subset dirs to convert; the rest are ablations (default: Room House)",
    )
    ap.add_argument("--workers", type=int, default=2, help="scenes baked in parallel")
    ap.add_argument("--gpus", nargs="*", type=int, default=[0], help="CUDA devices")
    ap.add_argument("--samples", type=int, default=192)
    ap.add_argument(
        "--compress",
        default="uastc",
        choices=["uastc", "etc1s", "none"],
        help="KTX2 mode for the baked textures (default: uastc)",
    )
    ap.add_argument("--max-res", type=int, default=2048)
    ap.add_argument("--timeout", type=float, default=3600.0, help="seconds per scene")
    ap.add_argument(
        "--limit", type=int, default=0, help="stop after N scenes (0 = all)"
    )
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--redo", action="store_true", help="rebake even if the GLB exists")
    ap.add_argument(
        "--keep-extracted", action="store_true", help="debugging: leave the tree behind"
    )
    ap.add_argument(
        "--scratch", default=None, help="where to extract (default: --out/_work)"
    )
    return ap.parse_args()


def load_state(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"warning: {path} is corrupt, starting fresh", file=sys.stderr)
    return {}


def save_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True))
    tmp.replace(path)  # atomic


def free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / 1e9


def find_blend(root: Path) -> Path | None:
    direct = root / "combined_house" / "house.blend"
    if direct.exists():
        return direct
    found = sorted(root.glob("*/combined_house/house.blend"))
    return found[0] if found else None


def compress_textures(glb: Path, mode: str, timeout: float) -> str | None:
    # must end in .glb, or gltf-transform writes a loose glTF tree
    tmp = glb.with_name(f".part_{glb.name}")
    proc = subprocess.run(
        ["npx", "--yes", "@gltf-transform/cli", mode, str(glb), str(tmp)],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        tail = proc.stderr.strip().splitlines()
        return f"{mode} failed: {tail[-1] if tail else proc.returncode}"
    tmp.replace(glb)
    return None


def convert_one(tar_path: Path, args, out_glb: Path, log_path: Path, gpu: int) -> dict:
    started = time.time()
    scratch = Path(args.scratch or (Path(args.out) / "_work"))
    scratch.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{tar_path.stem}_", dir=scratch))
    try:
        if free_gb(scratch) < MIN_FREE_GB:
            return {"status": "failed", "error": "out of disk space", "seconds": 0}

        with tarfile.open(tar_path) as tf:
            tf.extractall(work)  # noqa: S202

        blend = find_blend(work)
        if blend is None:
            return {
                "status": "failed",
                "error": "no combined_house/house.blend",
                "seconds": 0,
            }

        cmd = [
            args.blender,
            "-b",
            str(blend),
            "-P",
            args.bake_script,
            "--",
            "--out",
            str(out_glb),
            "--samples",
            str(args.samples),
            "--max-res",
            str(args.max_res),
        ]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        with log_path.open("w") as log:
            proc = subprocess.run(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=args.timeout,
                check=False,
            )

        if proc.returncode != 0:
            return {
                "status": "failed",
                "error": f"blender exit {proc.returncode}",
                "log": str(log_path),
                "seconds": round(time.time() - started, 1),
            }
        if not out_glb.exists():
            return {
                "status": "failed",
                "error": "blender wrote no GLB",
                "log": str(log_path),
                "seconds": round(time.time() - started, 1),
            }
        baked_bytes = out_glb.stat().st_size
        if args.compress != "none":
            error = compress_textures(out_glb, args.compress, args.timeout)
            if error:
                return {
                    "status": "failed",
                    "error": error,
                    "log": str(log_path),
                    "seconds": round(time.time() - started, 1),
                }
        return {
            "status": "done",
            "compress": args.compress,
            "baked_bytes": baked_bytes,
            "bytes": out_glb.stat().st_size,
            "seconds": round(time.time() - started, 1),
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "failed",
            "error": f"timeout after {args.timeout}s",
            "log": str(log_path),
            "seconds": round(time.time() - started, 1),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "seconds": round(time.time() - started, 1),
        }
    finally:
        if not args.keep_extracted:
            shutil.rmtree(work, ignore_errors=True)


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / LOG_DIR_NAME).mkdir(exist_ok=True)
    state_path = out_dir / STATE_NAME
    state = load_state(state_path)

    if not Path(args.blender).exists():
        raise SystemExit(f"blender not found: {args.blender}")
    if not Path(args.bake_script).exists():
        raise SystemExit(f"bake script not found: {args.bake_script}")
    if args.compress != "none":
        for tool, why in (
            ("npx", "nodejs, for gltf-transform"),
            ("ktx", "KTX-Software"),
        ):
            if shutil.which(tool) is None:
                raise SystemExit(
                    f"--compress {args.compress} needs `{tool}` on PATH ({why})"
                )

    tars: list[Path] = []
    for subset in args.subsets:
        subset_dir = dataset / subset
        if not subset_dir.is_dir():
            print(f"warning: no such subset {subset_dir}", file=sys.stderr)
            continue
        tars.extend(sorted(subset_dir.glob("scene_*.tar")))
    if not tars:
        raise SystemExit(f"no scene tars under {dataset} for subsets {args.subsets}")

    todo = []
    for tar_path in tars:
        key = f"{tar_path.parent.name}/{tar_path.stem}"
        out_glb = out_dir / tar_path.parent.name / f"{tar_path.stem}.glb"
        record = state.get(key, {})
        if record.get("status") == "done" and out_glb.exists() and not args.redo:
            continue
        if record.get("status") == "failed" and not (args.retry_failed or args.redo):
            continue
        todo.append((key, tar_path, out_glb))
    if args.limit:
        todo = todo[: args.limit]

    done_already = sum(1 for r in state.values() if r.get("status") == "done")
    print(
        f"{len(tars)} scenes in {args.subsets}: {done_already} already done, "
        f"{len(todo)} to convert, {args.workers} workers on GPUs {args.gpus}",
        flush=True,
    )
    if not todo:
        return

    started = time.time()
    counters = {"done": 0, "failed": 0}

    def run(index_item):
        index, (key, tar_path, out_glb) = index_item
        gpu = args.gpus[index % len(args.gpus)]
        out_glb.parent.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / LOG_DIR_NAME / f"{key.replace('/', '_')}.log"
        record = convert_one(tar_path, args, out_glb, log_path, gpu)
        record["gpu"] = gpu
        record["when"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with _state_lock:
            state[key] = record
            save_state(state_path, state)
            counters[record["status"]] = counters.get(record["status"], 0) + 1
            finished = counters["done"] + counters["failed"]
            rate = (time.time() - started) / max(finished, 1)
            left = (len(todo) - finished) * rate / 3600.0
            flag = "ok " if record["status"] == "done" else "FAIL"
            print(
                f"[{finished}/{len(todo)}] {flag} {key} "
                f"({record.get('seconds', 0):.0f}s, gpu{gpu})"
                + (f" -- {record.get('error')}" if record["status"] == "failed" else "")
                + f" | {left:.1f}h left",
                flush=True,
            )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run, enumerate(todo)))

    elapsed = (time.time() - started) / 3600.0
    print(
        f"finished in {elapsed:.1f}h: {counters['done']} converted, "
        f"{counters['failed']} failed (state: {state_path})"
    )


main()
