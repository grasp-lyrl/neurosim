"""Compress a neurosim H5 recording in place with LZF, after the run wrote it uncompressed.

python scripts/compress_h5.py outputs/flight.h5 [more.h5 ...]
"""

import argparse
import os
from pathlib import Path

import h5py

COPY_BYTES = 256 * 2**20


def copy_compressed(src: h5py.File, dst: h5py.File) -> None:
    """Every group, dataset and attribute of src into dst, datasets LZF + shuffle."""
    dst.attrs.update(src.attrs)

    def copy(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if isinstance(obj, h5py.Group):
            dst.require_group(name).attrs.update(obj.attrs)
            return
        if obj.ndim == 0:
            out = dst.create_dataset(name, data=obj[()], dtype=obj.dtype)
        else:
            # one frame per chunk: a compressed chunk is decompressed whole to read any of it
            chunks = (1, *obj.shape[1:]) if obj.ndim >= 3 else obj.chunks or True
            out = dst.create_dataset(
                name,
                shape=obj.shape,
                maxshape=obj.maxshape,
                dtype=obj.dtype,
                chunks=chunks,
                compression="lzf",
                shuffle=True,
            )
            rows = max(1, COPY_BYTES // max(1, obj[:1].nbytes))
            for start in range(0, obj.shape[0], rows):
                out[start : start + rows] = obj[start : start + rows]
        out.attrs.update(obj.attrs)

    src.visititems(copy)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("h5", type=Path, nargs="+", help="recordings to compress")
    for path in parser.parse_args().h5:
        tmp = path.with_suffix(".lzf.tmp")
        with h5py.File(path, "r") as src, h5py.File(tmp, "w", libver="latest") as dst:
            copy_compressed(src, dst)
        before = path.stat().st_size
        os.replace(tmp, path)
        print(
            f"{path}: {before / 2**20:.0f} MiB -> {path.stat().st_size / 2**20:.0f} MiB"
        )


if __name__ == "__main__":
    main()
