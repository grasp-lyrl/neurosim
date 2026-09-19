"""The vectorized CDR event parse must agree with the reference ROS2 decoder."""

import importlib.util
import struct
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mcap_to_h5.py"
DATA = Path(__file__).resolve().parents[1] / "data"
# Bags live one rosbag2 directory each (data/<set>/<run>/<stamp>.mcap), and are named by
# recording time, so pick up whichever one is present.
BAGS = sorted(DATA.rglob("*.mcap"))

mcap = pytest.importorskip("mcap.reader")
mcap_ros2 = pytest.importorskip("mcap_ros2.decoder")


def load_module():
    spec = importlib.util.spec_from_file_location("mcap_to_h5", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encode_event_array(width: int, height: int, events: list[tuple]) -> bytes:
    """A CDR ``dv_ros2_msgs/EventArray`` payload, aligning each field as ROS2 does."""
    body = bytearray(
        struct.pack("<iII", 7, 8000, 1) + b"\x00"
    )  # stamp + empty frame_id
    body += b"\x00" * (-len(body) % 4)
    body += struct.pack("<III", height, width, len(events))
    for x, y, sec, nsec, polarity in events:
        body += b"\x00" * (-len(body) % 2)
        body += struct.pack("<HH", x, y)
        body += b"\x00" * (-len(body) % 4)
        body += struct.pack("<iIB", sec, nsec, polarity)
    return b"\x00\x01\x00\x00" + bytes(body)


@pytest.mark.parametrize("count", [0, 1, 2, 3, 17])
def test_parse_matches_hand_encoded(count):
    """Round-trip against an independent encoder, including the 0/1/2-event edges."""
    module = load_module()
    rng = np.random.default_rng(count)
    events = [
        (
            int(rng.integers(0, 640)),
            int(rng.integers(0, 480)),
            7,
            1000 * i + 3000,
            i % 2,
        )
        for i in range(count)
    ]
    width, height, x, y, t, p = module.parse_event_array(
        encode_event_array(640, 480, events)
    )

    assert (width, height) == (640, 480)
    assert len(x) == count
    for i, (ex, ey, sec, nsec, polarity) in enumerate(events):
        assert (x[i], y[i], p[i]) == (ex, ey, polarity)
        assert t[i] == sec * 1_000_000 + nsec // 1000


@pytest.mark.skipif(not BAGS, reason="no flight bag downloaded")
def test_parse_matches_ros2_decoder():
    """Same events as mcap_ros2's per-message decoder, on real recorded payloads."""
    module = load_module()
    reader = mcap.make_reader(
        BAGS[0].open("rb"), decoder_factories=[mcap_ros2.DecoderFactory()]
    )
    checked = 0
    for _, _, message, decoded in reader.iter_decoded_messages(
        topics=["/neurofly1/events"]
    ):
        width, height, x, y, t, p = module.parse_event_array(message.data)
        assert (width, height) == (decoded.width, decoded.height)
        assert len(x) == len(decoded.events)
        expected = np.array(
            [
                (e.x, e.y, module.to_us(e.ts.sec, e.ts.nanosec), e.polarity)
                for e in decoded.events
            ]
        )
        assert np.array_equal(x, expected[:, 0])
        assert np.array_equal(y, expected[:, 1])
        assert np.array_equal(t, expected[:, 2])
        assert np.array_equal(p, expected[:, 3])
        checked += 1
        if checked == 25:
            break
    assert checked == 25


def test_ms_to_idx_indexes_first_event_at_or_after_each_ms():
    module = load_module()
    t = np.array([0, 500, 1000, 1500, 4000, 4001], np.uint32)
    idx = module.build_ms_to_idx(t, block=3)  # forces the multi-block path
    assert idx.tolist() == [0, 2, 4, 4, 4]
    assert all(t[i] >= ms * 1000 for ms, i in enumerate(idx) if i < len(t))


@pytest.mark.parametrize("mode,expected", [("01", {0, 1}), ("pm1", {-1, 1})])
def test_event_writer_keeps_negative_polarity(tmp_path, mode, expected):
    """HDF5 clips on a narrowing cast, so a -1/+1 column needs a signed dataset."""
    h5py = pytest.importorskip("h5py")
    module = load_module()
    p = np.array([0, 1, 1, 0], np.uint8)
    if mode == "pm1":
        p = np.where(p > 0, 1, -1).astype(np.int8)

    path = tmp_path / "events.h5"
    with h5py.File(path, "w") as f:
        writer = module.EventWriter(
            f.create_group("event_camera_1"), module.POLARITY_DTYPE[mode]
        )
        zeros = np.zeros(4, np.uint16)
        writer.append(x=zeros, y=zeros, t=zeros.astype(np.uint32), p=p)
    with h5py.File(path) as f:
        assert set(f["event_camera_1"]["p"][:].tolist()) == expected
