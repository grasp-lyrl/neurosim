"""Convert a rosbag2 MCAP flight recording into a Neurosim-layout event H5.

WHY: the depth tools all read Neurosim H5 event groups
(``applications/f3_depth_training/replay_depth_h5.py``, ``scripts/h5_to_video.py``,
``scripts/visualize_h5_events.py``), but the NeuroFly quadrotor logs arrive as ROS2
MCAP bags carrying a ``dv_ros2_msgs/msg/EventArray`` stream.

Event messages are decoded straight out of their CDR payloads with numpy. A 60 s bag
holds a few hundred million events and the generic ROS2 decoder builds one Python
object per event, which would take hours; the layout below reads the same bytes in
one vectorized pass. Every other topic is low rate, so it goes through the ROS2
decoder and needs no special casing.

Timestamps are microseconds relative to the first event, matching the ``t`` convention
of every other Neurosim H5 (streams that start before the first event go negative).

HOW (neurosim env, repo root):

    python scripts/mcap_to_h5.py data/nf1_2026-09-04-13-53-43_0.mcap
    python scripts/mcap_to_h5.py data/nf1_....mcap --out /tmp/head.h5 --max-seconds 5
"""

import argparse
import struct
from pathlib import Path

import h5py
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from tqdm import tqdm

EVENT_SCHEMA = "dv_ros2_msgs/msg/EventArray"

# CDR aligns every field to its own width, counted from the start of the message body,
# and an EventArray's events start 4-aligned. So the first event packs tight into 13
# bytes (x@0 y@2 sec@4 nsec@8 polarity@12) and leaves the next one 2-aligned but not
# 4-aligned: from there on each event sits on a 16-byte stride with two pad bytes in
# front of its timestamp. Both layouts are checked against the reference ROS2 decoder
# by tests/test_mcap_to_h5.py.
FIRST_EVENT_BYTES = 14
EVENT_DTYPE = np.dtype(
    {
        "names": ["x", "y", "sec", "nsec", "p"],
        "formats": ["<u2", "<u2", "<i4", "<u4", "u1"],
        "offsets": [0, 2, 6, 10, 14],
        "itemsize": 16,
    }
)

# Fields worth keeping off the low-rate topics: h5 dataset -> attribute path on the msg.
IMU_FIELDS = {
    "angular_velocity": (
        "angular_velocity.x",
        "angular_velocity.y",
        "angular_velocity.z",
    ),
    "linear_acceleration": (
        "linear_acceleration.x",
        "linear_acceleration.y",
        "linear_acceleration.z",
    ),
    "orientation": (
        "orientation.x",
        "orientation.y",
        "orientation.z",
        "orientation.w",
    ),
}
ODOM_FIELDS = {
    "position": (
        "pose.pose.position.x",
        "pose.pose.position.y",
        "pose.pose.position.z",
    ),
    "orientation": (
        "pose.pose.orientation.x",
        "pose.pose.orientation.y",
        "pose.pose.orientation.z",
        "pose.pose.orientation.w",
    ),
    "linear_velocity": (
        "twist.twist.linear.x",
        "twist.twist.linear.y",
        "twist.twist.linear.z",
    ),
    "angular_velocity": (
        "twist.twist.angular.x",
        "twist.twist.angular.y",
        "twist.twist.angular.z",
    ),
}
RANGE_FIELDS = {"range": ("range",)}
# --polarity choice -> (stored values, h5 dtype)
POLARITY_DTYPE = {"01": "u1", "pm1": "i1"}
# ROS type -> (h5 group, field map). Anything else in the bag is skipped.
LOW_RATE_TOPICS = {
    "sensor_msgs/msg/Imu": ("imu", IMU_FIELDS),
    "nav_msgs/msg/Odometry": ("odom", ODOM_FIELDS),
    "sensor_msgs/msg/Range": ("range", RANGE_FIELDS),
}


def to_us(sec: int, nsec: int) -> int:
    """ROS ``builtin_interfaces/Time`` -> integer microseconds."""
    return int(sec) * 1_000_000 + int(nsec) // 1000


def parse_event_array(
    raw: bytes,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode one CDR ``EventArray`` payload into ``(width, height, x, y, t_us, p)``."""
    # 4-byte encapsulation header, then header.stamp, then the frame_id string.
    frame_len = struct.unpack_from("<I", raw, 12)[0]
    off = 16 + frame_len
    off += -off % 4  # height is a uint32, so it is 4-aligned
    height, width, count = struct.unpack_from("<III", raw, off)
    off += 12

    x = np.empty(count, np.uint16)
    y = np.empty(count, np.uint16)
    t = np.empty(count, np.int64)
    p = np.empty(count, np.uint8)
    if count == 0:
        return width, height, x, y, t, p

    x[0], y[0], sec0, nsec0, p[0] = struct.unpack_from("<HHiIB", raw, off)
    t[0] = to_us(sec0, nsec0)
    if count > 1:
        start = off + FIRST_EVENT_BYTES
        need = EVENT_DTYPE.itemsize * (count - 1)
        tail = np.frombuffer(
            raw, np.uint8, count=min(need, len(raw) - start), offset=start
        )
        if tail.size < need:  # the final event carries no trailing padding
            tail = np.concatenate([tail, np.zeros(need - tail.size, np.uint8)])
        rest = tail.view(EVENT_DTYPE)
        x[1:], y[1:], p[1:] = rest["x"], rest["y"], rest["p"]
        t[1:] = rest["sec"].astype(np.int64) * 1_000_000 + rest["nsec"] // 1000
    return width, height, x, y, t, p


def getpath(msg, path: str) -> float:
    """``msg.pose.pose.position.x`` from the string ``"pose.pose.position.x"``."""
    for part in path.split("."):
        msg = getattr(msg, part)
    return msg


class EventWriter:
    """Resizable ``x/y/t/p`` datasets, filled in blocks so the bag never lands in RAM."""

    def __init__(self, group: h5py.Group, p_dtype: str = "u1", chunk: int = 1 << 20):
        self.n = 0
        # p_dtype follows --polarity: HDF5 *clips* on a narrowing cast rather than
        # wrapping, so -1 written into a uint8 column would silently land as 0.
        self.dsets = {
            name: group.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                dtype=dtype,
                chunks=(chunk,),
                compression="lzf",
                shuffle=True,
            )
            for name, dtype in (("x", "u2"), ("y", "u2"), ("t", "u4"), ("p", p_dtype))
        }

    def append(self, **columns: np.ndarray) -> None:
        added = len(next(iter(columns.values())))
        for name, values in columns.items():
            dset = self.dsets[name]
            dset.resize(self.n + added, axis=0)
            dset[self.n :] = values
        self.n += added


def build_ms_to_idx(t: h5py.Dataset, block: int = 50_000_000) -> np.ndarray:
    """``ms_to_idx[m]`` = first event index with ``t >= m * 1000`` us, as build_ms_to_idx.py."""
    total = int(t.shape[0])
    if total == 0:
        return np.zeros(1, np.int64)
    edges = np.arange(int(t[-1]) // 1000 + 1, dtype=np.int64) * 1000
    idx = np.empty(edges.shape, np.int64)
    done = 0
    for start in range(0, total, block):
        chunk = t[start : start + block]
        # Edges landing inside this block get their index; later ones are filled next round.
        hits = np.searchsorted(chunk, edges[done:], side="left") + start
        inside = hits < start + len(chunk)
        idx[done : done + int(inside.sum())] = hits[inside]
        done += int(inside.sum())
    idx[done:] = total
    return idx


def write_low_rate(h5: h5py.File, rows: dict, t0_us: int) -> None:
    """One group per low-rate topic: ``t`` plus a column per kept field."""
    for name, (stamps, fields) in rows.items():
        if not stamps:
            continue
        group = h5.create_group(name)
        group.create_dataset("t", data=np.asarray(stamps, np.int64) - t0_us)
        for field, values in fields.items():
            column = np.asarray(values, np.float64)
            # A one-component field (range) is a scalar per sample, not a length-1 vector.
            group.create_dataset(
                field, data=column.reshape(-1) if column.shape[1] == 1 else column
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a rosbag2 MCAP recording into a Neurosim event H5."
    )
    parser.add_argument("mcap", help="Input .mcap bag")
    parser.add_argument("--out", help="Output .h5 (default: the bag's name)")
    parser.add_argument(
        "--sensor", default="event_camera_1", help="H5 group for the event stream"
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        help="Stop after this many seconds of events, for a quick look",
    )
    parser.add_argument(
        "--block-events",
        type=int,
        default=20_000_000,
        help="Events buffered before each H5 append",
    )
    parser.add_argument(
        "--polarity",
        default="01",
        choices=["01", "pm1"],
        help="Store polarity as 0/1 (M3ED) or -1/+1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.mcap)
    out = Path(args.out) if args.out else src.with_suffix(".h5")

    with open(src, "rb") as f:
        summary = make_reader(f).get_summary()
    topics = {
        channel.topic: summary.schemas[channel.schema_id].name
        for channel in summary.channels.values()
    }
    event_topics = [t for t, schema in topics.items() if schema == EVENT_SCHEMA]
    assert len(event_topics) == 1, (
        f"expected one {EVENT_SCHEMA} topic, got {event_topics}"
    )
    event_topic = event_topics[0]
    wanted = [event_topic] + [t for t, s in topics.items() if s in LOW_RATE_TOPICS]
    print(f"{src.name}: {summary.statistics.message_count:,} messages")
    for topic, schema in sorted(topics.items()):
        mark = " " if topic in wanted else "-"  # '-' topics are dropped
        print(f" {mark} {topic:42s} {schema}")

    rows = {
        group: ([], {field: [] for field in fields})
        for group, fields in LOW_RATE_TOPICS.values()
    }
    total, t0_us, t_last, out_of_order = 0, None, None, 0
    width = height = 0

    with h5py.File(out, "w") as h5:
        events = EventWriter(
            h5.create_group(args.sensor), POLARITY_DTYPE[args.polarity]
        )
        buffered: list[tuple[np.ndarray, ...]] = []
        buffered_n = 0

        def flush() -> None:
            nonlocal buffered, buffered_n
            if not buffered:
                return
            x, y, t, p = (np.concatenate(col) for col in zip(*buffered))
            events.append(x=x, y=y, t=(t - t0_us).astype(np.uint32), p=p)
            buffered, buffered_n = [], 0

        with open(src, "rb") as f:
            # Raw messages, not iter_decoded_messages: that decodes eagerly, and running
            # every EventArray through the per-event ROS2 decoder is what we are avoiding.
            reader = make_reader(f)
            factory = DecoderFactory()
            stream = reader.iter_messages(topics=wanted)
            progress = tqdm(stream, total=summary.statistics.message_count, unit="msg")
            for schema, channel, message in progress:
                if channel.topic == event_topic:
                    width, height, x, y, t, p = parse_event_array(message.data)
                    if len(x) == 0:
                        continue
                    if t0_us is None:
                        t0_us = int(t[0])
                    if args.max_seconds and t[0] - t0_us > args.max_seconds * 1e6:
                        break
                    if t_last is not None and t[0] < t_last:
                        out_of_order += 1
                    out_of_order += int((np.diff(t) < 0).sum())
                    t_last = int(t[-1])
                    if args.polarity == "pm1":
                        p = np.where(p > 0, 1, -1).astype(np.int8)
                    buffered.append((x, y, t, p))
                    buffered_n += len(x)
                    total += len(x)
                    if buffered_n >= args.block_events:
                        flush()
                        progress.set_postfix_str(f"{total / 1e6:.0f}M events")
                else:
                    group, fields = LOW_RATE_TOPICS[schema.name]
                    decoded = factory.decoder_for(channel.message_encoding, schema)(
                        message.data
                    )
                    stamps, columns = rows[group]
                    stamps.append(
                        to_us(decoded.header.stamp.sec, decoded.header.stamp.nanosec)
                    )
                    for field, paths in fields.items():
                        columns[field].append(
                            [getpath(decoded, path) for path in paths]
                        )
            flush()

        assert total, "no events in the bag"
        sensor = h5[args.sensor]
        sensor.attrs["width"] = width
        sensor.attrs["height"] = height
        sensor.attrs["t0_us"] = t0_us  # absolute ROS time of t == 0
        sensor.attrs["source"] = f"{src.name}:{event_topic}"
        write_low_rate(h5, rows, t0_us)
        if out_of_order:
            print(
                f"WARNING: {out_of_order:,} timestamps step backwards; ms_to_idx assumes sorted t"
            )
        ms_to_idx = build_ms_to_idx(sensor["t"])
        sensor.create_dataset("ms_to_idx", data=ms_to_idx, dtype="i8")

    span = (t_last - t0_us) / 1e6
    print(
        f"\n{out}: {total:,} events, {width}x{height}, {span:.2f} s "
        f"({total / span / 1e6:.2f} Mev/s), ms_to_idx {len(ms_to_idx):,}"
    )
    for name, (stamps, _) in rows.items():
        if stamps:
            print(f"  {name}: {len(stamps):,} samples")
    print(f"  size on disk: {out.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
