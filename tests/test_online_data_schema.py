"""Unit tests for the online-data sample + schema layer (PR1).

Pure-Python: no Habitat / GPU. Covers the wire unit (SampleMeta /
TimeAlignedSample / ownership guard) and the sensor schema (kind/role
resolution, dispatch-table totality, build validation, cross-producer checks).
"""

import numpy as np
import pytest

from neurosim.online_data import (
    SampleMeta,
    TimeAlignedSample,
    assert_owned_array,
    SensorKind,
    SensorRole,
    SampleSchema,
    infer_kind,
    BATCH_STRATEGY_FOR_KIND,
)


SENSOR_CONFIGS = {
    "depth_camera_1": {"type": "depth", "height": 480, "width": 640},
    "color_camera_1": {"type": "color", "height": 480, "width": 640},
    "event_camera_1": {"type": "event", "height": 480, "width": 640},
    "imu_1": {"type": "imu"},
}


# --------------------------------------------------------------------------- #
# SampleMeta / TimeAlignedSample
# --------------------------------------------------------------------------- #
def test_episode_id_unique_across_workers():
    a = SampleMeta.make_episode_id(worker_id=0, episode_idx=5)
    b = SampleMeta.make_episode_id(worker_id=1, episode_idx=5)
    c = SampleMeta.make_episode_id(worker_id=0, episode_idx=6)
    assert a != b and a != c and b != c


def test_episode_id_range_checks():
    with pytest.raises(ValueError):
        SampleMeta.make_episode_id(worker_id=-1, episode_idx=0)
    with pytest.raises(ValueError):
        SampleMeta.make_episode_id(worker_id=0, episode_idx=-1)


def _meta(**kw):
    base = dict(
        worker_id=0,
        spec_id=0,
        scene="apartment_1",
        seed=42,
        t_us=1000,
        window_us=50,
        anchor_uuids=("depth_camera_1",),
        episode_id=SampleMeta.make_episode_id(0, 0),
        step_idx=0,
        is_first=True,
        is_last=False,
        sample_uid=0,
    )
    base.update(kw)
    return SampleMeta(**base)


def test_sample_nbytes_and_uuids():
    depth = np.zeros((480, 640), dtype=np.float32)
    events = {
        "x": np.zeros(10, np.int16),
        "y": np.zeros(10, np.int16),
        "t": np.zeros(10, np.int64),
        "p": np.zeros(10, np.int8),
    }
    s = TimeAlignedSample(
        meta=_meta(), sensors={"depth_camera_1": depth, "event_camera_1": events}
    )
    assert set(s.uuids()) == {"depth_camera_1", "event_camera_1"}
    assert s.nbytes() == depth.nbytes + sum(v.nbytes for v in events.values())


def test_assert_owned_array():
    owned = np.zeros((4, 4), dtype=np.float32)
    assert assert_owned_array(owned) is owned

    view = owned[1:3]  # a view: OWNDATA is False
    assert not view.flags["OWNDATA"]
    with pytest.raises(ValueError):
        assert_owned_array(view, name="depth")

    with pytest.raises(TypeError):
        assert_owned_array([1, 2, 3])  # not an ndarray


# --------------------------------------------------------------------------- #
# kind inference + dispatch table
# --------------------------------------------------------------------------- #
def test_infer_kind():
    assert infer_kind("depth") is SensorKind.FRAME
    assert infer_kind("color") is SensorKind.FRAME
    assert infer_kind("event") is SensorKind.EVENT_STREAM
    assert infer_kind("imu") is SensorKind.VECTOR_STREAM
    assert infer_kind("event", override="vector") is SensorKind.VECTOR
    with pytest.raises(ValueError):
        infer_kind("totally_unknown_sensor")


def test_batch_strategy_table_total_over_kinds():
    # The batcher dispatches on this table; it must cover every kind.
    assert set(BATCH_STRATEGY_FOR_KIND) == set(SensorKind)
    assert BATCH_STRATEGY_FOR_KIND[SensorKind.FRAME] == "stack"
    assert BATCH_STRATEGY_FOR_KIND[SensorKind.EVENT_STREAM] == "concat_counts"


# --------------------------------------------------------------------------- #
# SampleSchema build
# --------------------------------------------------------------------------- #
def test_schema_v1_events_depth():
    schema = SampleSchema.from_sensor_configs(
        SENSOR_CONFIGS,
        anchor=["depth_camera_1"],
        stream=["event_camera_1"],
    )
    assert schema.anchor_uuids == ("depth_camera_1",)
    assert schema.stream_uuids == ("event_camera_1",)
    assert schema.latest_uuids == ()
    assert schema.deliver_uuids() == ["depth_camera_1", "event_camera_1"]

    assert schema.kind_of("depth_camera_1") is SensorKind.FRAME
    assert schema.role_of("depth_camera_1") is SensorRole.ANCHOR
    assert schema.batch_strategy_of("event_camera_1") == "concat_counts"

    depth = schema.specs["depth_camera_1"]
    assert depth.shape == (480, 640)
    assert depth.dtype == "float32"

    ev = schema.specs["event_camera_1"]
    assert ev.shape is None  # variable-length stream
    assert ev.extras["width"] == 640 and ev.extras["height"] == 480


def test_schema_color_channels_and_latest_role():
    schema = SampleSchema.from_sensor_configs(
        SENSOR_CONFIGS,
        anchor=["depth_camera_1"],
        stream=["event_camera_1"],
        latest=["color_camera_1"],
    )
    assert schema.specs["color_camera_1"].shape == (480, 640, 3)
    assert schema.specs["color_camera_1"].dtype == "uint8"
    assert schema.role_of("color_camera_1") is SensorRole.LATEST


def test_schema_deliver_order_explicit():
    schema = SampleSchema.from_sensor_configs(
        SENSOR_CONFIGS,
        anchor=["depth_camera_1"],
        stream=["event_camera_1"],
        deliver=["event_camera_1", "depth_camera_1"],
    )
    assert schema.deliver_uuids() == ["event_camera_1", "depth_camera_1"]


def test_schema_requires_anchor():
    with pytest.raises(ValueError, match="at least one anchor"):
        SampleSchema.from_sensor_configs(
            SENSOR_CONFIGS, anchor=[], stream=["event_camera_1"]
        )


def test_schema_rejects_multi_role():
    with pytest.raises(ValueError, match="multiple roles"):
        SampleSchema.from_sensor_configs(
            SENSOR_CONFIGS,
            anchor=["depth_camera_1"],
            stream=["depth_camera_1"],
        )


def test_schema_rejects_streaming_anchor():
    with pytest.raises(ValueError, match="streaming kind"):
        SampleSchema.from_sensor_configs(SENSOR_CONFIGS, anchor=["event_camera_1"])


def test_schema_rejects_unknown_uuid():
    with pytest.raises(ValueError, match="not found"):
        SampleSchema.from_sensor_configs(SENSOR_CONFIGS, anchor=["nope_camera"])


def test_schema_rejects_delivered_without_role():
    with pytest.raises(ValueError, match="no role"):
        SampleSchema.from_sensor_configs(
            SENSOR_CONFIGS,
            anchor=["depth_camera_1"],
            deliver=["depth_camera_1", "color_camera_1"],
        )


# --------------------------------------------------------------------------- #
# cross-producer validation
# --------------------------------------------------------------------------- #
def test_validate_against_matching_producer():
    schema = SampleSchema.from_sensor_configs(
        SENSOR_CONFIGS, anchor=["depth_camera_1"], stream=["event_camera_1"]
    )
    # A second producer with identical sensor geometry passes.
    schema.validate_against(dict(SENSOR_CONFIGS), producer="p1")


def test_validate_against_shape_mismatch():
    schema = SampleSchema.from_sensor_configs(
        SENSOR_CONFIGS, anchor=["depth_camera_1"], stream=["event_camera_1"]
    )
    other = {
        "depth_camera_1": {"type": "depth", "height": 240, "width": 320},
        "event_camera_1": {"type": "event", "height": 480, "width": 640},
    }
    with pytest.raises(ValueError, match="shape mismatch"):
        schema.validate_against(other, producer="p1")


def test_validate_against_kind_mismatch():
    schema = SampleSchema.from_sensor_configs(
        SENSOR_CONFIGS, anchor=["depth_camera_1"], stream=["event_camera_1"]
    )
    other = {
        "depth_camera_1": {"type": "depth", "height": 480, "width": 640},
        # event UUID now declared as a color sensor -> kind mismatch
        "event_camera_1": {"type": "color", "height": 480, "width": 640},
    }
    with pytest.raises(ValueError, match="kind mismatch"):
        schema.validate_against(other, producer="p1")


def test_validate_against_missing_uuid():
    schema = SampleSchema.from_sensor_configs(
        SENSOR_CONFIGS, anchor=["depth_camera_1"], stream=["event_camera_1"]
    )
    with pytest.raises(ValueError, match="missing"):
        schema.validate_against({"depth_camera_1": SENSOR_CONFIGS["depth_camera_1"]})
