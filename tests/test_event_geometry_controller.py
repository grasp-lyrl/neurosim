import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "applications" / "rl"))

from evaluate_event_geometry_controller import (  # noqa: E402
    LatchedGeometryController,
    TrackerPrediction,
)


def prediction(probability, u=0.5, v=0.5, inbound_probability=0.0):
    return TrackerPrediction(
        probability=float(probability),
        centre=np.asarray([u, v], dtype=np.float32),
        geometry=np.zeros(2, dtype=np.float32),
        inbound_probability=float(inbound_probability),
    )


def test_controller_requires_confirmation_and_holds_one_direction():
    controller = LatchedGeometryController(
        threshold=0.9,
        minimum_track_age_steps=0,
        confirm_steps=2,
        hold_steps=3,
        centre_deadband=0.01,
        control_mode="lateral",
    )
    action, committed = controller.step(prediction(0.95, u=0.7))
    assert not committed
    np.testing.assert_array_equal(action, np.zeros(3))

    action, committed = controller.step(prediction(0.95, u=0.68))
    assert committed
    np.testing.assert_array_equal(action, [0.0, 1.0, 0.0])

    # A contradictory centre during the pulse cannot reverse it.
    action, committed = controller.step(prediction(0.99, u=0.2))
    assert not committed
    np.testing.assert_array_equal(action, [0.0, 1.0, 0.0])
    action, _ = controller.step(prediction(0.0))
    np.testing.assert_array_equal(action, [0.0, 1.0, 0.0])


def test_controller_needs_absence_before_rearming():
    controller = LatchedGeometryController(
        threshold=0.9,
        minimum_track_age_steps=0,
        confirm_steps=1,
        hold_steps=1,
        refractory_absence_steps=2,
        control_mode="lateral",
    )
    _, committed = controller.step(prediction(1.0, u=0.7))
    assert committed
    for _ in range(3):
        action, committed = controller.step(prediction(1.0, u=0.2))
        assert not committed
        np.testing.assert_array_equal(action, np.zeros(3))
    controller.step(prediction(0.0))
    controller.step(prediction(0.0))
    _, committed = controller.step(prediction(1.0, u=0.2))
    assert committed


def test_image_plane_controller_is_normalized_and_latched():
    controller = LatchedGeometryController(
        threshold=0.9,
        minimum_track_age_steps=0,
        confirm_steps=1,
        hold_steps=2,
        control_mode="image_plane",
    )
    action, committed = controller.step(prediction(1.0, u=0.8, v=0.9))
    assert committed
    assert action[0] == 0.0
    assert action[1] > 0.0
    assert action[2] > 0.0
    np.testing.assert_allclose(np.linalg.norm(action[1:]), 1.0, atol=1e-6)


def test_central_detection_uses_stable_fallback_side():
    controller = LatchedGeometryController(
        threshold=0.9,
        minimum_track_age_steps=0,
        confirm_steps=1,
        fallback_lateral_sign=-1,
        control_mode="lateral",
    )
    action, _ = controller.step(prediction(1.0, u=0.501))
    np.testing.assert_array_equal(action, [0.0, -1.0, 0.0])


def test_direction_is_latched_before_track_age_delay():
    controller = LatchedGeometryController(
        threshold=0.9,
        track_threshold=0.7,
        minimum_track_age_steps=4,
        confirm_steps=2,
        hold_steps=2,
        motion_lead_steps=0.0,
        control_mode="lateral",
    )
    controller.step(prediction(1.0, u=0.8))
    action, committed = controller.step(prediction(1.0, u=0.75))
    assert not committed
    np.testing.assert_array_equal(action, np.zeros(3))
    # Dither crosses the image centre before the launch delay expires. The
    # executed direction must remain the initially selected positive side.
    controller.step(prediction(1.0, u=0.2))
    action, committed = controller.step(prediction(1.0, u=0.2))
    assert committed
    np.testing.assert_array_equal(action, [0.0, 1.0, 0.0])


def test_confirmed_inbound_track_overrides_age_delay():
    controller = LatchedGeometryController(
        threshold=0.9,
        track_threshold=0.7,
        minimum_track_age_steps=12,
        inbound_threshold=0.95,
        inbound_confirm_steps=2,
        confirm_steps=2,
        control_mode="lateral",
    )
    action, committed = controller.step(
        prediction(0.99, u=0.8, inbound_probability=0.99)
    )
    assert not committed
    np.testing.assert_array_equal(action, np.zeros(3))
    action, committed = controller.step(
        prediction(0.99, u=0.75, inbound_probability=0.99)
    )
    assert committed
    assert controller.last_release_reason == "inbound"
    np.testing.assert_array_equal(action, [0.0, 1.0, 0.0])


def test_single_inbound_spike_does_not_override_age_delay():
    controller = LatchedGeometryController(
        threshold=0.9,
        minimum_track_age_steps=12,
        inbound_threshold=0.95,
        inbound_confirm_steps=2,
        confirm_steps=2,
        control_mode="lateral",
    )
    controller.step(prediction(0.99, u=0.8, inbound_probability=0.99))
    action, committed = controller.step(
        prediction(0.99, u=0.75, inbound_probability=0.1)
    )
    assert not committed
    np.testing.assert_array_equal(action, np.zeros(3))
