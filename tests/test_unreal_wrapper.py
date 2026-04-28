"""Tier-1 tests for ``UnrealWrapper``: no UE, no GPU.

We stub out ``UnrealProcess`` and ``UnrealTransport`` at import-seam boundaries
so the wrapper exercises its own logic — not the transport, not the subprocess
manager (those have their own tests).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import neurosim.core.visual_backend.unreal_wrapper as uw_mod


class _FakeProcess:
    def __init__(self, *_, **__):
        self._alive = False

    def start(self) -> None:
        self._alive = True

    def terminate(self) -> None:
        self._alive = False

    def is_alive(self) -> bool:
        return self._alive

    @property
    def pid(self) -> int:
        return 4242


class _FakeTransport:
    def __init__(self, path: str, connect_timeout_s: float = 30.0):
        self.path = path
        self.connected = False
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._state = {"pos": [0.0, 0.0, 0.0], "rot": [1.0, 0.0, 0.0, 0.0]}

    def connect(self) -> None:
        self.connected = True

    def close(self) -> None:
        self.connected = False

    def call(self, method: str, **params: Any) -> dict[str, Any]:
        self.calls.append((method, params))
        if method == "handshake":
            return {"server_version": "0.1.0", "pid": 42, "sensors": {}}
        if method == "set_agent_pose":
            self._state["pos"] = list(params["position"])
            self._state["rot"] = list(params["rotation"])
            return {"ok": True}
        if method == "get_agent_pose":
            return {"position": self._state["pos"], "rotation": self._state["rot"]}
        if method == "render_frame":
            return {"ok": True}
        if method == "shutdown":
            return {"ok": True}
        raise KeyError(method)


@pytest.fixture
def settings_minimal(tmp_path):
    # UnrealProcess validates the executable path, but we've stubbed it out.
    # Pass any path; the stub ignores it.
    return {
        "backend_type": "unreal",
        "ue_executable": str(tmp_path / "Fake.sh"),
        "scene": "/Game/Maps/Minimal",
        "gpu_id": 0,
        "default_agent": 0,
        "agent_height": 1.5,
        "agent_radius": 0.1,
        "socket_path": str(tmp_path / "ue.sock"),
        "sensors": {
            "rgb": {"type": "color", "width": 64, "height": 48,
                    "hfov": 90, "zfar": 100.0,
                    "position": [0, 0, 0], "orientation": [0, 0, 0]},
            "dep": {"type": "depth", "width": 64, "height": 48,
                    "hfov": 90, "zfar": 100.0,
                    "position": [0, 0, 0], "orientation": [0, 0, 0]},
        },
    }


@pytest.fixture
def patch_wrapper_deps(monkeypatch):
    monkeypatch.setattr(uw_mod, "UnrealProcess", _FakeProcess)
    monkeypatch.setattr(uw_mod, "UnrealTransport", _FakeTransport)


def test_init_runs_handshake(settings_minimal, patch_wrapper_deps):
    w = uw_mod.UnrealWrapper(settings_minimal)
    try:
        # The transport saw a handshake call with the right params.
        calls = w._transport.calls  # type: ignore[attr-defined]
        assert calls[0][0] == "handshake"
        assert calls[0][1]["gpu_id"] == 0
    finally:
        w.close()


def test_update_agent_state_forwards_pose(settings_minimal, patch_wrapper_deps):
    w = uw_mod.UnrealWrapper(settings_minimal)
    try:
        pos = np.array([1.0, 2.0, 3.0])
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        w.update_agent_state(pos, quat)
        methods = [m for m, _ in w._transport.calls]  # type: ignore[attr-defined]
        assert "set_agent_pose" in methods
        params = dict(w._transport.calls)["set_agent_pose"]  # type: ignore[attr-defined]
        assert params["position"] == [1.0, 2.0, 3.0]
        assert params["rotation"] == [1.0, 0.0, 0.0, 0.0]
    finally:
        w.close()


def test_get_agent_pose_roundtrip(settings_minimal, patch_wrapper_deps):
    w = uw_mod.UnrealWrapper(settings_minimal)
    try:
        w.update_agent_state(
            np.array([4.0, 5.0, 6.0]),
            np.array([0.0, 1.0, 0.0, 0.0]),
        )
        pos, rot = w.get_agent_pose()
        np.testing.assert_allclose(pos, [4.0, 5.0, 6.0])
        np.testing.assert_allclose(rot, [0.0, 1.0, 0.0, 0.0])
    finally:
        w.close()


def test_render_color_and_depth_shapes(settings_minimal, patch_wrapper_deps):
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available in this environment")

    w = uw_mod.UnrealWrapper(settings_minimal)
    try:
        color = w.render_color("rgb")
        depth = w.render_depth("dep")
        assert color.shape == (48, 64, 4)
        assert color.dtype == torch.uint8
        assert depth.shape == (48, 64)
        assert depth.dtype == torch.float32
        assert color.device.type == "cuda"
        assert depth.device.type == "cuda"
    finally:
        w.close()


def test_unsupported_sensor_type_rejected_at_init(settings_minimal, patch_wrapper_deps):
    settings_minimal["sensors"]["ev"] = {
        "type": "event", "width": 32, "height": 32, "hfov": 90, "zfar": 100,
        "position": [0, 0, 0], "orientation": [0, 0, 0],
    }
    with pytest.raises(NotImplementedError, match="color/depth"):
        uw_mod.UnrealWrapper(settings_minimal)


def test_v2_sensor_methods_raise(settings_minimal, patch_wrapper_deps):
    w = uw_mod.UnrealWrapper(settings_minimal)
    try:
        with pytest.raises(NotImplementedError):
            w.render_events("rgb", 0)
        with pytest.raises(NotImplementedError):
            w.render_optical_flow("rgb")
        with pytest.raises(NotImplementedError):
            w.render_corners("rgb")
        with pytest.raises(NotImplementedError):
            w.render_edges("rgb")
        with pytest.raises(NotImplementedError):
            w.render_grayscale("rgb")
        with pytest.raises(NotImplementedError):
            w.render_navmesh()
    finally:
        w.close()


def test_render_on_wrong_type_raises(settings_minimal, patch_wrapper_deps):
    w = uw_mod.UnrealWrapper(settings_minimal)
    try:
        with pytest.raises(ValueError, match="render_color"):
            w.render_color("dep")  # dep is type=depth, not color
        with pytest.raises(ValueError, match="render_depth"):
            w.render_depth("rgb")
    finally:
        w.close()


def test_update_dynamic_obstacles_is_noop(settings_minimal, patch_wrapper_deps):
    w = uw_mod.UnrealWrapper(settings_minimal)
    try:
        w.update_dynamic_obstacles(0.0, 0.01)  # must not raise
    finally:
        w.close()


def test_close_sends_shutdown(settings_minimal, patch_wrapper_deps):
    w = uw_mod.UnrealWrapper(settings_minimal)
    w.close()
    methods = [m for m, _ in w._transport.calls]  # type: ignore[attr-defined]
    assert "shutdown" in methods
