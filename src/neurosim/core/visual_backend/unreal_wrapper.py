"""Unreal Engine 5 visual backend.

Phase-1 scope: connects to a packaged UE build, moves a floating-camera pawn,
returns placeholder tensors for color/depth. Phase 2 swaps the placeholders
for zero-copy CUDA views onto Vulkan-external-memory render targets.

See ``implementation.md`` at the repo root.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch

from neurosim.core.visual_backend.base import VisualBackendProtocol
from neurosim.core.visual_backend.corner_detector import FeatureDetectionResult
from neurosim.core.visual_backend.unreal.process_manager import UnrealProcess
from neurosim.core.visual_backend.unreal.transport import (
    UnrealTransport,
    default_socket_path,
)

logger = logging.getLogger(__name__)


class UnrealWrapper(VisualBackendProtocol):
    """Visual backend backed by an Unreal Engine 5 process.

    Settings dict (see ``implementation.md`` for the full contract):

        backend_type:     "unreal"
        ue_executable:    path to packaged UE binary (e.g. MyProject.sh)
        scene:            UE package path (e.g. /Game/Maps/Minimal); optional at init
        gpu_id:           int; passed to UE and used for torch.cuda
        default_agent:    int; kept for parity with Habitat, ignored here
        agent_height:     float; kept for parity; floating camera has no body
        agent_radius:     float; ditto
        sensors:          {name: {type, width, height, hfov, zfar, position, orientation, ...}}

    Only ``color`` and ``depth`` sensor types are supported in v1. Phase-2 work
    turns the placeholder render tensors into zero-copy CUDA views.
    """

    def __init__(self, settings: dict[str, Any]):
        self.settings = settings
        self._gpu_id = int(settings.get("gpu_id", 0))
        self._device = torch.device(f"cuda:{self._gpu_id}")

        # Pre-flight validation: reject unsupported sensor types loudly now
        # rather than at first render.
        for name, cfg in settings.get("sensors", {}).items():
            t = cfg.get("type")
            if t not in ("color", "depth"):
                raise NotImplementedError(
                    f"UE backend v1 supports only color/depth sensors; "
                    f"sensor {name!r} has type {t!r}"
                )
        self._sensor_cfg: dict[str, dict[str, Any]] = dict(settings.get("sensors", {}))

        socket_path = settings.get("socket_path") or default_socket_path()
        self._socket_path = socket_path

        self._process = UnrealProcess(
            executable=settings["ue_executable"],
            socket_path=socket_path,
            gpu_id=self._gpu_id,
            scene=settings.get("scene"),
        )
        self._transport = UnrealTransport(socket_path)

        self._process.start()
        try:
            self._transport.connect()
            handshake = self._transport.call(
                "handshake",
                client_version="0.1.0",
                gpu_id=self._gpu_id,
            )
            logger.info(
                "UnrealWrapper: handshake ok (server=%s pid=%s)",
                handshake.get("server_version"), handshake.get("pid"),
            )
        except Exception:
            # Bring UE down if the handshake failed; otherwise it will linger.
            self._process.terminate()
            raise

        logger.info("════════════════════════════════════════════════════════════════")
        logger.info(
            "✅ Unreal simulator initialized (scene=%s, gpu=%d)",
            settings.get("scene"), self._gpu_id,
        )
        logger.info("════════════════════════════════════════════════════════════════")

    # -- protocol: agent state ------------------------------------------------

    def update_agent_state(
        self, position: np.ndarray, quaternion: np.ndarray
    ) -> None:
        """Update the floating-camera pose.

        Args:
            position: ``(3,)`` world position, Habitat conventions (y-up, meters).
            quaternion: ``(4,)`` ``[w, x, y, z]``.
        """
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat = np.asarray(
            quaternion.components if hasattr(quaternion, "components") else quaternion,
            dtype=np.float64,
        ).reshape(4)
        self._transport.call(
            "set_agent_pose",
            position=pos.tolist(),
            rotation=quat.tolist(),
        )

    def get_agent_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Read the current pose back. Not in ``VisualBackendProtocol``, but
        useful for Phase-1 smoke tests."""
        r = self._transport.call("get_agent_pose")
        return (
            np.asarray(r["position"], dtype=np.float32),
            np.asarray(r["rotation"], dtype=np.float32),
        )

    # -- protocol: rendering --------------------------------------------------
    #
    # Phase 1 returns deterministic placeholder tensors so upstream code can
    # exercise the call graph. Phase 2 replaces these with zero-copy CUDA
    # views onto Vulkan-external-memory render targets.

    def _require_sensor(self, uuid: str, expected_type: str) -> dict[str, Any]:
        cfg = self._sensor_cfg.get(uuid)
        if cfg is None:
            raise KeyError(f"no sensor named {uuid!r}")
        if cfg.get("type") != expected_type:
            raise ValueError(
                f"sensor {uuid!r} has type {cfg.get('type')!r}, "
                f"render_{expected_type} called"
            )
        return cfg

    def render_color(self, uuid: str) -> torch.Tensor:
        cfg = self._require_sensor(uuid, "color")
        self._transport.call("render_frame")
        # Phase-2 placeholder: a black frame on the target GPU.
        return torch.zeros(
            (cfg["height"], cfg["width"], 4),
            dtype=torch.uint8,
            device=self._device,
        )

    def render_depth(self, uuid: str) -> torch.Tensor:
        cfg = self._require_sensor(uuid, "depth")
        self._transport.call("render_frame")
        return torch.zeros(
            (cfg["height"], cfg["width"]),
            dtype=torch.float32,
            device=self._device,
        )

    # -- protocol: v2 sensors (deliberately not supported yet) ----------------

    def render_events(
        self, uuid: str, time: int, to_numpy: bool = False
    ) -> tuple[Any, ...] | None:
        raise NotImplementedError(
            "UE backend v1 supports only color/depth; event cameras are v2."
        )

    def render_optical_flow(self, uuid: str) -> torch.Tensor:
        raise NotImplementedError("UE backend v1 does not implement optical flow.")

    def render_corners(self, uuid: str) -> FeatureDetectionResult:
        raise NotImplementedError("UE backend v1 does not implement corner detection.")

    def render_edges(self, uuid: str) -> torch.Tensor:
        raise NotImplementedError("UE backend v1 does not implement edge detection.")

    def render_grayscale(self, uuid: str) -> torch.Tensor:
        raise NotImplementedError("UE backend v1 does not implement grayscale.")

    def render_semantic(self, uuid: str) -> torch.Tensor:
        raise NotImplementedError("UE backend v1 does not implement semantic.")

    def render_navmesh(self, meters_per_pixel: float = 0.1) -> np.ndarray:
        raise NotImplementedError("UE backend v1 does not expose navmesh.")

    def update_dynamic_obstacles(self, sim_time: float, dt: float) -> None:
        # No-op rather than raise: lets trajectory / rollout code that calls
        # this unconditionally keep working when dynamic obstacles aren't
        # configured.
        return

    # -- protocol: lifecycle --------------------------------------------------

    def reconfigure(self, new_settings: dict[str, Any]) -> None:
        """Hard reconfigure: kill the UE process and spawn a new one.

        Per the plan (``implementation.md``), we do NOT attempt an in-process
        scene swap in v1 — kill + respawn is simpler and race-free.
        """
        logger.info("UnrealWrapper: reconfiguring — restarting UE process.")
        self.close()
        self.__init__(new_settings)

    def close(self) -> None:
        """Best-effort shutdown of the UE process."""
        try:
            if self._transport is not None:
                try:
                    self._transport.call("shutdown")
                except Exception:  # UE may already be exiting
                    pass
                self._transport.close()
        finally:
            if self._process is not None:
                self._process.terminate()

        # Clean up the socket file if UE didn't (defensive).
        try:
            if os.path.exists(self._socket_path):
                os.unlink(self._socket_path)
        except OSError:
            pass
