"""Manage the lifecycle of a packaged UE5 process.

Responsibilities (v1):
    - Spawn the UE executable with the right flags for headless operation and
      for pointing the plugin at our AF_UNIX socket.
    - Reap it on shutdown (graceful, then SIGTERM, then SIGKILL).
    - Surface early crashes as a clear exception rather than a hang.

Explicitly out of scope: auto-restart, log tailing, GPU selection (we pass
``-graphicsadapter=N`` but verification that CUDA and UE picked the same
device is the caller's problem — see ``implementation.md``).
"""

from __future__ import annotations

import logging
import os
import shlex
import signal
import subprocess
import time
from typing import Sequence

logger = logging.getLogger(__name__)


class UnrealProcessError(RuntimeError):
    pass


class UnrealProcess:
    """Spawns and tears down a packaged UE build."""

    def __init__(
        self,
        executable: str,
        socket_path: str,
        *,
        gpu_id: int = 0,
        scene: str | None = None,
        extra_args: Sequence[str] = (),
        startup_grace_s: float = 2.0,
        shutdown_grace_s: float = 5.0,
    ):
        if not os.path.isfile(executable):
            raise UnrealProcessError(f"UE executable not found: {executable!r}")
        if not os.access(executable, os.X_OK):
            raise UnrealProcessError(f"UE executable not executable: {executable!r}")

        self._executable = executable
        self._socket_path = socket_path
        self._gpu_id = gpu_id
        self._scene = scene
        self._extra_args = list(extra_args)
        self._startup_grace_s = startup_grace_s
        self._shutdown_grace_s = shutdown_grace_s
        self._proc: subprocess.Popen[bytes] | None = None

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        if self._proc is not None:
            raise UnrealProcessError("process already started")

        cmd = [self._executable]
        if self._scene:
            cmd.append(self._scene)
        cmd += [
            f"-NeurosimSocket={self._socket_path}",
            f"-graphicsadapter={self._gpu_id}",
            "-nosound",
            "-unattended",
        ]
        cmd += self._extra_args

        logger.info("Launching UE: %s", " ".join(shlex.quote(c) for c in cmd))
        # Put UE in its own process group so a SIGTERM cleanly reaps any
        # child helper processes UE might spawn.
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        # If UE crashes during PostEngineInit (bad scene path, missing plugin,
        # etc.), fail fast rather than letting the client hang waiting for a
        # socket that will never appear.
        t0 = time.monotonic()
        while time.monotonic() - t0 < self._startup_grace_s:
            rc = self._proc.poll()
            if rc is not None:
                raise UnrealProcessError(
                    f"UE exited during startup with code {rc}. "
                    f"Check the process stdout for details."
                )
            time.sleep(0.05)

    def terminate(self) -> None:
        """Best-effort graceful shutdown.

        Expects the caller to have already sent the ``shutdown`` RPC; this is
        the fallback for when that didn't take.
        """
        if self._proc is None:
            return

        if self._proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self._proc.wait(timeout=self._shutdown_grace_s)
            except subprocess.TimeoutExpired:
                logger.warning("UE did not exit on SIGTERM; sending SIGKILL")
                try:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self._proc.wait(timeout=2.0)

        self._proc = None

    # -- introspection --------------------------------------------------------

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None
