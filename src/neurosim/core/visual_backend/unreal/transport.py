"""JSON-over-AF_UNIX client for the Neurosim UE bridge.

Wire format: ``[uint32 big-endian length][utf-8 json body]``. Each request is a
single JSON object ``{"id": int, "method": str, "params": dict}``; each
response is ``{"id": int, "result": {...}}`` or ``{"id": int, "error": str}``.

Only one request is in flight at a time: the client blocks until the matching
response arrives. If a higher-frequency control plane is ever needed (e.g. for
multi-sensor parallel render kicks), swap in an id-indexed map here.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import struct
import threading
from typing import Any

logger = logging.getLogger(__name__)

_FRAME_HEADER = struct.Struct(">I")
_MAX_FRAME_BYTES = 16 << 20  # 16 MiB; server enforces the same cap.


class TransportError(RuntimeError):
    """Raised on framing / connection failures."""


class RemoteError(RuntimeError):
    """Raised when the server returns an ``error`` response."""


class UnrealTransport:
    """Synchronous JSON client. Not thread-safe; one instance per UE process."""

    def __init__(self, socket_path: str, connect_timeout_s: float = 30.0):
        self._path = socket_path
        self._connect_timeout_s = connect_timeout_s
        self._sock: socket.socket | None = None
        self._next_id = 1
        self._lock = threading.Lock()

    # -- connection lifecycle -------------------------------------------------

    def connect(self) -> None:
        """Connect to the UE server, retrying until ``connect_timeout_s`` elapses.

        UE takes several seconds to boot, bind the socket, and load the module;
        we spin on ECONNREFUSED rather than failing on the first attempt.
        """
        if self._sock is not None:
            return

        import time
        deadline = time.monotonic() + self._connect_timeout_s
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(self._path)
                self._sock = s
                logger.info("UnrealTransport: connected to %s", self._path)
                return
            except (FileNotFoundError, ConnectionRefusedError) as e:
                last_err = e
                time.sleep(0.25)
        raise TransportError(
            f"Could not connect to UE socket {self._path!r} within "
            f"{self._connect_timeout_s}s: {last_err!r}"
        )

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._sock.close()
            self._sock = None

    def __enter__(self) -> UnrealTransport:
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- request/response -----------------------------------------------------

    def call(self, method: str, **params: Any) -> dict[str, Any]:
        """Send one request, block for the matching response, return ``result``.

        Raises:
            TransportError: connection / framing failure.
            RemoteError: server returned ``error``.
        """
        if self._sock is None:
            raise TransportError("transport is not connected")

        with self._lock:
            req_id = self._next_id
            self._next_id += 1

            payload = json.dumps(
                {"id": req_id, "method": method, "params": params},
                separators=(",", ":"),
            ).encode("utf-8")
            self._send_frame(payload)

            resp_bytes = self._recv_frame()

        try:
            resp = json.loads(resp_bytes.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise TransportError(f"malformed server response: {e!r}") from None

        if resp.get("id") != req_id:
            raise TransportError(
                f"id mismatch: sent {req_id} got {resp.get('id')}"
            )
        if "error" in resp:
            raise RemoteError(f"{method}: {resp['error']}")
        return resp.get("result", {})

    # -- framing --------------------------------------------------------------

    def _send_frame(self, payload: bytes) -> None:
        if len(payload) > _MAX_FRAME_BYTES:
            raise TransportError(f"frame too large: {len(payload)} bytes")
        assert self._sock is not None
        try:
            self._sock.sendall(_FRAME_HEADER.pack(len(payload)) + payload)
        except OSError as e:
            raise TransportError(f"send failed: {e!r}") from None

    def _recv_frame(self) -> bytes:
        header = self._recv_exact(_FRAME_HEADER.size)
        (length,) = _FRAME_HEADER.unpack(header)
        if length == 0 or length > _MAX_FRAME_BYTES:
            raise TransportError(f"invalid frame length {length}")
        return self._recv_exact(length)

    def _recv_exact(self, n: int) -> bytes:
        assert self._sock is not None
        buf = bytearray()
        while len(buf) < n:
            try:
                chunk = self._sock.recv(n - len(buf))
            except OSError as e:
                raise TransportError(f"recv failed: {e!r}") from None
            if not chunk:
                raise TransportError("connection closed by server")
            buf.extend(chunk)
        return bytes(buf)


def default_socket_path(pid: int | None = None) -> str:
    """Return a per-process socket path under ``$XDG_RUNTIME_DIR`` or ``/tmp``."""
    base = os.environ.get("XDG_RUNTIME_DIR") or "/tmp"
    pid = pid if pid is not None else os.getpid()
    return os.path.join(base, f"neurosim-ue-{pid}.sock")
