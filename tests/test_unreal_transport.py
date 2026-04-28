"""Tier-1 tests for the UE transport: no UE, no GPU, no network.

Spins up an in-process AF_UNIX server that speaks the same wire format as the
UE plugin and exercises the client state machine end-to-end.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import tempfile
import threading
from collections.abc import Callable

import pytest

from neurosim.core.visual_backend.unreal.transport import (
    RemoteError,
    TransportError,
    UnrealTransport,
)

_HEADER = struct.Struct(">I")


class _FakeUEServer:
    """Minimal AF_UNIX server: one client, one handler function."""

    def __init__(self, handler: Callable[[dict], dict]):
        self._handler = handler
        self._sock_path = os.path.join(
            tempfile.mkdtemp(prefix="neurosim-ue-test-"), "srv.sock"
        )
        self._listen: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def path(self) -> str:
        return self._sock_path

    def start(self) -> None:
        self._listen = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listen.bind(self._sock_path)
        self._listen.listen(1)
        self._listen.settimeout(0.5)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._listen is not None:
            try:
                self._listen.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._listen.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            os.unlink(self._sock_path)
        except OSError:
            pass

    def _run(self) -> None:
        assert self._listen is not None
        while not self._stop.is_set():
            try:
                client, _ = self._listen.accept()
            except (OSError, socket.timeout):
                continue
            try:
                self._serve_client(client)
            finally:
                client.close()

    def _serve_client(self, client: socket.socket) -> None:
        while not self._stop.is_set():
            header = _recv_exact(client, 4)
            if header is None:
                return
            (length,) = _HEADER.unpack(header)
            body = _recv_exact(client, length)
            if body is None:
                return
            req = json.loads(body.decode())
            try:
                result = self._handler(req)
                resp = {"id": req.get("id"), "result": result}
            except Exception as e:
                resp = {"id": req.get("id"), "error": str(e)}
            payload = json.dumps(resp).encode()
            client.sendall(_HEADER.pack(len(payload)) + payload)


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except OSError:
            return None
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


# ---------------------------------------------------------------------------


@pytest.fixture
def fake_server():
    servers: list[_FakeUEServer] = []

    def _make(handler):
        s = _FakeUEServer(handler)
        s.start()
        servers.append(s)
        return s

    yield _make

    for s in servers:
        s.stop()


def test_connect_and_handshake(fake_server):
    def handler(req):
        assert req["method"] == "handshake"
        assert req["params"] == {"client_version": "0.1.0", "gpu_id": 0}
        return {"server_version": "0.1.0", "pid": 42, "sensors": {}}

    srv = fake_server(handler)
    t = UnrealTransport(srv.path, connect_timeout_s=2.0)
    t.connect()
    try:
        r = t.call("handshake", client_version="0.1.0", gpu_id=0)
        assert r == {"server_version": "0.1.0", "pid": 42, "sensors": {}}
    finally:
        t.close()


def test_id_correlation_across_calls(fake_server):
    seen_ids = []

    def handler(req):
        seen_ids.append(req["id"])
        return {"ok": True}

    srv = fake_server(handler)
    with UnrealTransport(srv.path, connect_timeout_s=2.0) as t:
        for _ in range(5):
            t.call("noop")
    assert seen_ids == [1, 2, 3, 4, 5]


def test_remote_error_surfaces_as_remote_error(fake_server):
    def handler(req):
        raise ValueError("boom")

    srv = fake_server(handler)
    with UnrealTransport(srv.path, connect_timeout_s=2.0) as t:
        with pytest.raises(RemoteError, match="bad_method: boom"):
            t.call("bad_method")


def test_connect_times_out_when_no_server():
    t = UnrealTransport("/tmp/does-not-exist.sock", connect_timeout_s=0.5)
    with pytest.raises(TransportError):
        t.connect()


def test_pose_roundtrip(fake_server):
    state = {}

    def handler(req):
        if req["method"] == "set_agent_pose":
            state["pos"] = req["params"]["position"]
            state["rot"] = req["params"]["rotation"]
            return {"ok": True}
        if req["method"] == "get_agent_pose":
            return {"position": state["pos"], "rotation": state["rot"]}
        raise KeyError(req["method"])

    srv = fake_server(handler)
    with UnrealTransport(srv.path, connect_timeout_s=2.0) as t:
        t.call("set_agent_pose", position=[1.0, 2.0, 3.0], rotation=[1.0, 0.0, 0.0, 0.0])
        r = t.call("get_agent_pose")
        assert r["position"] == [1.0, 2.0, 3.0]
        assert r["rotation"] == [1.0, 0.0, 0.0, 0.0]
