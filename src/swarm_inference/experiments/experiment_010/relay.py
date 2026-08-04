"""Independent coordinator-relay process for the relayed TCP baseline."""

from __future__ import annotations

import json
import os
import socket
import socketserver
import struct
import subprocess
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_010.schemas import NetworkShapeProfile
from swarm_inference.experiments.experiment_010.transport import NetworkShaper
from swarm_inference.experiments.experiment_010.wire import (
    MAX_FRAME_BYTES,
    ExpertPacket,
    decode_packet,
    encode_packet,
    frame_with_length,
)

_LENGTH = struct.Struct(">Q")


def _recv_exact(connection: socket.socket, byte_count: int) -> bytes:
    payload = bytearray()
    while len(payload) < byte_count:
        chunk = connection.recv(byte_count - len(payload))
        if not chunk:
            raise ConnectionError("relay peer closed a partial frame")
        payload.extend(chunk)
    return bytes(payload)


def _recv_frame(connection: socket.socket) -> bytes:
    size = _LENGTH.unpack(_recv_exact(connection, _LENGTH.size))[0]
    if size > MAX_FRAME_BYTES:
        raise ValueError("relay frame exceeds maximum size")
    return _recv_exact(connection, size)


class RelayServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        target: tuple[str, int],
        profile: NetworkShapeProfile,
        metrics_path: Path,
    ) -> None:
        self.target = target
        self.profile = profile
        self.shaper = NetworkShaper(profile)
        self.metrics_path = metrics_path
        self.forwarded_requests = 0
        self.forwarded_bytes = 0
        self.failures = 0
        self.metrics_write_failures = 0
        self.last_error: str | None = None
        self.metrics_lock = threading.Lock()
        self.started_ns = time.time_ns()
        super().__init__(address, RelayHandler, bind_and_activate=True)
        self.request_queue_size = profile.queue_depth

    def snapshot(self) -> dict[str, Any]:
        return {
            "pid": os.getpid(),
            "target": f"{self.target[0]}:{self.target[1]}",
            "endpoint": f"{self.server_address[0]}:{self.server_address[1]}",
            "forwarded_requests": self.forwarded_requests,
            "forwarded_bytes": self.forwarded_bytes,
            "failures": self.failures,
            "metrics_write_failures": self.metrics_write_failures,
            "last_error": self.last_error,
            "started_ns": self.started_ns,
            "shaper": self.shaper.snapshot(),
        }

    def save_metrics(self) -> None:
        # Metrics persistence is observational and must never break the tensor
        # data plane (OneDrive and antivirus scanners can briefly lock a path on
        # Windows). A per-thread temporary also makes concurrent coordinators
        # safe.
        with self.metrics_lock:
            temporary = self.metrics_path.with_name(
                f"{self.metrics_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
            )
            try:
                temporary.write_text(json.dumps(self.snapshot(), indent=2) + "\n", encoding="utf-8")
                temporary.replace(self.metrics_path)
            except OSError as error:
                self.metrics_write_failures += 1
                self.last_error = f"metrics persistence: {type(error).__name__}: {error}"
                with suppress(OSError):
                    temporary.unlink()


class RelayHandler(socketserver.BaseRequestHandler):
    server: RelayServer

    def handle(self) -> None:
        try:
            payload = _recv_frame(self.request)
            packet = decode_packet(payload)
            if packet.kind == "control" and packet.semantic.get("command") == "relay_metrics":
                response = encode_packet(
                    ExpertPacket(
                        kind="control",
                        semantic={"ok": True, "metrics": self.server.snapshot()},
                        blobs=(),
                    )
                )
                self.request.sendall(frame_with_length(response))
                return
            # Patched Colibri keeps one socket per destination worker. Preserve
            # that connection shape through the relay so every frame—not merely
            # the first MoE call—traverses the measured shaped data plane.
            with socket.create_connection(self.server.target, timeout=60.0) as worker:
                worker.settimeout(3600.0)
                while True:
                    framed_size = len(payload) + _LENGTH.size
                    with self.server.shaper.flow(3600.0):
                        self.server.shaper.enforce(framed_size, direction="relay_to_worker")
                    worker.sendall(frame_with_length(payload))
                    response = _recv_frame(worker)
                    with self.server.shaper.flow(3600.0):
                        self.server.shaper.enforce(
                            len(response) + _LENGTH.size,
                            direction="worker_to_relay",
                        )
                    self.request.sendall(frame_with_length(response))
                    self.server.forwarded_requests += 1
                    self.server.forwarded_bytes += framed_size + len(response) + _LENGTH.size
                    if self.server.forwarded_requests % 64 == 0:
                        self.server.save_metrics()
                    try:
                        payload = _recv_frame(self.request)
                    except (ConnectionError, OSError):
                        break
        except Exception as error:
            self.server.failures += 1
            self.server.last_error = f"{type(error).__name__}: {error}"
            self.server.save_metrics()
            response = encode_packet(
                ExpertPacket(
                    kind="control",
                    semantic={"ok": False, "error": f"{type(error).__name__}: {error}"},
                    blobs=(),
                )
            )
            with suppress(ConnectionError, OSError):
                self.request.sendall(frame_with_length(response))


@dataclass(slots=True)
class RelayProcess:
    process: subprocess.Popen[bytes]
    endpoint: str
    ready_path: Path
    metrics_path: Path
    stdout_path: Path
    stderr_path: Path


class ExpertRelayManager:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.relays: list[RelayProcess] = []
        self.lifecycle_records: list[dict[str, Any]] = []
        self._lifecycle_by_pid: dict[int, dict[str, Any]] = {}

    def start(
        self,
        *,
        target_endpoint: str,
        profile: NetworkShapeProfile,
        timeout_seconds: float = 15.0,
    ) -> RelayProcess:
        index = len(self.relays)
        directory = (self.root / f"relay-{index:02d}").resolve()
        if self.root not in directory.parents:
            raise ValueError("relay directory escaped manager root")
        directory.mkdir(parents=True, exist_ok=True)
        config_path = directory / "relay-config.json"
        ready_path = directory / "ready.json"
        metrics_path = directory / "metrics.json"
        if ready_path.is_file():
            ready_path.unlink()
        config_path.write_text(
            json.dumps(
                {
                    "target_endpoint": target_endpoint,
                    "profile": profile.model_dump(mode="json"),
                    "host": "127.0.0.1",
                    "port": 0,
                    "metrics_path": str(metrics_path),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        stdout_path, stderr_path = directory / "stdout.log", directory / "stderr.log"
        stdout, stderr = stdout_path.open("ab"), stderr_path.open("ab")
        repository_root = Path(__file__).resolve().parents[4]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            part
            for part in (str(repository_root / "src"), environment.get("PYTHONPATH", ""))
            if part
        )
        command = [
            sys.executable,
            "-m",
            "swarm_inference.experiments.experiment_010.relay_main",
            "--config",
            str(config_path),
            "--ready",
            str(ready_path),
        ]
        started_ns = time.time_ns()
        process = subprocess.Popen(
            command,
            cwd=repository_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        stdout.close()
        stderr.close()
        deadline = time.monotonic() + timeout_seconds
        try:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"relay exited with {process.returncode}")
                if ready_path.is_file():
                    ready = json.loads(ready_path.read_text(encoding="utf-8"))
                    relay = RelayProcess(
                        process=process,
                        endpoint=str(ready["endpoint"]),
                        ready_path=ready_path,
                        metrics_path=metrics_path,
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                    )
                    self.relays.append(relay)
                    lifecycle = {
                        "role": "expert_relay",
                        "pid": process.pid,
                        "command": command,
                        "target_endpoint": target_endpoint,
                        "endpoint": relay.endpoint,
                        "network_profile": profile.name,
                        "started_ns": started_ns,
                        "stopped_ns": None,
                        "exit_code": None,
                        "status": "RUNNING",
                    }
                    self.lifecycle_records.append(lifecycle)
                    self._lifecycle_by_pid[process.pid] = lifecycle
                    return relay
                time.sleep(0.05)
            raise TimeoutError("relay did not become ready")
        except Exception:
            self._stop(process)
            self.lifecycle_records.append(
                {
                    "role": "expert_relay",
                    "pid": process.pid,
                    "command": command,
                    "target_endpoint": target_endpoint,
                    "network_profile": profile.name,
                    "started_ns": started_ns,
                    "stopped_ns": time.time_ns(),
                    "exit_code": process.poll(),
                    "status": "START_FAILED",
                }
            )
            raise

    @staticmethod
    def _stop(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is None:
            process.terminate()
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=3)
            if process.poll() is None:
                process.kill()
                process.wait(timeout=3)

    @staticmethod
    def snapshot(relay: RelayProcess, *, timeout_seconds: float = 10.0) -> dict[str, Any]:
        host, separator, raw_port = relay.endpoint.rpartition(":")
        if not separator:
            raise ValueError(f"invalid relay endpoint {relay.endpoint!r}")
        request = encode_packet(
            ExpertPacket(
                kind="control",
                semantic={"command": "relay_metrics"},
                blobs=(),
            )
        )
        with socket.create_connection((host, int(raw_port)), timeout=timeout_seconds) as client:
            client.settimeout(timeout_seconds)
            client.sendall(frame_with_length(request))
            response = decode_packet(_recv_frame(client))
        if response.kind != "control" or not response.semantic.get("ok"):
            raise RuntimeError(str(response.semantic.get("error", "relay snapshot failed")))
        metrics = dict(response.semantic["metrics"])
        relay.metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        return metrics

    def snapshots(self) -> list[dict[str, Any]]:
        return [self.snapshot(relay) for relay in self.relays]

    def close(self) -> None:
        for relay in self.relays:
            with suppress(Exception):
                self.snapshot(relay)
            self._stop(relay.process)
            lifecycle = self._lifecycle_by_pid.get(relay.process.pid)
            if lifecycle is not None:
                lifecycle.update(
                    {
                        "stopped_ns": time.time_ns(),
                        "exit_code": relay.process.poll(),
                        "exit_expected": True,
                        "termination_mode": "manager_terminate",
                        "status": "STOPPED",
                    }
                )
        self.relays.clear()

    def __enter__(self) -> ExpertRelayManager:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
