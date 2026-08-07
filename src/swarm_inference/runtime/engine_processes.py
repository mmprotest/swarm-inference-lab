"""Owned, hash-verified engine process lifecycle."""

from __future__ import annotations

import hashlib
import ipaddress
import os
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


def sha256_file(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def require_private_bind_host(host: str) -> str:
    """Reject wildcard and public addresses for unauthenticated engine protocols."""

    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError("engine bind host must be an explicit IP address") from exc
    if address.is_unspecified or address.is_multicast or not (
        address.is_private or address.is_loopback or address.is_link_local
    ):
        raise ValueError("engine protocol must bind a private, loopback, or link-local address")
    return str(address)


@dataclass(slots=True)
class ManagedEngineProcess:
    deployment_id: str
    role: str
    command: tuple[str, ...]
    process: subprocess.Popen[str]
    stdout_handle: TextIO
    stderr_handle: TextIO
    started_unix_ns: int

    def stop(self, *, timeout_seconds: float = 30.0) -> int:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.stdout_handle.close()
        self.stderr_handle.close()
        return int(self.process.returncode or 0)


class EngineProcessManager:
    """Associates every native process with one deployment for bounded cleanup."""

    def __init__(self, log_directory: Path) -> None:
        self.log_directory = log_directory.expanduser().resolve()
        self.log_directory.mkdir(parents=True, exist_ok=True)
        self._processes: dict[tuple[str, str], ManagedEngineProcess] = {}
        self._lock = threading.RLock()

    def start(
        self,
        *,
        deployment_id: str,
        role: str,
        executable: Path,
        expected_sha256: str,
        arguments: tuple[str, ...],
        environment: dict[str, str] | None = None,
        ready: Callable[[subprocess.Popen[str]], None] | None = None,
    ) -> ManagedEngineProcess:
        resolved = executable.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        actual = sha256_file(resolved)
        if actual.lower() != expected_sha256.removeprefix("sha256:").lower():
            raise RuntimeError(f"engine binary hash mismatch: {resolved}")
        key = (deployment_id, role)
        with self._lock:
            existing = self._processes.get(key)
            if existing is not None and existing.process.poll() is None:
                return existing
            stdout_path = self.log_directory / f"{deployment_id}-{role}.stdout.log"
            stderr_path = self.log_directory / f"{deployment_id}-{role}.stderr.log"
            stdout = stdout_path.open("a", encoding="utf-8")
            stderr = stderr_path.open("a", encoding="utf-8")
            creationflags = 0
            if os.name == "nt":
                creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                process = subprocess.Popen(
                    [str(resolved), *arguments],
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    shell=False,
                    env={**os.environ, **(environment or {})},
                    creationflags=creationflags,
                )
                managed = ManagedEngineProcess(
                    deployment_id=deployment_id,
                    role=role,
                    command=(str(resolved), *arguments),
                    process=process,
                    stdout_handle=stdout,
                    stderr_handle=stderr,
                    started_unix_ns=time.time_ns(),
                )
                if ready is not None:
                    ready(process)
                if process.poll() is not None:
                    raise RuntimeError(
                        f"engine process {role} exited during startup with {process.returncode}"
                    )
                self._processes[key] = managed
                return managed
            except Exception:
                if "process" in locals() and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                stdout.close()
                stderr.close()
                raise

    def stop(self, deployment_id: str, role: str) -> int | None:
        with self._lock:
            process = self._processes.pop((deployment_id, role), None)
        return None if process is None else process.stop()

    def stop_deployment(self, deployment_id: str) -> dict[str, int]:
        with self._lock:
            roles = [role for selected, role in self._processes if selected == deployment_id]
        return {
            role: int(code or 0)
            for role in roles
            if (code := self.stop(deployment_id, role)) is not None
        }


__all__ = [
    "EngineProcessManager",
    "ManagedEngineProcess",
    "require_private_bind_host",
    "sha256_file",
]
