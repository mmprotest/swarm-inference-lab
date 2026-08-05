"""Bounded ownership and cleanup for multi-process product tests."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import queue
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any
from uuid import uuid4


class ChildStartupError(RuntimeError):
    """A managed child reported an error or exited before readiness."""


class ProcessCleanupError(RuntimeError):
    """A managed child required force or exited unexpectedly."""


@dataclass(slots=True)
class ProcessEvent:
    """Spawn-safe polling event backed by semaphore-free shared memory.

    Product tests only need a one-way, idempotent shutdown latch.  A raw byte
    gives that contract without the semaphore set allocated by
    ``multiprocessing.Event``.  The parent owns the value; children only read
    it, and the cluster drops its ownership after every child has joined.
    """

    _value: Any

    def is_set(self) -> bool:
        return bool(self._value.value)

    def set(self) -> None:
        self._value.value = 1

    def clear(self) -> None:
        self._value.value = 0

    def wait(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.is_set():
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        return True


@dataclass(slots=True)
class ManagedProcess:
    """Own one child from start through graceful stop, kill fallback, and close."""

    role: str
    process: BaseProcess
    shutdown: Callable[[], None] | None = None
    graceful_timeout_s: float = 10.0
    terminate_timeout_s: float = 5.0
    _closed: bool = field(default=False, init=False)
    _shutdown_requested: bool = field(default=False, init=False)
    _expected_exit_reason: str | None = field(default=None, init=False)
    final_pid: int | None = field(default=None, init=False)
    final_exitcode: int | None = field(default=None, init=False)
    graceful_shutdown_count: int = field(default=0, init=False)
    unexpected_terminate_count: int = field(default=0, init=False)
    unexpected_kill_count: int = field(default=0, init=False)
    expected_terminate_count: int = field(default=0, init=False)
    expected_kill_count: int = field(default=0, init=False)

    def start(self) -> None:
        if self._closed:
            raise RuntimeError(f"managed process {self.role!r} is closed")
        if self.process.pid is not None:
            raise RuntimeError(f"managed process {self.role!r} was already started")
        self.process.start()

    def wait_ready(self, readiness_queue: Any, timeout: float) -> Any:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ChildStartupError(
                    f"managed child {self.role!r} did not become ready within {timeout:.1f}s; "
                    f"pid={self.process.pid} exitcode={self.process.exitcode}"
                )
            try:
                payload = readiness_queue.get(timeout=min(remaining, 0.1))
            except queue.Empty:
                if self.process.exitcode is not None:
                    raise ChildStartupError(
                        f"managed child {self.role!r} exited before readiness; "
                        f"pid={self.process.pid} exitcode={self.process.exitcode}"
                    ) from None
                continue
            if isinstance(payload, dict) and payload.get("error"):
                raise ChildStartupError(
                    f"managed child {self.role!r} startup failed: {payload['error']}"
                )
            return payload

    def request_shutdown(self) -> None:
        if self.shutdown is not None and not self._shutdown_requested:
            self._shutdown_requested = True
            self.shutdown()

    def expect_exit(self, reason: str) -> None:
        if not reason.strip():
            raise ValueError("expected process exit requires a reason")
        self._expected_exit_reason = reason

    def crash(self, *, reason: str, timeout: float | None = None) -> None:
        """Intentionally terminate one explicitly identified fault target."""

        if self._closed:
            raise RuntimeError(f"managed process {self.role!r} is closed")
        self.expect_exit(reason)
        if self.process.pid is None or self.process.exitcode is not None:
            return
        self.expected_terminate_count += 1
        self.process.terminate()
        self.process.join(self.terminate_timeout_s if timeout is None else timeout)
        if self.process.is_alive():
            self.expected_kill_count += 1
            self.process.kill()
            self.process.join(self.terminate_timeout_s)
        if self.process.is_alive():
            raise ProcessCleanupError(
                f"expected crash target {self.role!r} survived kill; pid={self.process.pid}"
            )

    def _force_cleanup(self) -> None:
        self.unexpected_terminate_count += 1
        self.process.terminate()
        self.process.join(self.terminate_timeout_s)
        if self.process.is_alive():
            self.unexpected_kill_count += 1
            self.process.kill()
            self.process.join(self.terminate_timeout_s)
        if self.process.is_alive():
            raise ProcessCleanupError(
                f"managed child {self.role!r} survived kill; pid={self.process.pid}"
            )

    def close(self) -> None:
        if self._closed:
            return
        errors: list[str] = []
        try:
            if self.process.pid is not None:
                self.final_pid = self.process.pid
                if self.process.exitcode is None:
                    try:
                        self.request_shutdown()
                    except BaseException as exc:
                        errors.append(f"shutdown request failed: {type(exc).__name__}: {exc}")
                    self.process.join(self.graceful_timeout_s)
                if self.process.is_alive():
                    self._force_cleanup()
                    errors.append(
                        "graceful shutdown timed out and force cleanup was required "
                        f"after {self.graceful_timeout_s:.1f}s"
                    )
                elif self._shutdown_requested and self.process.exitcode == 0:
                    self.graceful_shutdown_count += 1
                self.process.join(timeout=0)
                self.final_exitcode = self.process.exitcode
                if self.final_exitcode not in {0, None} and self._expected_exit_reason is None:
                    errors.append(f"unexpected exit code {self.final_exitcode}")
                self.process.close()
        finally:
            self._closed = True
        if errors:
            raise ProcessCleanupError(
                f"managed child {self.role!r} cleanup failed; pid={self.final_pid}; "
                + "; ".join(errors)
            )

    def cleanup_record(self) -> dict[str, object]:
        return {
            "role": self.role,
            "pid": self.final_pid,
            "exit_code": self.final_exitcode,
            "expected_exit_reason": self._expected_exit_reason,
            "graceful_shutdown_count": self.graceful_shutdown_count,
            "unexpected_terminate_count": self.unexpected_terminate_count,
            "unexpected_kill_count": self.unexpected_kill_count,
            "expected_terminate_count": self.expected_terminate_count,
            "expected_kill_count": self.expected_kill_count,
        }


class ProductCluster:
    """Own all child, IPC, and asynchronous resources created by a process test."""

    def __init__(self, *, context: BaseContext | None = None) -> None:
        self.context: Any = context or multiprocessing.get_context("spawn")
        self.cluster_id = uuid4().hex
        self.processes: dict[str, ManagedProcess] = {}
        self._queues: list[Any] = []
        self._events: list[ProcessEvent] = []
        self._async_closers: list[tuple[str, Callable[[], Any]]] = []
        self._closed = False

    def queue(self, *, maxsize: int = 0) -> Any:
        managed_queue = self.context.Queue(maxsize=maxsize)
        self._queues.append(managed_queue)
        return managed_queue

    def event(self) -> ProcessEvent:
        event = ProcessEvent(self.context.RawValue("b", 0))
        self._events.append(event)
        return event

    def process(
        self,
        role: str,
        *,
        target: Callable[..., object],
        args: Sequence[object] = (),
        shutdown: Callable[[], None] | None = None,
    ) -> BaseProcess:
        if role in self.processes:
            raise ValueError(f"duplicate managed process role {role!r}")
        raw: BaseProcess = self.context.Process(
            target=target,
            args=tuple(args),
            name=f"swarm-test:{role}",
        )
        self.processes[role] = ManagedProcess(role=role, process=raw, shutdown=shutdown)
        return raw

    def track_process(
        self,
        role: str,
        process: BaseProcess,
        *,
        shutdown: Callable[[], None] | None = None,
    ) -> ManagedProcess:
        if role in self.processes:
            raise ValueError(f"duplicate managed process role {role!r}")
        managed = ManagedProcess(role=role, process=process, shutdown=shutdown)
        self.processes[role] = managed
        return managed

    def track_async_closer(self, role: str, closer: Callable[[], Any]) -> None:
        self._async_closers.append((role, closer))

    def expect_process_exit(self, role: str, *, reason: str) -> None:
        try:
            managed = self.processes[role]
        except KeyError as exc:
            raise KeyError(f"unknown managed process role {role!r}") from exc
        managed.expect_exit(reason)

    def crash_process(
        self,
        role: str,
        *,
        reason: str,
        timeout: float | None = None,
    ) -> None:
        try:
            managed = self.processes[role]
        except KeyError as exc:
            raise KeyError(f"unknown managed process role {role!r}") from exc
        managed.crash(reason=reason, timeout=timeout)

    def start(self) -> None:
        started: list[ManagedProcess] = []
        try:
            for managed in self.processes.values():
                managed.start()
                started.append(managed)
        except BaseException:
            for managed in started:
                with suppress(BaseException):
                    managed.request_shutdown()
            for managed in reversed(started):
                with suppress(ProcessCleanupError):
                    managed.close()
            raise

    async def wait_ready(self, readiness_queue: Any, *, count: int, timeout: float) -> list[Any]:
        deadline = time.monotonic() + timeout
        payloads: list[Any] = []
        while len(payloads) < count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                states = ", ".join(
                    f"{role}=pid:{item.process.pid}/exit:{item.process.exitcode}"
                    for role, item in self.processes.items()
                )
                raise ChildStartupError(
                    f"only {len(payloads)}/{count} children became ready within {timeout:.1f}s; "
                    f"{states}"
                )
            try:
                payload = await asyncio.to_thread(
                    readiness_queue.get,
                    True,
                    min(remaining, 0.25),
                )
            except queue.Empty:
                failed = [
                    (role, item.process.exitcode)
                    for role, item in self.processes.items()
                    if item.process.pid is not None and item.process.exitcode is not None
                ]
                if failed:
                    for role, _exit_code in failed:
                        self.processes[role].expect_exit("failed during startup")
                    raise ChildStartupError(f"children exited before readiness: {failed}") from None
                continue
            if isinstance(payload, dict) and payload.get("error"):
                worker_id = payload.get("worker_id")
                for role, managed in self.processes.items():
                    if role == worker_id or role.endswith(str(worker_id)):
                        managed.expect_exit("reported startup failure")
                raise ChildStartupError(f"child startup failed: {payload}")
            payloads.append(payload)
        return payloads

    async def close(self) -> None:
        if self._closed:
            return
        errors: list[str] = []
        # Parent-side clients must release calls and channels before child
        # services begin their graceful drain.
        for role, closer in reversed(self._async_closers):
            try:
                result = closer()
                if result is not None:
                    await result
            except BaseException as exc:
                errors.append(f"{role} async cleanup: {exc}")
        self._async_closers.clear()
        # Signal every child first so shutdown can proceed concurrently.
        for managed in self.processes.values():
            try:
                managed.request_shutdown()
            except BaseException as exc:
                errors.append(f"{managed.role} shutdown signal: {exc}")
        for managed in reversed(tuple(self.processes.values())):
            try:
                await asyncio.to_thread(managed.close)
            except BaseException as exc:
                errors.append(f"{managed.role} process cleanup: {exc}")
        for managed_queue in reversed(self._queues):
            try:
                managed_queue.close()
                await asyncio.to_thread(managed_queue.join_thread)
            except BaseException as exc:
                errors.append(f"multiprocessing queue cleanup: {exc}")
        self._queues.clear()
        self._events.clear()
        self._closed = True
        self._write_lifecycle_records()
        if errors:
            detail = "; ".join(errors)
            counts = self.lifecycle_counts()
            if counts["unexpected_terminate_count"] or counts["unexpected_kill_count"]:
                raise ProcessCleanupError(detail)
            raise RuntimeError(detail)

    def _write_lifecycle_records(self) -> None:
        configured = os.environ.get("SWARM_PROCESS_LIFECYCLE_LOG")
        if not configured:
            return
        path = Path(configured).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "cluster_id": self.cluster_id,
            "processes": [managed.cleanup_record() for managed in self.processes.values()],
            **self.lifecycle_counts(),
        }
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")

    def lifecycle_counts(self) -> dict[str, int]:
        return {
            "graceful_shutdown_count": sum(
                item.graceful_shutdown_count for item in self.processes.values()
            ),
            "unexpected_terminate_count": sum(
                item.unexpected_terminate_count for item in self.processes.values()
            ),
            "unexpected_kill_count": sum(
                item.unexpected_kill_count for item in self.processes.values()
            ),
            "expected_terminate_count": sum(
                item.expected_terminate_count for item in self.processes.values()
            ),
            "expected_kill_count": sum(
                item.expected_kill_count for item in self.processes.values()
            ),
            "leaked_process_count": len(self.live_children()),
        }

    def live_children(self) -> list[dict[str, object]]:
        return [
            {
                "role": role,
                "pid": managed.process.pid,
                "exitcode": managed.process.exitcode,
            }
            for role, managed in self.processes.items()
            if not managed._closed
            and managed.process.pid is not None
            and managed.process.is_alive()
        ]

    def queue_threads(self) -> list[str]:
        return [
            thread.name
            for thread in threading.enumerate()
            if thread.name == "QueueFeederThread" and thread.is_alive()
        ]

    async def __aenter__(self) -> ProductCluster:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()


__all__ = [
    "ChildStartupError",
    "ManagedProcess",
    "ProcessCleanupError",
    "ProcessEvent",
    "ProductCluster",
]
