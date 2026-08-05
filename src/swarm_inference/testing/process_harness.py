"""Bounded ownership and cleanup for multi-process product tests."""

from __future__ import annotations

import asyncio
import multiprocessing
import queue
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from typing import Any


class ChildStartupError(RuntimeError):
    """A managed child reported an error or exited before readiness."""


@dataclass(slots=True)
class ManagedProcess:
    """Own one child from start through graceful stop, kill fallback, and close."""

    role: str
    process: BaseProcess
    shutdown: Callable[[], None] | None = None
    graceful_timeout_s: float = 10.0
    terminate_timeout_s: float = 5.0
    _closed: bool = field(default=False, init=False)

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
        if self.shutdown is not None:
            self.shutdown()

    def terminate(self, timeout: float | None = None) -> None:
        if self.process.pid is None or self.process.exitcode is not None:
            return
        self.request_shutdown()
        self.process.join(self.graceful_timeout_s if timeout is None else timeout)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(self.terminate_timeout_s)
        if self.process.is_alive():
            self.kill()

    def kill(self) -> None:
        if self.process.pid is None or not self.process.is_alive():
            return
        self.process.kill()
        self.process.join(self.terminate_timeout_s)
        if self.process.is_alive():
            raise RuntimeError(f"managed child {self.role!r} survived kill; pid={self.process.pid}")

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.terminate()
            if self.process.pid is not None:
                self.process.join(timeout=0)
                if self.process.is_alive():
                    raise RuntimeError(
                        f"managed child {self.role!r} remains alive; pid={self.process.pid}"
                    )
                self.process.close()
        finally:
            self._closed = True


class ProductCluster:
    """Own all child, IPC, and asynchronous resources created by a process test."""

    def __init__(self, *, context: BaseContext | None = None) -> None:
        self.context: Any = context or multiprocessing.get_context("spawn")
        self.processes: dict[str, ManagedProcess] = {}
        self._queues: list[Any] = []
        self._async_closers: list[tuple[str, Callable[[], Any]]] = []
        self._closed = False

    def queue(self, *, maxsize: int = 0) -> Any:
        managed_queue = self.context.Queue(maxsize=maxsize)
        self._queues.append(managed_queue)
        return managed_queue

    def event(self) -> Any:
        return self.context.Event()

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

    def start(self) -> None:
        started: list[ManagedProcess] = []
        try:
            for managed in self.processes.values():
                managed.start()
                started.append(managed)
        except BaseException:
            for managed in reversed(started):
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
                    raise ChildStartupError(f"children exited before readiness: {failed}") from None
                continue
            if isinstance(payload, dict) and payload.get("error"):
                raise ChildStartupError(f"child startup failed: {payload}")
            payloads.append(payload)
        return payloads

    async def close(self) -> None:
        if self._closed:
            return
        errors: list[str] = []
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
        for role, closer in reversed(self._async_closers):
            try:
                result = closer()
                if result is not None:
                    await result
            except BaseException as exc:
                errors.append(f"{role} async cleanup: {exc}")
        for managed_queue in reversed(self._queues):
            try:
                managed_queue.close()
                await asyncio.to_thread(managed_queue.join_thread)
            except BaseException as exc:
                errors.append(f"multiprocessing queue cleanup: {exc}")
        self._closed = True
        if errors:
            raise RuntimeError("; ".join(errors))

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


__all__ = ["ChildStartupError", "ManagedProcess", "ProductCluster"]
