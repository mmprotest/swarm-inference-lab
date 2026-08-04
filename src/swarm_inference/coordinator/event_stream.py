"""Bounded, strictly ordered per-request event streams."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from swarm_inference.exceptions import BackpressureError
from swarm_inference.protocol.messages import StreamEventType, SubmitStreamEvent


class BoundedRequestEventStream:
    """A bounded queue with an out-of-band terminal record for safe failure."""

    def __init__(self, *, request_id: str, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("event stream capacity must be positive")
        self.request_id = request_id
        self.capacity = capacity
        self._queue: asyncio.Queue[SubmitStreamEvent] = asyncio.Queue(maxsize=capacity)
        self._wake = asyncio.Event()
        self._next_sequence = 0
        self._terminal: SubmitStreamEvent | None = None
        self._closed = False
        self.backpressure_failures = 0

    @property
    def qsize(self) -> int:
        return self._queue.qsize()

    @property
    def closed(self) -> bool:
        return self._closed and self._queue.empty() and self._terminal is None

    def _event(self, event_type: StreamEventType, **values: Any) -> SubmitStreamEvent:
        return SubmitStreamEvent(
            event_type=event_type,
            request_id=self.request_id,
            sequence_number=self._next_sequence,
            monotonic_timestamp_ns=time.monotonic_ns(),
            **values,
        )

    def publish(self, event_type: StreamEventType, **values: Any) -> SubmitStreamEvent:
        if self._closed or self._terminal is not None:
            raise RuntimeError("request event stream is already terminal")
        event = self._event(event_type, **values)
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull as exc:
            self.backpressure_failures += 1
            self._terminal = self._event(
                StreamEventType.REQUEST_FAILED,
                status_detail="bounded client event queue exhausted",
            )
            self._next_sequence += 1
            self._closed = True
            self._wake.set()
            raise BackpressureError("bounded client event queue exhausted") from exc
        self._next_sequence += 1
        self._wake.set()
        return event

    def finish(self, event_type: StreamEventType, **values: Any) -> SubmitStreamEvent:
        event = self.publish(event_type, **values)
        self._closed = True
        self._wake.set()
        return event

    def fail(
        self,
        detail: str,
        *,
        cancelled: bool = False,
        session_id: str | None = None,
        topology_id: str | None = None,
        model_revision: str | None = None,
    ) -> None:
        if self._terminal is not None or self._closed:
            return
        event_type = (
            StreamEventType.REQUEST_CANCELLED if cancelled else StreamEventType.REQUEST_FAILED
        )
        terminal = self._event(
            event_type,
            status_detail=detail,
            session_id=session_id,
            topology_id=topology_id,
            model_revision=model_revision,
        )
        self._next_sequence += 1
        try:
            self._queue.put_nowait(terminal)
        except asyncio.QueueFull:
            self._terminal = terminal
        self._closed = True
        self._wake.set()

    def __aiter__(self) -> BoundedRequestEventStream:
        return self

    async def __anext__(self) -> SubmitStreamEvent:
        while True:
            if not self._queue.empty():
                event = self._queue.get_nowait()
                self._queue.task_done()
                return event
            if self._terminal is not None:
                terminal = self._terminal
                self._terminal = None
                return terminal
            if self._closed:
                raise StopAsyncIteration
            self._wake.clear()
            if not self._queue.empty() or self._terminal is not None or self._closed:
                continue
            await self._wake.wait()


__all__ = ["BoundedRequestEventStream"]
