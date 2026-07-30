"""Monotonic deterministic event clock."""

from __future__ import annotations

import heapq
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(order=True, slots=True)
class ScheduledEvent:
    time_s: float
    sequence: int
    callback: Callable[[], None] = field(compare=False, repr=False)
    name: str = field(compare=False, default="")


class SimClock:
    """A discrete-event queue; callbacks execute at monotonic simulated time."""

    def __init__(self) -> None:
        self._now_s = 0.0
        self._next_sequence = 0
        self._queue: list[ScheduledEvent] = []

    @property
    def now_s(self) -> float:
        return self._now_s

    @property
    def pending_events(self) -> int:
        return len(self._queue)

    def schedule_at(
        self,
        time_s: float,
        callback: Callable[[], None],
        *,
        name: str = "",
    ) -> int:
        if time_s < self._now_s:
            raise ValueError(f"cannot schedule event at {time_s}; simulated time is {self._now_s}")
        sequence = self._next_sequence
        self._next_sequence += 1
        heapq.heappush(
            self._queue,
            ScheduledEvent(time_s=time_s, sequence=sequence, callback=callback, name=name),
        )
        return sequence

    def schedule_in(
        self,
        delay_s: float,
        callback: Callable[[], None],
        *,
        name: str = "",
    ) -> int:
        if delay_s < 0:
            raise ValueError("event delay cannot be negative")
        return self.schedule_at(self._now_s + delay_s, callback, name=name)

    def run(self, *, until_s: float | None = None, maximum_events: int | None = None) -> int:
        executed = 0
        while self._queue:
            if maximum_events is not None and executed >= maximum_events:
                break
            event = heapq.heappop(self._queue)
            if until_s is not None and event.time_s > until_s:
                heapq.heappush(self._queue, event)
                self._now_s = max(self._now_s, until_s)
                break
            if event.time_s < self._now_s:
                raise RuntimeError("event queue violated monotonic simulated time")
            self._now_s = event.time_s
            event.callback()
            executed += 1
        return executed

    def snapshot(self) -> list[tuple[float, int, str]]:
        return [(event.time_s, event.sequence, event.name) for event in sorted(self._queue)]
