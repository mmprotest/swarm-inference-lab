"""Interval calendars for deterministic compute and shaped-network resources."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Any

from swarm_inference.experiments.experiment_022.models import Inventory

from .models import NetworkMode


@dataclass(frozen=True, slots=True)
class ReservedInterval:
    start_ms: float
    finish_ms: float
    metadata: dict[str, Any]

    @property
    def duration_ms(self) -> float:
        return self.finish_ms - self.start_ms


class ResourceCalendar:
    """Sorted non-overlapping reservations for one exclusive resource."""

    __slots__ = ("_intervals", "_parent", "_starts", "resource_id")

    def __init__(
        self,
        resource_id: str,
        intervals: tuple[ReservedInterval, ...] = (),
        *,
        parent: ResourceCalendar | None = None,
    ) -> None:
        if not resource_id:
            raise ValueError("resource calendar requires an identifier")
        self.resource_id = resource_id
        self._intervals = list(intervals)
        self._starts = [value.start_ms for value in intervals]
        self._parent = parent

    @property
    def intervals(self) -> tuple[ReservedInterval, ...]:
        if self._parent is None:
            return tuple(self._intervals)
        return tuple(
            sorted(
                (*self._parent.intervals, *self._intervals),
                key=lambda value: (value.start_ms, value.finish_ms),
            )
        )

    @property
    def latest_finish_ms(self) -> float:
        local = self._intervals[-1].finish_ms if self._intervals else 0.0
        parent = self._parent.latest_finish_ms if self._parent is not None else 0.0
        return max(local, parent)

    def clone(self) -> ResourceCalendar:
        return ResourceCalendar(self.resource_id, self.intervals)

    def overlay(self) -> ResourceCalendar:
        """Return a cheap speculative calendar layered over this calendar.

        Mask enumeration creates many short-lived what-if schedules.  Copying a
        long steady-state interval history into every mask is both unnecessary
        and prohibitively expensive; the overlay keeps only new reservations
        while querying the immutable parent history.
        """

        return ResourceCalendar(self.resource_id, parent=self)

    def _earliest_local_start(
        self, earliest_dependency_time: float, duration: float
    ) -> float:
        candidate = earliest_dependency_time
        index = bisect.bisect_right(self._starts, candidate) - 1
        if index >= 0 and self._intervals[index].finish_ms > candidate:
            candidate = self._intervals[index].finish_ms
        index += 1
        for interval in self._intervals[index:]:
            if interval.finish_ms <= candidate:
                continue
            if candidate + duration <= interval.start_ms:
                return candidate
            candidate = interval.finish_ms
        return candidate

    def earliest_start(self, earliest_dependency_time: float, duration: float) -> float:
        if not math.isfinite(earliest_dependency_time) or earliest_dependency_time < 0:
            raise ValueError("earliest dependency time must be finite and non-negative")
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("resource duration must be finite and non-negative")
        candidate = earliest_dependency_time
        while True:
            local_start = self._earliest_local_start(candidate, duration)
            parent_start = (
                self._parent.earliest_start(candidate, duration)
                if self._parent is not None
                else candidate
            )
            updated = max(local_start, parent_start)
            if updated == candidate:
                return candidate
            candidate = updated

    def reserve(
        self,
        start: float,
        duration: float,
        metadata: dict[str, Any],
    ) -> ReservedInterval:
        if not math.isfinite(start) or start < 0:
            raise ValueError("reservation start must be finite and non-negative")
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("reservation duration must be finite and non-negative")
        finish = start + duration
        if (
            self._parent is not None
            and self._parent.earliest_start(start, duration) != start
        ):
            raise ValueError(f"resource overlap on parent {self.resource_id}")
        index = bisect.bisect_left(self._starts, start)
        if index and self._intervals[index - 1].finish_ms > start:
            raise ValueError(f"resource overlap on {self.resource_id}")
        if index < len(self._intervals) and finish > self._intervals[index].start_ms:
            raise ValueError(f"resource overlap on {self.resource_id}")
        interval = ReservedInterval(start, finish, dict(metadata))
        self._intervals.insert(index, interval)
        self._starts.insert(index, start)
        return interval

    def busy_between(self, start_ms: float, finish_ms: float) -> float:
        if finish_ms < start_ms:
            raise ValueError("busy window finish precedes start")
        local = sum(
            max(0.0, min(finish_ms, row.finish_ms) - max(start_ms, row.start_ms))
            for row in self._intervals
        )
        parent = (
            self._parent.busy_between(start_ms, finish_ms)
            if self._parent is not None
            else 0.0
        )
        return parent + local


@dataclass(frozen=True, slots=True)
class MultiResourceReservation:
    resource_ids: tuple[str, ...]
    start_ms: float
    finish_ms: float
    duration_ms: float
    queue_wait_ms: float


class ResourceCalendars:
    """A lazily forkable set of concrete interval calendars."""

    __slots__ = ("_calendars", "_parent")

    def __init__(self, parent: ResourceCalendars | None = None) -> None:
        self._parent = parent
        self._calendars: dict[str, ResourceCalendar] = {}

    def calendar(self, resource_id: str) -> ResourceCalendar:
        value = self._calendars.get(resource_id)
        if value is not None:
            return value
        value = (
            self._parent.calendar(resource_id).overlay()
            if self._parent is not None
            else ResourceCalendar(resource_id)
        )
        self._calendars[resource_id] = value
        return value

    def fork(self) -> ResourceCalendars:
        return ResourceCalendars(parent=self)

    def resource_ids(self) -> tuple[str, ...]:
        values = set(self._calendars)
        if self._parent is not None:
            values.update(self._parent.resource_ids())
        return tuple(sorted(values))

    def earliest_common_start(
        self,
        resource_ids: tuple[str, ...],
        earliest_ms: float,
        duration_ms: float,
    ) -> float:
        resources = tuple(dict.fromkeys(resource_ids))
        if not resources:
            return earliest_ms
        candidate = earliest_ms
        while True:
            starts = tuple(
                self.calendar(resource).earliest_start(candidate, duration_ms)
                for resource in resources
            )
            updated = max(starts)
            if updated == candidate:
                return candidate
            candidate = updated

    def reserve(
        self,
        resource_ids: tuple[str, ...],
        *,
        earliest_ms: float,
        duration_ms: float,
        metadata: dict[str, Any],
    ) -> MultiResourceReservation:
        resources = tuple(dict.fromkeys(resource_ids))
        if not resources:
            raise ValueError("reservation requires at least one concrete resource")
        start = self.earliest_common_start(resources, earliest_ms, duration_ms)
        for resource in resources:
            self.calendar(resource).reserve(start, duration_ms, metadata)
        return MultiResourceReservation(
            resource_ids=resources,
            start_ms=start,
            finish_ms=start + duration_ms,
            duration_ms=duration_ms,
            queue_wait_ms=start - earliest_ms,
        )

    def reserve_append_order(
        self,
        resource_ids: tuple[str, ...],
        *,
        earliest_ms: float,
        duration_ms: float,
        metadata: dict[str, Any],
    ) -> MultiResourceReservation:
        """Append in task-evaluation order for exact E022 legacy replay."""

        resources = tuple(dict.fromkeys(resource_ids))
        if not resources:
            raise ValueError("reservation requires at least one concrete resource")
        start = max(
            earliest_ms,
            *(self.calendar(resource).latest_finish_ms for resource in resources),
        )
        for resource in resources:
            self.calendar(resource).reserve(start, duration_ms, metadata)
        return MultiResourceReservation(
            resource_ids=resources,
            start_ms=start,
            finish_ms=start + duration_ms,
            duration_ms=duration_ms,
            queue_wait_ms=start - earliest_ms,
        )


def transfer_resource_ids(
    mode: NetworkMode,
    source: str,
    destination: str,
) -> tuple[str, ...]:
    if source == destination:
        raise ValueError("same-node movement is not a network transfer")
    link = f"link:{source}->{destination}"
    if mode is NetworkMode.LEGACY_DIRECTED_LINK:
        return (link,)
    if mode is NetworkMode.SHARED_NIC:
        return (link, f"nic_tx:{source}", f"nic_rx:{destination}")
    raise ValueError(f"unknown network mode {mode}")


def reserve_transfer(
    calendars: ResourceCalendars,
    inventory: Inventory,
    mode: NetworkMode,
    *,
    source: str,
    destination: str,
    earliest_ms: float,
    payload_bytes: int,
    metadata: dict[str, Any],
) -> MultiResourceReservation:
    if payload_bytes <= 0:
        raise ValueError("network transfer requires positive payload bytes")
    nodes = inventory.node_map()
    duration = nodes[source].peer(destination).transfer_ms(payload_bytes)
    return calendars.reserve(
        transfer_resource_ids(mode, source, destination),
        earliest_ms=earliest_ms,
        duration_ms=duration,
        metadata=metadata,
    )


__all__ = [
    "MultiResourceReservation",
    "ReservedInterval",
    "ResourceCalendar",
    "ResourceCalendars",
    "reserve_transfer",
    "transfer_resource_ids",
]
