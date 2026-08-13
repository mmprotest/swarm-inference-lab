"""Concrete-worker deterministic event engine shared by all E022 planners."""

from __future__ import annotations

import heapq
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EventTask:
    task_id: str
    resource_id: str
    dependency_ids: tuple[str, ...]
    duration_ms: float
    category: str
    node_id: str | None = None
    source_node_id: str | None = None
    destination_node_id: str | None = None
    payload_bytes: int = 0
    layer_id: int | None = None
    chunk_id: int | None = None
    operation: str = ""

    def __post_init__(self) -> None:
        if not self.task_id or not self.resource_id or self.duration_ms < 0:
            raise ValueError("event task requires an id, resource, and non-negative duration")
        if self.category == "compute" and not self.node_id:
            raise ValueError("compute service must belong to a concrete node")
        if self.category == "network":
            if not self.source_node_id or not self.destination_node_id:
                raise ValueError("network service requires concrete endpoints")
            if self.source_node_id == self.destination_node_id:
                raise ValueError("same-node dependencies are not network tasks")
            if self.payload_bytes <= 0:
                raise ValueError("network tasks require positive payload bytes")


@dataclass(frozen=True, slots=True)
class EventRecord:
    task_id: str
    resource_id: str
    start_ms: float
    finish_ms: float
    duration_ms: float
    category: str
    dependency_ids: tuple[str, ...]
    critical_predecessor: str | None
    node_id: str | None
    source_node_id: str | None
    destination_node_id: str | None
    payload_bytes: int
    layer_id: int | None
    chunk_id: int | None
    operation: str


@dataclass(frozen=True, slots=True)
class EventRun:
    makespan_ms: float
    total_compute_ms: float
    total_network_ms: float
    total_network_bytes: int
    messages: int
    serial_waits: int
    worker_utilization: dict[str, float]
    critical_path_task_ids: tuple[str, ...]
    critical_path_compute_ms: float
    critical_path_network_ms: float
    records: tuple[EventRecord, ...]

    def as_dict(self, *, include_records: bool = False) -> dict[str, Any]:
        value = {
            "schema_version": "experiment-022-worker-event-run-v1",
            "makespan_ms": self.makespan_ms,
            "total_compute_ms": self.total_compute_ms,
            "total_network_ms": self.total_network_ms,
            "total_network_bytes": self.total_network_bytes,
            "messages": self.messages,
            "serial_waits": self.serial_waits,
            "worker_utilization": self.worker_utilization,
            "critical_path_task_ids": list(self.critical_path_task_ids),
            "critical_path_compute_ms": self.critical_path_compute_ms,
            "critical_path_network_ms": self.critical_path_network_ms,
            "task_count": len(self.records),
        }
        if include_records:
            value["records"] = [asdict(record) for record in self.records]
        return value


class DeterministicWorkerEventEngine:
    """Schedule a DAG on exclusive concrete node/link resources."""

    def run(self, tasks: Iterable[EventTask]) -> EventRun:
        values = list(tasks)
        pending = {task.task_id: task for task in values}
        if not pending or len(pending) != len(values):
            raise ValueError("event task identifiers must be non-empty and unique")
        known = set(pending)
        successors: dict[str, list[str]] = {identifier: [] for identifier in known}
        remaining: dict[str, int] = {}
        for task in values:
            missing = set(task.dependency_ids).difference(known)
            if missing:
                raise ValueError(f"task {task.task_id} has missing dependencies {missing}")
            remaining[task.task_id] = len(task.dependency_ids)
            for dependency in task.dependency_ids:
                successors[dependency].append(task.task_id)
        ready = [identifier for identifier, count in remaining.items() if count == 0]
        heapq.heapify(ready)
        resource_available: dict[str, float] = {}
        resource_last: dict[str, str] = {}
        completed: dict[str, EventRecord] = {}
        busy_by_node: dict[str, float] = {}
        while pending:
            if not ready:
                raise ValueError("event DAG contains a cycle")
            identifier = heapq.heappop(ready)
            task = pending.pop(identifier)
            predecessor_ids = list(task.dependency_ids)
            prior_resource = resource_last.get(task.resource_id)
            if prior_resource is not None:
                predecessor_ids.append(prior_resource)
            critical_predecessor = max(
                predecessor_ids,
                key=lambda item: (completed[item].finish_ms, item),
                default=None,
            )
            dependency_ready = (
                completed[critical_predecessor].finish_ms
                if critical_predecessor is not None
                else 0.0
            )
            start = max(dependency_ready, resource_available.get(task.resource_id, 0.0))
            finish = start + task.duration_ms
            record = EventRecord(
                task_id=task.task_id,
                resource_id=task.resource_id,
                start_ms=start,
                finish_ms=finish,
                duration_ms=task.duration_ms,
                category=task.category,
                dependency_ids=task.dependency_ids,
                critical_predecessor=critical_predecessor,
                node_id=task.node_id,
                source_node_id=task.source_node_id,
                destination_node_id=task.destination_node_id,
                payload_bytes=task.payload_bytes,
                layer_id=task.layer_id,
                chunk_id=task.chunk_id,
                operation=task.operation,
            )
            completed[identifier] = record
            resource_available[task.resource_id] = finish
            resource_last[task.resource_id] = identifier
            if task.category == "compute" and task.node_id:
                busy_by_node[task.node_id] = busy_by_node.get(task.node_id, 0.0) + task.duration_ms
            for successor in successors[identifier]:
                remaining[successor] -= 1
                if remaining[successor] == 0:
                    heapq.heappush(ready, successor)
        records = tuple(sorted(completed.values(), key=lambda row: (row.start_ms, row.task_id)))
        final = max(records, key=lambda row: (row.finish_ms, row.task_id))
        path: list[str] = []
        cursor: str | None = final.task_id
        while cursor is not None:
            path.append(cursor)
            cursor = completed[cursor].critical_predecessor
        path.reverse()
        critical_compute = sum(
            completed[task].duration_ms
            for task in path
            if completed[task].category == "compute"
        )
        critical_network = sum(
            completed[task].duration_ms
            for task in path
            if completed[task].category == "network"
        )
        makespan = final.finish_ms
        network_records = [record for record in records if record.category == "network"]
        return EventRun(
            makespan_ms=makespan,
            total_compute_ms=sum(
                record.duration_ms for record in records if record.category == "compute"
            ),
            total_network_ms=sum(record.duration_ms for record in network_records),
            total_network_bytes=sum(record.payload_bytes for record in network_records),
            messages=len(network_records),
            serial_waits=sum(
                bool(record.dependency_ids) and record.start_ms > 0 for record in records
            ),
            worker_utilization={
                node: busy / makespan if makespan else 0.0
                for node, busy in sorted(busy_by_node.items())
            },
            critical_path_task_ids=tuple(path),
            critical_path_compute_ms=critical_compute,
            critical_path_network_ms=critical_network,
            records=records,
        )


__all__ = [
    "DeterministicWorkerEventEngine",
    "EventRecord",
    "EventRun",
    "EventTask",
]
