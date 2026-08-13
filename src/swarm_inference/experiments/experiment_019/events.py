"""Deterministic resource-explicit event engine for Experiment 019.

Only :class:`MicroworkerTask` instances carry compute service.  Pods never
carry service time or capacity; they are labels used to select network links.
"""

from __future__ import annotations

import math
import heapq
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal


@dataclass(frozen=True, slots=True)
class NetworkProfile:
    name: str
    rtt_ms: float
    bandwidth_gbps: float
    software_overhead_ms: float

    def one_way_ms(self, payload_bytes: int) -> float:
        if payload_bytes < 0 or self.rtt_ms < 0 or self.bandwidth_gbps <= 0:
            raise ValueError("invalid network transfer geometry")
        serialization = payload_bytes * 8 / (self.bandwidth_gbps * 1_000_000)
        return self.rtt_ms / 2 + serialization + self.software_overhead_ms


@dataclass(frozen=True, slots=True)
class MicroworkerTask:
    task_id: str
    worker_id: str
    pod_id: str
    request_id: str
    block_id: str
    chunk_id: int
    layer_id: int
    operator: str
    shard_id: str
    dependency_ids: tuple[str, ...]
    input_refs: tuple[str, ...]
    state_refs: tuple[str, ...]
    service_time_ms: float
    resource_type: Literal["microworker"] = "microworker"

    def __post_init__(self) -> None:
        if not self.worker_id or not self.pod_id or self.service_time_ms < 0:
            raise ValueError("microworker tasks require a concrete worker and service")


@dataclass(frozen=True, slots=True)
class CommunicationEdge:
    source_worker_id: str
    destination_worker_id: str
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class NetworkTask:
    task_id: str
    request_id: str
    block_id: str
    chunk_id: int
    layer_id: int
    operator: str
    dependency_ids: tuple[str, ...]
    edges: tuple[CommunicationEdge, ...]
    profile: NetworkProfile
    collective_algorithm: str | None = None
    collective_participants: tuple[str, ...] = ()
    collective_steps: int = 1
    resource_type: Literal["network"] = "network"

    def __post_init__(self) -> None:
        if not self.edges or self.collective_steps < 1:
            raise ValueError("network tasks require explicit communication edges")
        if any(edge.source_worker_id == edge.destination_worker_id for edge in self.edges):
            raise ValueError("network edges require distinct source and destination workers")

    @property
    def service_time_ms(self) -> float:
        per_step = max(self.profile.one_way_ms(edge.payload_bytes) for edge in self.edges)
        return self.collective_steps * per_step

    @property
    def aggregate_wire_bytes(self) -> int:
        # ``edges`` enumerates every directed transfer in the algorithm.  The
        # latency model still has ``collective_steps`` sequential phases, but
        # multiplying the complete edge set by that depth would count each
        # tree edge once per level even though it is active in only one level.
        return sum(edge.payload_bytes for edge in self.edges)


Task = MicroworkerTask | NetworkTask


@dataclass(frozen=True, slots=True)
class EventRecord:
    task_id: str
    resource_type: str
    resource_id: str
    start_time: float
    finish_time: float
    dependency_ids: tuple[str, ...]
    worker_id: str | None
    pod_id: str | None
    request_id: str
    block_id: str
    chunk_id: int
    layer_id: int
    operator: str
    shard_id: str | None
    input_refs: tuple[str, ...]
    state_refs: tuple[str, ...]
    communication_edges: tuple[dict[str, Any], ...]
    collective_algorithm: str | None
    collective_participants: tuple[str, ...]
    payload_bytes: int
    collective_steps: int
    base_network_latency_ms: float
    serialization_ms: float
    software_overhead_ms: float
    network_profile: str | None

    @property
    def duration_ms(self) -> float:
        return self.finish_time - self.start_time


@dataclass(frozen=True, slots=True)
class EventRun:
    records: tuple[EventRecord, ...]
    makespan_ms: float
    total_compute_work_ms: float
    critical_path_compute_ms: float
    network_critical_path_ms: float
    worker_utilization: dict[str, float]
    worker_idle_ms: dict[str, float]
    total_network_bytes: int
    critical_path_task_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "experiment-019-microworker-event-trace-v1",
            "makespan_ms": self.makespan_ms,
            "total_compute_work_ms": self.total_compute_work_ms,
            "critical_path_compute_ms": self.critical_path_compute_ms,
            "network_critical_path_ms": self.network_critical_path_ms,
            "worker_utilization": self.worker_utilization,
            "worker_idle_ms": self.worker_idle_ms,
            "total_network_bytes": self.total_network_bytes,
            "critical_path_task_ids": self.critical_path_task_ids,
            "total_task_count": len(self.records),
            "records": [asdict(record) for record in self.records],
        }


class DeterministicMicroworkerEngine:
    """Schedule an explicit DAG against exclusive workers and directed links."""

    def run(self, tasks: Iterable[Task]) -> EventRun:
        task_values = list(tasks)
        pending = {task.task_id: task for task in task_values}
        if len(pending) == 0:
            raise ValueError("event run requires tasks")
        if len(pending) != len(task_values):
            raise ValueError("task identifiers must be unique")
        known = set(pending)
        successors: dict[str, list[str]] = {identifier: [] for identifier in known}
        remaining_dependencies: dict[str, int] = {}
        for task in pending.values():
            missing = set(task.dependency_ids).difference(known)
            if missing:
                raise ValueError(f"task {task.task_id} has missing dependencies {missing}")
            remaining_dependencies[task.task_id] = len(task.dependency_ids)
            for dependency in task.dependency_ids:
                successors[dependency].append(task.task_id)
        ready_heap = [
            identifier
            for identifier, count in remaining_dependencies.items()
            if count == 0
        ]
        heapq.heapify(ready_heap)
        completed: dict[str, EventRecord] = {}
        worker_available: dict[str, float] = {}
        worker_last_task: dict[str, str] = {}
        link_available: dict[tuple[str, str], float] = {}
        link_last_task: dict[tuple[str, str], str] = {}
        busy: dict[str, float] = {}
        critical_compute: dict[str, float] = {}
        critical_network: dict[str, float] = {}
        critical_predecessor: dict[str, str | None] = {}
        while pending:
            if not ready_heap:
                raise ValueError("event DAG contains a cycle")
            task_identifier = heapq.heappop(ready_heap)
            task = pending[task_identifier]
            candidate_predecessors = list(task.dependency_ids)
            if isinstance(task, MicroworkerTask):
                prior_worker_task = worker_last_task.get(task.worker_id)
                if prior_worker_task is not None:
                    candidate_predecessors.append(prior_worker_task)
                predecessor = max(
                    candidate_predecessors,
                    key=lambda item: (completed[item].finish_time, item),
                    default=None,
                )
                dependency_finish = (
                    completed[predecessor].finish_time if predecessor is not None else 0.0
                )
                start = max(dependency_finish, worker_available.get(task.worker_id, 0.0))
                finish = start + task.service_time_ms
                worker_available[task.worker_id] = finish
                worker_last_task[task.worker_id] = task.task_id
                busy[task.worker_id] = busy.get(task.worker_id, 0.0) + task.service_time_ms
                record = EventRecord(
                    task_id=task.task_id,
                    resource_type=task.resource_type,
                    resource_id=f"microworker:{task.worker_id}",
                    start_time=start,
                    finish_time=finish,
                    dependency_ids=task.dependency_ids,
                    worker_id=task.worker_id,
                    pod_id=task.pod_id,
                    request_id=task.request_id,
                    block_id=task.block_id,
                    chunk_id=task.chunk_id,
                    layer_id=task.layer_id,
                    operator=task.operator,
                    shard_id=task.shard_id,
                    input_refs=task.input_refs,
                    state_refs=task.state_refs,
                    communication_edges=(),
                    collective_algorithm=None,
                    collective_participants=(),
                    payload_bytes=0,
                    collective_steps=0,
                    base_network_latency_ms=0.0,
                    serialization_ms=0.0,
                    software_overhead_ms=0.0,
                    network_profile=None,
                )
                critical_compute[task.task_id] = (
                    critical_compute[predecessor] if predecessor is not None else 0.0
                ) + task.service_time_ms
                critical_network[task.task_id] = (
                    critical_network[predecessor] if predecessor is not None else 0.0
                )
            else:
                keys = [
                    (edge.source_worker_id, edge.destination_worker_id) for edge in task.edges
                ]
                candidate_predecessors.extend(
                    value
                    for key in keys
                    if (value := link_last_task.get(key)) is not None
                )
                predecessor = max(
                    candidate_predecessors,
                    key=lambda item: (completed[item].finish_time, item),
                    default=None,
                )
                dependency_finish = (
                    completed[predecessor].finish_time if predecessor is not None else 0.0
                )
                start = max(
                    dependency_finish,
                    max((link_available.get(key, 0.0) for key in keys), default=0.0),
                )
                finish = start + task.service_time_ms
                for key in keys:
                    link_available[key] = finish
                    link_last_task[key] = task.task_id
                record = EventRecord(
                    task_id=task.task_id,
                    resource_type=task.resource_type,
                    resource_id="network:" + ",".join(f"{a}->{b}" for a, b in keys),
                    start_time=start,
                    finish_time=finish,
                    dependency_ids=task.dependency_ids,
                    worker_id=None,
                    pod_id=None,
                    request_id=task.request_id,
                    block_id=task.block_id,
                    chunk_id=task.chunk_id,
                    layer_id=task.layer_id,
                    operator=task.operator,
                    shard_id=None,
                    input_refs=(),
                    state_refs=(),
                    communication_edges=tuple(asdict(edge) for edge in task.edges),
                    collective_algorithm=task.collective_algorithm,
                    collective_participants=task.collective_participants,
                    payload_bytes=task.aggregate_wire_bytes,
                    collective_steps=task.collective_steps,
                    base_network_latency_ms=(
                        task.collective_steps * task.profile.rtt_ms / 2
                    ),
                    serialization_ms=(
                        task.collective_steps
                        * max(edge.payload_bytes for edge in task.edges)
                        * 8
                        / (task.profile.bandwidth_gbps * 1_000_000)
                    ),
                    software_overhead_ms=(
                        task.collective_steps * task.profile.software_overhead_ms
                    ),
                    network_profile=task.profile.name,
                )
                critical_compute[task.task_id] = (
                    critical_compute[predecessor] if predecessor is not None else 0.0
                )
                critical_network[task.task_id] = (
                    critical_network[predecessor] if predecessor is not None else 0.0
                ) + task.service_time_ms
            critical_predecessor[task.task_id] = predecessor
            completed[task.task_id] = record
            del pending[task.task_id]
            for successor in successors[task.task_id]:
                remaining_dependencies[successor] -= 1
                if remaining_dependencies[successor] == 0:
                    heapq.heappush(ready_heap, successor)
        records = tuple(sorted(completed.values(), key=lambda item: (item.start_time, item.task_id)))
        makespan = max(record.finish_time for record in records)
        utilization = {worker: value / makespan for worker, value in sorted(busy.items())}
        idle = {worker: makespan - value for worker, value in sorted(busy.items())}
        terminal = max(completed, key=lambda item: completed[item].finish_time)
        critical_path: list[str] = []
        cursor: str | None = terminal
        while cursor is not None:
            critical_path.append(cursor)
            cursor = critical_predecessor[cursor]
        critical_path.reverse()
        return EventRun(
            records=records,
            makespan_ms=makespan,
            total_compute_work_ms=sum(busy.values()),
            critical_path_compute_ms=critical_compute[terminal],
            network_critical_path_ms=critical_network[terminal],
            worker_utilization=utilization,
            worker_idle_ms=idle,
            total_network_bytes=sum(
                record.payload_bytes for record in records if record.resource_type == "network"
            ),
            critical_path_task_ids=tuple(critical_path),
        )


def tree_collective(
    *,
    task_id: str,
    participants: tuple[str, ...],
    payload_bytes: int,
    dependency_ids: tuple[str, ...],
    profile: NetworkProfile,
    request_id: str,
    block_id: str,
    chunk_id: int,
    layer_id: int,
    operator: str,
    all_reduce: bool = True,
) -> NetworkTask:
    if len(participants) < 2:
        raise ValueError("a collective requires at least two participants")
    reduce_edges = tuple(
        CommunicationEdge(participants[index], participants[index // 2], payload_bytes)
        for index in range(1, len(participants))
    )
    edges = (
        reduce_edges
        + tuple(
            CommunicationEdge(edge.destination_worker_id, edge.source_worker_id, payload_bytes)
            for edge in reversed(reduce_edges)
        )
        if all_reduce
        else reduce_edges
    )
    steps = math.ceil(math.log2(len(participants))) * (2 if all_reduce else 1)
    return NetworkTask(
        task_id=task_id,
        request_id=request_id,
        block_id=block_id,
        chunk_id=chunk_id,
        layer_id=layer_id,
        operator=operator,
        dependency_ids=dependency_ids,
        edges=edges,
        profile=profile,
        collective_algorithm="binary_tree_allreduce" if all_reduce else "binary_tree_reduce",
        collective_participants=participants,
        collective_steps=steps,
    )


def assert_headline_trace(records: Iterable[EventRecord]) -> None:
    forbidden = {"microcell", "layer", "stage", "aggregate_gpu"}
    compute = [record for record in records if record.resource_type != "network"]
    if not compute:
        raise AssertionError("headline trace has no compute events")
    for record in compute:
        if record.resource_type != "microworker":
            raise AssertionError(f"compute task {record.task_id} is not a microworker")
        if not record.worker_id or record.resource_type in forbidden:
            raise AssertionError(f"compute task {record.task_id} has a forbidden resource")


__all__ = [
    "CommunicationEdge",
    "DeterministicMicroworkerEngine",
    "EventRecord",
    "EventRun",
    "MicroworkerTask",
    "NetworkProfile",
    "NetworkTask",
    "assert_headline_trace",
    "tree_collective",
]
