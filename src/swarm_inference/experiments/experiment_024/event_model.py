"""Deterministic concrete-resource scheduler used by E024.

Tasks wait on explicit dependency completion and on one exclusive concrete
compute or directed-link resource.  The scheduler advances by completion time,
so ready work from independent decode streams can overlap without depending on
task-name ordering.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

Category = Literal["compute", "network"]


@dataclass(frozen=True, slots=True)
class EventTask:
    resource_id: str
    dependency_ids: tuple[int, ...]
    duration_ms: float
    category: Category
    measured: bool
    operation: str
    node_id: str | None = None
    source_node_id: str | None = None
    destination_node_id: str | None = None
    payload_bytes: int = 0
    layer_id: int | None = None

    def __post_init__(self) -> None:
        if not self.resource_id or self.duration_ms < 0:
            raise ValueError("event tasks require a resource and non-negative service")
        if self.category == "compute" and self.node_id is None:
            raise ValueError("compute must belong to a concrete node")
        if self.category == "network":
            if self.source_node_id is None or self.destination_node_id is None:
                raise ValueError("network events require concrete endpoints")
            if self.source_node_id == self.destination_node_id:
                raise ValueError("same-node dependencies are not network events")
            if self.payload_bytes <= 0:
                raise ValueError("network events require a positive payload")


class TaskGraph:
    def __init__(self) -> None:
        self.tasks: list[EventTask] = []

    def add(self, task: EventTask) -> int:
        identifier = len(self.tasks)
        if any(value < 0 or value >= identifier for value in task.dependency_ids):
            raise ValueError("task dependencies must refer to earlier tasks")
        self.tasks.append(task)
        return identifier


@dataclass(frozen=True, slots=True)
class EventRun:
    makespan_ms: float
    start_ms: tuple[float, ...]
    finish_ms: tuple[float, ...]
    dependency_ready_ms: tuple[float, ...]
    measured_compute_ms: float
    measured_network_ms: float
    measured_network_bytes: int
    measured_network_messages: int
    measured_compute_queue_wait_ms: float
    measured_network_queue_wait_ms: float
    measured_compute_ms_by_node: dict[str, float]
    measured_network_ms_by_link: dict[str, float]
    measured_compute_ms_by_layer: dict[int, float]

    def task_finish(self, task_id: int) -> float:
        return self.finish_ms[task_id]


class DeterministicEventScheduler:
    """Run a finite task DAG with FIFO-by-readiness resource queues."""

    def run(self, graph: TaskGraph) -> EventRun:
        tasks = graph.tasks
        if not tasks:
            raise ValueError("event graph cannot be empty")
        count = len(tasks)
        remaining = [len(task.dependency_ids) for task in tasks]
        successors: list[list[int]] = [[] for _ in tasks]
        for identifier, task in enumerate(tasks):
            for dependency in task.dependency_ids:
                successors[dependency].append(identifier)

        resource_queues: dict[str, list[tuple[float, int]]] = defaultdict(list)
        busy_resources: set[str] = set()
        completions: list[tuple[float, int]] = []
        start = [-1.0] * count
        finish = [-1.0] * count
        ready_at = [-1.0] * count
        completed = 0

        def enqueue(identifier: int, value: float) -> None:
            ready_at[identifier] = value
            heapq.heappush(
                resource_queues[tasks[identifier].resource_id],
                (value, identifier),
            )

        def start_resource(resource: str, now: float) -> None:
            if resource in busy_resources or not resource_queues[resource]:
                return
            available_at, identifier = heapq.heappop(resource_queues[resource])
            begun = max(now, available_at)
            start[identifier] = begun
            finish[identifier] = begun + tasks[identifier].duration_ms
            busy_resources.add(resource)
            heapq.heappush(completions, (finish[identifier], identifier))

        initial_resources: set[str] = set()
        for identifier, value in enumerate(remaining):
            if value == 0:
                enqueue(identifier, 0.0)
                initial_resources.add(tasks[identifier].resource_id)
        for resource in sorted(initial_resources):
            start_resource(resource, 0.0)

        while completions:
            now = completions[0][0]
            finishing: list[int] = []
            while completions and completions[0][0] == now:
                _, identifier = heapq.heappop(completions)
                finishing.append(identifier)
            affected_resources: set[str] = set()
            for identifier in sorted(finishing):
                resource = tasks[identifier].resource_id
                busy_resources.remove(resource)
                affected_resources.add(resource)
                completed += 1
            for identifier in sorted(finishing):
                for successor in successors[identifier]:
                    remaining[successor] -= 1
                    if remaining[successor] == 0:
                        dependency_ready = max(
                            (finish[value] for value in tasks[successor].dependency_ids),
                            default=0.0,
                        )
                        enqueue(successor, dependency_ready)
                        affected_resources.add(tasks[successor].resource_id)
            for resource in sorted(affected_resources):
                start_resource(resource, now)

        if completed != count:
            raise ValueError("event task graph contains a cycle or unscheduled resource")

        compute_by_node: dict[str, float] = defaultdict(float)
        network_by_link: dict[str, float] = defaultdict(float)
        compute_by_layer: dict[int, float] = defaultdict(float)
        measured_compute = 0.0
        measured_network = 0.0
        network_bytes = 0
        network_messages = 0
        compute_queue = 0.0
        network_queue = 0.0
        for identifier, task in enumerate(tasks):
            if not task.measured:
                continue
            queue_wait = max(0.0, start[identifier] - ready_at[identifier])
            if task.category == "compute":
                measured_compute += task.duration_ms
                compute_queue += queue_wait
                compute_by_node[str(task.node_id)] += task.duration_ms
                if task.layer_id is not None:
                    compute_by_layer[task.layer_id] += task.duration_ms
            else:
                measured_network += task.duration_ms
                network_queue += queue_wait
                network_bytes += task.payload_bytes
                network_messages += 1
                network_by_link[task.resource_id] += task.duration_ms
        return EventRun(
            makespan_ms=max(finish),
            start_ms=tuple(start),
            finish_ms=tuple(finish),
            dependency_ready_ms=tuple(ready_at),
            measured_compute_ms=measured_compute,
            measured_network_ms=measured_network,
            measured_network_bytes=network_bytes,
            measured_network_messages=network_messages,
            measured_compute_queue_wait_ms=compute_queue,
            measured_network_queue_wait_ms=network_queue,
            measured_compute_ms_by_node=dict(sorted(compute_by_node.items())),
            measured_network_ms_by_link=dict(sorted(network_by_link.items())),
            measured_compute_ms_by_layer=dict(sorted(compute_by_layer.items())),
        )


__all__ = [
    "DeterministicEventScheduler",
    "EventRun",
    "EventTask",
    "TaskGraph",
]
