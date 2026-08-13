"""Dependency-first wavefront scheduling and persistent microcell primitives.

The event engine in this module models independent resources.  It never treats
one physical GPU as concurrent microcells.  Physical service measurements are
inputs; overlap exists only in the deterministic independent-resource model.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import queue
import statistics
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from itertools import pairwise
from typing import Any

from swarm_inference.experiments.experiment_015.network import NetworkProfile

HIDDEN_SIZE = 7168
ATTNRES_SLOTS = 8
ATTNRES_SNAPSHOT_LAYERS = (0, 12, 24, 36, 48, 60, 72, 84)
MICROCELL_DEPTH = 8
MICROCELL_COUNT = 12
LAYER_COUNT = 93
OBJECT_ID_BYTES = 24
MESSAGE_METADATA_BYTES = 64


class TaskKind(StrEnum):
    COMPUTE = "compute"
    HANDOFF = "handoff"
    CACHE_SEED = "cache_seed"
    EXPERT = "expert"
    REDUCTION = "reduction"
    CONTROL = "control"


@dataclass(frozen=True, slots=True)
class VerificationBlockSpec:
    """One exact contiguous target block, including its accepted bonus row."""

    block_id: str
    request_id: str
    positions: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.block_id or not self.request_id or not self.positions:
            raise ValueError("verification block identity and positions are required")
        if self.positions[0] < 0:
            raise ValueError("verification positions must be non-negative")
        expected = tuple(range(self.positions[0], self.positions[0] + len(self.positions)))
        if self.positions != expected:
            raise ValueError("verification positions must be contiguous")

    @property
    def accepted_rows(self) -> int:
        return len(self.positions)


@dataclass(frozen=True, slots=True)
class WavefrontChunk:
    block_id: str
    chunk_id: int
    start_position: int
    end_position: int

    def __post_init__(self) -> None:
        if not self.block_id or self.chunk_id < 0:
            raise ValueError("wavefront chunk identity is invalid")
        if self.start_position < 0 or self.end_position <= self.start_position:
            raise ValueError("wavefront chunk position range is invalid")

    @property
    def row_count(self) -> int:
        return self.end_position - self.start_position

    @property
    def positions(self) -> tuple[int, ...]:
        return tuple(range(self.start_position, self.end_position))


@dataclass(frozen=True, slots=True)
class MicrocellTask:
    request_id: str
    block_id: str
    chunk_id: int
    microcell_id: int
    dependencies: tuple[str, ...]
    input_refs: tuple[str, ...]
    state_refs: tuple[str, ...]
    output_ref: str


@dataclass(frozen=True, slots=True)
class ExpertTask:
    request_id: str
    block_id: str
    chunk_id: int
    microcell_id: int
    layer: int
    expert: int
    shard: int
    route_weight: float
    dependencies: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReductionTask:
    task_id: str
    dependency_task_ids: tuple[str, ...]
    stable_order: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AttnResObject:
    request_id: str
    block_id: str
    chunk_id: int
    snapshot_layer: int
    content_hash: str
    version: int
    producer_microcell: int
    cached_locations: tuple[int, ...]
    size_bytes: int

    @property
    def object_id(self) -> str:
        return (
            f"attnres:{self.request_id}:{self.block_id}:{self.chunk_id}:"
            f"l{self.snapshot_layer}:v{self.version}:{self.content_hash[:16]}"
        )


@dataclass(frozen=True, slots=True)
class RequestScopedObject:
    object_type: str
    request_id: str
    model_scope: str
    version: int
    content_hash: str
    producer: str
    consumers: tuple[str, ...]
    size_bytes: int
    lifetime: str
    invalidator: str
    payload: bytes = field(repr=False, compare=False)

    @property
    def object_id(self) -> str:
        return (
            f"{self.object_type}:{self.request_id}:{self.model_scope}:"
            f"v{self.version}:{self.content_hash[:20]}"
        )


class StaleObjectError(RuntimeError):
    """Raised when a consumer asks for a superseded immutable object."""


class ImmutableObjectCache:
    """Bounded request-scoped exact object cache with explicit invalidation."""

    def __init__(self, *, maximum_bytes: int) -> None:
        if maximum_bytes <= 0:
            raise ValueError("object cache budget must be positive")
        self.maximum_bytes = maximum_bytes
        self._objects: dict[str, RequestScopedObject] = {}
        self._latest: dict[tuple[str, str, str], int] = {}
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self.invalidations = 0
        self.seed_bytes = 0

    @staticmethod
    def create(
        *,
        object_type: str,
        request_id: str,
        model_scope: str,
        version: int,
        producer: str,
        consumers: Sequence[str],
        lifetime: str,
        invalidator: str,
        payload: bytes,
    ) -> RequestScopedObject:
        if not object_type or not request_id or not model_scope or version < 0:
            raise ValueError("immutable object identity is invalid")
        digest = hashlib.sha256(payload).hexdigest()
        return RequestScopedObject(
            object_type=object_type,
            request_id=request_id,
            model_scope=model_scope,
            version=version,
            content_hash=digest,
            producer=producer,
            consumers=tuple(consumers),
            size_bytes=len(payload),
            lifetime=lifetime,
            invalidator=invalidator,
            payload=payload,
        )

    def seed(self, value: RequestScopedObject) -> str:
        identity = (value.object_type, value.request_id, value.model_scope)
        latest = self._latest.get(identity)
        if latest is not None and value.version < latest:
            raise StaleObjectError("cannot seed a stale immutable object version")
        if latest == value.version:
            same_version = [
                item
                for item in self._objects.values()
                if (
                    item.object_type,
                    item.request_id,
                    item.model_scope,
                    item.version,
                )
                == (*identity, value.version)
            ]
            if same_version and same_version[0].content_hash != value.content_hash:
                raise StaleObjectError(
                    "an immutable version cannot be reseeded with different content"
                )
        previous = self._objects.get(value.object_id)
        incremental = value.size_bytes - (previous.size_bytes if previous else 0)
        if self.bytes + incremental > self.maximum_bytes:
            raise MemoryError("immutable object cache budget exceeded")
        if latest is not None and value.version > latest:
            self.invalidate_scope(*identity)
        self._objects[value.object_id] = value
        self._latest[identity] = value.version
        self.bytes += incremental
        self.seed_bytes += max(0, incremental)
        return value.object_id

    def get(
        self,
        object_id: str,
        *,
        consumer: str,
        expected_version: int,
        expected_hash: str,
    ) -> RequestScopedObject:
        value = self._objects.get(object_id)
        if value is None:
            self.misses += 1
            raise KeyError(object_id)
        if value.version != expected_version or value.content_hash != expected_hash:
            self.misses += 1
            raise StaleObjectError("immutable object version or hash is stale")
        if consumer not in value.consumers:
            self.misses += 1
            raise PermissionError("consumer is outside immutable object scope")
        self.hits += 1
        return value

    def invalidate_scope(self, object_type: str, request_id: str, model_scope: str) -> int:
        doomed = [
            object_id
            for object_id, value in self._objects.items()
            if (
                value.object_type == object_type
                and value.request_id == request_id
                and value.model_scope == model_scope
            )
        ]
        for object_id in doomed:
            self.bytes -= self._objects.pop(object_id).size_bytes
        if doomed:
            self.invalidations += len(doomed)
        self._latest.pop((object_type, request_id, model_scope), None)
        return len(doomed)

    def cleanup_request(self, request_id: str) -> int:
        doomed = [
            object_id
            for object_id, value in self._objects.items()
            if value.request_id == request_id
        ]
        for object_id in doomed:
            value = self._objects.pop(object_id)
            self.bytes -= value.size_bytes
            self._latest.pop((value.object_type, value.request_id, value.model_scope), None)
        self.invalidations += len(doomed)
        return len(doomed)

    def validate_versions(self, expected: Mapping[str, int]) -> None:
        """Reject missing or superseded dependencies before worker execution."""

        for object_id, version in expected.items():
            value = self._objects.get(object_id)
            if value is None:
                self.misses += 1
                raise KeyError(object_id)
            if value.version != version:
                self.misses += 1
                raise StaleObjectError(
                    f"object {object_id!r} is version {value.version}, expected {version}"
                )
            self.hits += 1

    def snapshot(self) -> dict[str, int]:
        return {
            "objects": len(self._objects),
            "bytes": self.bytes,
            "maximum_bytes": self.maximum_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "invalidations": self.invalidations,
            "seed_bytes": self.seed_bytes,
        }


@dataclass(frozen=True, slots=True)
class EventTask:
    task_id: str
    kind: TaskKind
    resource_id: str
    duration_ms: float
    dependencies: tuple[str, ...] = ()
    priority: int = 50
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.task_id or not self.resource_id:
            raise ValueError("event task and resource identity are required")
        if self.duration_ms < 0 or not math.isfinite(self.duration_ms):
            raise ValueError("event task duration must be finite and non-negative")
        if self.task_id in self.dependencies:
            raise ValueError("event task cannot depend on itself")


@dataclass(frozen=True, slots=True)
class EventRecord:
    task_id: str
    kind: str
    resource_id: str
    ready_ms: float
    start_ms: float
    finish_ms: float
    duration_ms: float
    dependencies: tuple[str, ...]
    critical_predecessor: str | None
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class EventRun:
    records: tuple[EventRecord, ...]
    makespan_ms: float
    serial_sum_ms: float
    critical_path: tuple[str, ...]
    resource_utilization: Mapping[str, float]
    peak_concurrency: int
    scheduler_cpu_ms: float

    def record_map(self) -> dict[str, EventRecord]:
        return {record.task_id: record for record in self.records}

    def to_json(self) -> dict[str, Any]:
        return {
            "makespan_ms": self.makespan_ms,
            "serial_sum_ms": self.serial_sum_ms,
            "critical_path": list(self.critical_path),
            "resource_utilization": dict(self.resource_utilization),
            "peak_concurrency": self.peak_concurrency,
            "scheduler_cpu_ms": self.scheduler_cpu_ms,
            "records": [asdict(record) for record in self.records],
        }


class DeterministicEventEngine:
    """Greedy event-driven list scheduler with explicit resource contention."""

    def run(self, tasks: Iterable[EventTask]) -> EventRun:
        started = time.perf_counter_ns()
        task_list = list(tasks)
        by_id = {task.task_id: task for task in task_list}
        if len(by_id) != len(task_list):
            raise ValueError("event task IDs must be unique")
        missing = sorted(
            {
                dependency
                for task in task_list
                for dependency in task.dependencies
                if dependency not in by_id
            }
        )
        if missing:
            raise ValueError(f"event DAG contains missing dependencies: {missing[:8]}")
        successors: dict[str, list[str]] = defaultdict(list)
        indegree: dict[str, int] = {}
        for task in task_list:
            indegree[task.task_id] = len(task.dependencies)
            for dependency in task.dependencies:
                successors[dependency].append(task.task_id)

        ready_heap: list[tuple[float, int, str]] = []
        for task in task_list:
            if indegree[task.task_id] == 0:
                heapq.heappush(ready_heap, (0.0, task.priority, task.task_id))
        finish: dict[str, float] = {}
        resource_available: dict[str, float] = defaultdict(float)
        resource_predecessor: dict[str, str] = {}
        records: list[EventRecord] = []
        critical_predecessors: dict[str, str | None] = {}

        while ready_heap:
            dependency_ready, _priority, task_id = heapq.heappop(ready_heap)
            task = by_id[task_id]
            dependency_predecessor = None
            if task.dependencies:
                dependency_predecessor = max(
                    task.dependencies, key=lambda item: (finish[item], item)
                )
                dependency_ready = finish[dependency_predecessor]
            resource_ready = resource_available[task.resource_id]
            start_ms = max(dependency_ready, resource_ready)
            predecessor = dependency_predecessor
            if resource_ready > dependency_ready or (
                resource_ready == dependency_ready and task.resource_id in resource_predecessor
            ):
                predecessor = resource_predecessor.get(task.resource_id, predecessor)
            finish_ms = start_ms + task.duration_ms
            finish[task_id] = finish_ms
            resource_available[task.resource_id] = finish_ms
            resource_predecessor[task.resource_id] = task_id
            critical_predecessors[task_id] = predecessor
            records.append(
                EventRecord(
                    task_id=task.task_id,
                    kind=task.kind.value,
                    resource_id=task.resource_id,
                    ready_ms=dependency_ready,
                    start_ms=start_ms,
                    finish_ms=finish_ms,
                    duration_ms=task.duration_ms,
                    dependencies=task.dependencies,
                    critical_predecessor=predecessor,
                    metadata=dict(task.metadata),
                )
            )
            for successor in sorted(successors[task_id]):
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    successor_task = by_id[successor]
                    ready_at = max(finish[item] for item in successor_task.dependencies)
                    heapq.heappush(
                        ready_heap,
                        (ready_at, successor_task.priority, successor),
                    )

        if len(records) != len(task_list):
            blocked = sorted(task_id for task_id, degree in indegree.items() if degree)
            raise ValueError(f"event DAG contains a cycle: {blocked[:8]}")
        makespan = max(finish.values(), default=0.0)
        final_task = max(finish, key=lambda item: (finish[item], item), default=None)
        path: list[str] = []
        cursor = final_task
        while cursor is not None:
            path.append(cursor)
            cursor = critical_predecessors[cursor]
        path.reverse()
        busy: dict[str, float] = defaultdict(float)
        endpoints: list[tuple[float, int]] = []
        for record in records:
            busy[record.resource_id] += record.duration_ms
            endpoints.append((record.start_ms, 1))
            endpoints.append((record.finish_ms, -1))
        concurrency = 0
        peak = 0
        for _timestamp, delta in sorted(endpoints, key=lambda item: (item[0], item[1])):
            concurrency += delta
            peak = max(peak, concurrency)
        scheduler_ms = (time.perf_counter_ns() - started) / 1e6
        return EventRun(
            records=tuple(sorted(records, key=lambda item: (item.start_ms, item.task_id))),
            makespan_ms=makespan,
            serial_sum_ms=sum(task.duration_ms for task in task_list),
            critical_path=tuple(path),
            resource_utilization={
                resource: (duration / makespan if makespan else 0.0)
                for resource, duration in sorted(busy.items())
            },
            peak_concurrency=peak,
            scheduler_cpu_ms=scheduler_ms,
        )


@dataclass(frozen=True, slots=True)
class MicrocellServiceProfile:
    microcell_id: int
    layer_start: int
    layer_end: int
    compute_ms_by_rows: Mapping[int, float]
    cuda_ms_by_rows: Mapping[int, float]
    host_overhead_ms_by_rows: Mapping[int, float]
    phase_ms_by_rows: Mapping[int, Mapping[str, float]]
    resident_bytes: int
    bottleneck_operator: str

    def __post_init__(self) -> None:
        if not 0 <= self.microcell_id < MICROCELL_COUNT:
            raise ValueError("microcell ID is outside the fixed topology")
        if not 0 <= self.layer_start < self.layer_end <= LAYER_COUNT:
            raise ValueError("microcell layer range is invalid")
        if set(self.compute_ms_by_rows) != {1, 2, 4, 8}:
            raise ValueError("microcell service requires physical rows 1/2/4/8")
        if any(value <= 0 for value in self.compute_ms_by_rows.values()):
            raise ValueError("microcell service must be positive")

    @property
    def internal_boundaries(self) -> int:
        return max(0, self.layer_end - self.layer_start - 1)


@dataclass(frozen=True, slots=True)
class WavefrontNetworkAccounting:
    current_total_bytes: int
    current_attnres_bytes: int
    cached_total_bytes: int
    cache_seed_bytes: int
    cache_reference_bytes: int
    cached_attnres_total_bytes: int
    steady_state_attnres_reduction_fraction: float
    total_attnres_reduction_fraction: float
    internal_boundary_bytes: int
    coarse_boundary_bytes: int
    cache_hits: int
    cache_misses: int
    invalidations: int


@dataclass(frozen=True, slots=True)
class WavefrontResult:
    block_candidates: int
    accepted_rows: int
    chunk_size: int
    cache_enabled: bool
    total_ms: float
    oracle_tok_s_per_user: float
    fill_ms: float
    steady_stage_period_ms: float
    drain_ms: float
    stage_utilization: Mapping[int, float]
    steady_stage_utilization: Mapping[int, float]
    median_steady_utilization: float
    pipeline_efficiency: float
    ideal_balanced_pipeline_ms: float
    useful_parallelism: float
    critical_path_fraction: float
    slowest_stage: int
    max_chunks_in_flight: int
    messages: int
    bytes: int
    critical_path_communication_ms: float
    scheduler_overhead_ms: float
    simulation_cpu_ms: float
    serial_waits: int
    event_run: EventRun
    network: WavefrontNetworkAccounting

    def summary(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("event_run")
        return value


def partition_rows(total_rows: int, maximum_chunk_rows: int) -> tuple[int, ...]:
    """Partition into measured exact row sizes without interpolation."""

    if total_rows <= 0 or maximum_chunk_rows not in (1, 2, 4, 8):
        raise ValueError("row partition requires positive rows and chunk 1/2/4/8")
    chunks: list[int] = []
    remaining = total_rows
    sizes = tuple(value for value in (8, 4, 2, 1) if value <= maximum_chunk_rows)
    while remaining:
        chunk = next(value for value in sizes if value <= remaining)
        chunks.append(chunk)
        remaining -= chunk
    return tuple(chunks)


def make_chunks(block: VerificationBlockSpec, maximum_chunk_rows: int) -> tuple[WavefrontChunk, ...]:
    sizes = partition_rows(block.accepted_rows, maximum_chunk_rows)
    chunks: list[WavefrontChunk] = []
    start = block.positions[0]
    for chunk_id, rows in enumerate(sizes):
        chunks.append(
            WavefrontChunk(
                block_id=block.block_id,
                chunk_id=chunk_id,
                start_position=start,
                end_position=start + rows,
            )
        )
        start += rows
    return tuple(chunks)


def current_boundary_payload(rows: int) -> int:
    return rows * (ATTNRES_SLOTS + 1) * HIDDEN_SIZE * 4


def hidden_payload(rows: int, object_count: int) -> int:
    return rows * HIDDEN_SIZE * 4 + MESSAGE_METADATA_BYTES + object_count * OBJECT_ID_BYTES


def snapshot_producer_cell(layer: int) -> int:
    if layer not in ATTNRES_SNAPSHOT_LAYERS:
        raise ValueError("layer is not an AttnRes snapshot producer")
    return layer // MICROCELL_DEPTH


class WavefrontModel:
    """Build and execute the exact coarse token-by-depth event graph."""

    def __init__(
        self,
        profiles: Sequence[MicrocellServiceProfile],
        *,
        internal_network: NetworkProfile | None = None,
        coarse_network: NetworkProfile | None = None,
        control_setup_ms: float = 0.0,
    ) -> None:
        ordered = tuple(sorted(profiles, key=lambda item: item.microcell_id))
        if [item.microcell_id for item in ordered] != list(range(MICROCELL_COUNT)):
            raise ValueError("wavefront model requires all 12 fixed microcells")
        self.profiles = ordered
        self.internal_network = internal_network or NetworkProfile(
            "internal", rtt_ms=0.25, bandwidth_gbps=25.0
        )
        self.coarse_network = coarse_network or NetworkProfile(
            "coarse", rtt_ms=5.0, bandwidth_gbps=10.0
        )
        self.control_setup_ms = control_setup_ms
        self.engine = DeterministicEventEngine()

    @staticmethod
    def _compute_id(cell: int, chunk: int) -> str:
        return f"compute:c{cell:02d}:q{chunk:03d}"

    @staticmethod
    def _handoff_id(cell: int, chunk: int) -> str:
        return f"handoff:c{cell:02d}-c{cell + 1:02d}:q{chunk:03d}"

    @staticmethod
    def _seed_id(layer: int, chunk: int, link: int) -> str:
        return f"seed:l{layer:02d}:q{chunk:03d}:link{link:02d}"

    def _attnres_objects(
        self, block: VerificationBlockSpec, chunks: Sequence[WavefrontChunk]
    ) -> tuple[AttnResObject, ...]:
        objects: list[AttnResObject] = []
        for chunk in chunks:
            for layer in ATTNRES_SNAPSHOT_LAYERS:
                producer = snapshot_producer_cell(layer)
                digest = hashlib.sha256(
                    f"{block.request_id}:{block.block_id}:{chunk.chunk_id}:{layer}".encode()
                ).hexdigest()
                objects.append(
                    AttnResObject(
                        request_id=block.request_id,
                        block_id=block.block_id,
                        chunk_id=chunk.chunk_id,
                        snapshot_layer=layer,
                        content_hash=digest,
                        version=1,
                        producer_microcell=producer,
                        cached_locations=tuple(range(producer + 1, MICROCELL_COUNT)),
                        size_bytes=chunk.row_count * HIDDEN_SIZE * 4,
                    )
                )
        return tuple(objects)

    def build_tasks(
        self,
        block: VerificationBlockSpec,
        *,
        maximum_chunk_rows: int,
        cache_enabled: bool,
    ) -> tuple[tuple[EventTask, ...], tuple[WavefrontChunk, ...], WavefrontNetworkAccounting]:
        chunks = make_chunks(block, maximum_chunk_rows)
        objects = self._attnres_objects(block, chunks)
        object_map = {(item.chunk_id, item.snapshot_layer): item for item in objects}
        tasks: list[EventTask] = []
        if self.control_setup_ms:
            tasks.append(
                EventTask(
                    task_id="control:bulk-dispatch",
                    kind=TaskKind.CONTROL,
                    resource_id="coordinator",
                    duration_ms=self.control_setup_ms,
                    priority=0,
                )
            )
        control_dependency = ("control:bulk-dispatch",) if self.control_setup_ms else ()

        for chunk in chunks:
            for profile in self.profiles:
                cell = profile.microcell_id
                dependencies = list(control_dependency)
                if chunk.chunk_id:
                    dependencies.append(self._compute_id(cell, chunk.chunk_id - 1))
                if cell:
                    dependencies.append(self._handoff_id(cell - 1, chunk.chunk_id))
                required_layers = tuple(
                    layer
                    for layer in ATTNRES_SNAPSHOT_LAYERS
                    if snapshot_producer_cell(layer) < cell
                )
                if cache_enabled:
                    dependencies.extend(
                        self._seed_id(layer, chunk.chunk_id, cell - 1)
                        for layer in required_layers
                        if snapshot_producer_cell(layer) < cell - 1
                    )
                if cache_enabled:
                    internal_payloads = [
                        hidden_payload(
                            chunk.row_count,
                            sum(layer <= boundary for layer in ATTNRES_SNAPSHOT_LAYERS),
                        )
                        for boundary in range(profile.layer_start, profile.layer_end - 1)
                    ]
                else:
                    internal_payloads = [
                        current_boundary_payload(chunk.row_count)
                    ] * profile.internal_boundaries
                internal_ms = sum(
                    self.internal_network.service_ms(payload)
                    for payload in internal_payloads
                )
                duration = float(profile.compute_ms_by_rows[chunk.row_count]) + internal_ms
                tasks.append(
                    EventTask(
                        task_id=self._compute_id(cell, chunk.chunk_id),
                        kind=TaskKind.COMPUTE,
                        resource_id=f"microcell:{cell:02d}",
                        duration_ms=duration,
                        dependencies=tuple(sorted(set(dependencies))),
                        priority=30,
                        metadata={
                            "cell": cell,
                            "chunk": chunk.chunk_id,
                            "rows": chunk.row_count,
                            "compute_ms": float(profile.compute_ms_by_rows[chunk.row_count]),
                            "internal_communication_ms": internal_ms,
                            "internal_boundaries": profile.internal_boundaries,
                            "internal_payload_bytes": sum(internal_payloads),
                            "attnres_reference_count": sum(
                                sum(layer <= boundary for layer in ATTNRES_SNAPSHOT_LAYERS)
                                for boundary in range(
                                    profile.layer_start, profile.layer_end - 1
                                )
                            )
                            if cache_enabled
                            else 0,
                            "required_attnres_layers": list(required_layers),
                            "bottleneck_operator": profile.bottleneck_operator,
                        },
                    )
                )

        for chunk in chunks:
            for cell in range(MICROCELL_COUNT - 1):
                completed_layers = tuple(
                    layer
                    for layer in ATTNRES_SNAPSHOT_LAYERS
                    if snapshot_producer_cell(layer) <= cell
                )
                new_objects = tuple(
                    object_map[(chunk.chunk_id, layer)]
                    for layer in completed_layers
                    if snapshot_producer_cell(layer) == cell
                )
                payload = current_boundary_payload(chunk.row_count)
                if cache_enabled:
                    payload = hidden_payload(chunk.row_count, len(completed_layers)) + sum(
                        value.size_bytes for value in new_objects
                    )
                tasks.append(
                    EventTask(
                        task_id=self._handoff_id(cell, chunk.chunk_id),
                        kind=TaskKind.HANDOFF,
                        resource_id=f"coarse-link:{cell:02d}",
                        duration_ms=self.coarse_network.service_ms(payload),
                        dependencies=(self._compute_id(cell, chunk.chunk_id),),
                        priority=20,
                        metadata={
                            "source_cell": cell,
                            "target_cell": cell + 1,
                            "chunk": chunk.chunk_id,
                            "rows": chunk.row_count,
                            "payload_bytes": payload,
                            "attnres_reference_count": (
                                len(completed_layers) if cache_enabled else 0
                            ),
                            "seed_object_bytes": (
                                sum(value.size_bytes for value in new_objects)
                                if cache_enabled
                                else 0
                            ),
                        },
                    )
                )
        if cache_enabled:
            for chunk in chunks:
                for layer in ATTNRES_SNAPSHOT_LAYERS:
                    value = object_map[(chunk.chunk_id, layer)]
                    # The first seed hop is coalesced with the producer's hidden-state
                    # handoff.  Later hops may run ahead of the compute wavefront.
                    previous = self._handoff_id(
                        value.producer_microcell, chunk.chunk_id
                    )
                    for link in range(value.producer_microcell + 1, MICROCELL_COUNT - 1):
                        task_id = self._seed_id(layer, chunk.chunk_id, link)
                        tasks.append(
                            EventTask(
                                task_id=task_id,
                                kind=TaskKind.CACHE_SEED,
                                resource_id=f"coarse-link:{link:02d}",
                                duration_ms=self.coarse_network.service_ms(
                                    value.size_bytes + MESSAGE_METADATA_BYTES
                                ),
                                dependencies=(previous,),
                                priority=10,
                                metadata={
                                    "object_id": value.object_id,
                                    "snapshot_layer": layer,
                                    "chunk": chunk.chunk_id,
                                    "payload_bytes": value.size_bytes + MESSAGE_METADATA_BYTES,
                                    "target_cell": link + 1,
                                },
                            )
                        )
                        previous = task_id

        total_boundaries = LAYER_COUNT - 1
        current_attnres = block.accepted_rows * ATTNRES_SLOTS * HIDDEN_SIZE * 4 * total_boundaries
        current_total = block.accepted_rows * (ATTNRES_SLOTS + 1) * HIDDEN_SIZE * 4 * total_boundaries
        reference_count = sum(
            int(task.metadata.get("attnres_reference_count", 0))
            for task in tasks
        )
        reference_bytes = reference_count * OBJECT_ID_BYTES if cache_enabled else 0
        seed_bytes = (
            sum(
                int(task.metadata.get("seed_object_bytes", 0))
                for task in tasks
                if task.kind == TaskKind.HANDOFF
            )
            + sum(
                int(task.metadata.get("payload_bytes", 0))
                for task in tasks
                if task.kind == TaskKind.CACHE_SEED
            )
            if cache_enabled
            else 0
        )
        cached_attnres = seed_bytes + reference_bytes
        internal_bytes = sum(
            int(task.metadata.get("internal_payload_bytes", 0))
            for task in tasks
            if task.kind == TaskKind.COMPUTE
        )
        coarse_bytes = sum(
            int(task.metadata.get("payload_bytes", 0))
            for task in tasks
            if task.kind in {TaskKind.HANDOFF, TaskKind.CACHE_SEED}
        )
        cached_total = internal_bytes + coarse_bytes if cache_enabled else 0
        accounting = WavefrontNetworkAccounting(
            current_total_bytes=current_total,
            current_attnres_bytes=current_attnres,
            cached_total_bytes=cached_total,
            cache_seed_bytes=seed_bytes,
            cache_reference_bytes=reference_bytes,
            cached_attnres_total_bytes=cached_attnres,
            steady_state_attnres_reduction_fraction=(
                1.0 - reference_bytes / current_attnres if current_attnres else 0.0
            ),
            total_attnres_reduction_fraction=(
                1.0 - cached_attnres / current_attnres if current_attnres else 0.0
            ),
            internal_boundary_bytes=internal_bytes,
            coarse_boundary_bytes=coarse_bytes,
            cache_hits=(reference_count if cache_enabled else 0),
            cache_misses=0,
            invalidations=0,
        )
        return tuple(tasks), chunks, accounting

    def run(
        self,
        *,
        block_candidates: int,
        maximum_chunk_rows: int,
        cache_enabled: bool,
        request_id: str = "experiment-018",
    ) -> WavefrontResult:
        accepted = block_candidates + 1
        block = VerificationBlockSpec(
            block_id=f"block-{block_candidates}",
            request_id=request_id,
            positions=tuple(range(accepted)),
        )
        tasks, chunks, accounting = self.build_tasks(
            block,
            maximum_chunk_rows=maximum_chunk_rows,
            cache_enabled=cache_enabled,
        )
        event_run = self.engine.run(tasks)
        records = event_run.record_map()
        final_starts = [
            records[self._compute_id(MICROCELL_COUNT - 1, chunk.chunk_id)].start_ms
            for chunk in chunks
        ]
        fill_ms = final_starts[0]
        periods = [
            following - previous
            for previous, following in pairwise(final_starts)
        ]
        steady_period = statistics.median(periods) if periods else event_run.makespan_ms
        final_compute = records[self._compute_id(MICROCELL_COUNT - 1, chunks[-1].chunk_id)]
        drain_ms = event_run.makespan_ms - final_compute.start_ms
        stage_busy: dict[int, float] = defaultdict(float)
        stage_start: dict[int, float] = {}
        stage_finish: dict[int, float] = {}
        for record in event_run.records:
            if record.kind != TaskKind.COMPUTE.value:
                continue
            cell = int(record.metadata["cell"])
            stage_busy[cell] += record.duration_ms
            stage_start[cell] = min(stage_start.get(cell, record.start_ms), record.start_ms)
            stage_finish[cell] = max(stage_finish.get(cell, record.finish_ms), record.finish_ms)
        utilization = {
            cell: stage_busy[cell] / event_run.makespan_ms for cell in range(MICROCELL_COUNT)
        }
        steady_start = final_starts[0]
        steady_end = records[self._compute_id(0, chunks[-1].chunk_id)].finish_ms
        steady_utilization: dict[int, float] = {}
        if steady_end > steady_start:
            for cell in range(MICROCELL_COUNT):
                overlap = 0.0
                for record in event_run.records:
                    if (
                        record.kind == TaskKind.COMPUTE.value
                        and int(record.metadata["cell"]) == cell
                    ):
                        overlap += max(
                            0.0,
                            min(steady_end, record.finish_ms)
                            - max(steady_start, record.start_ms),
                        )
                steady_utilization[cell] = overlap / (steady_end - steady_start)
        else:
            steady_utilization = dict(utilization)
        mean_by_chunk = [
            statistics.fmean(
                records[self._compute_id(cell, chunk.chunk_id)].duration_ms
                for cell in range(MICROCELL_COUNT)
            )
            for chunk in chunks
        ]
        ideal = sum(mean_by_chunk) + (MICROCELL_COUNT - 1) * max(mean_by_chunk)
        compute_records = {
            record.task_id: record
            for record in event_run.records
            if record.kind == TaskKind.COMPUTE.value
        }
        total_compute = sum(
            float(record.metadata["compute_ms"]) for record in compute_records.values()
        )
        critical_compute = sum(
            float(records[task_id].metadata.get("compute_ms", 0.0))
            for task_id in event_run.critical_path
            if task_id in records
        )
        critical_communication = sum(
            records[task_id].duration_ms
            for task_id in event_run.critical_path
            if records[task_id].kind in {TaskKind.HANDOFF.value, TaskKind.CACHE_SEED.value}
        )
        messages = sum(
            task.kind in {TaskKind.HANDOFF, TaskKind.CACHE_SEED} for task in tasks
        )
        bytes_ = sum(
            int(task.metadata.get("payload_bytes", 0))
            for task in tasks
            if task.kind in {TaskKind.HANDOFF, TaskKind.CACHE_SEED}
        )
        slowest = max(
            range(MICROCELL_COUNT),
            key=lambda cell: sum(
                records[self._compute_id(cell, chunk.chunk_id)].duration_ms
                for chunk in chunks
            ),
        )
        serial_waits = sum(
            records[task_id].kind != TaskKind.CONTROL.value
            for task_id in event_run.critical_path
        )
        chunk_endpoints: list[tuple[float, int]] = []
        for chunk in chunks:
            chunk_endpoints.append(
                (records[self._compute_id(0, chunk.chunk_id)].start_ms, 1)
            )
            chunk_endpoints.append(
                (
                    records[
                        self._compute_id(MICROCELL_COUNT - 1, chunk.chunk_id)
                    ].finish_ms,
                    -1,
                )
            )
        chunks_in_flight = 0
        maximum_chunks_in_flight = 0
        for _timestamp, delta in sorted(
            chunk_endpoints, key=lambda item: (item[0], -item[1])
        ):
            chunks_in_flight += delta
            maximum_chunks_in_flight = max(
                maximum_chunks_in_flight, chunks_in_flight
            )
        return WavefrontResult(
            block_candidates=block_candidates,
            accepted_rows=accepted,
            chunk_size=maximum_chunk_rows,
            cache_enabled=cache_enabled,
            total_ms=event_run.makespan_ms,
            oracle_tok_s_per_user=accepted * 1000.0 / event_run.makespan_ms,
            fill_ms=fill_ms,
            steady_stage_period_ms=steady_period,
            drain_ms=drain_ms,
            stage_utilization=utilization,
            steady_stage_utilization=steady_utilization,
            median_steady_utilization=statistics.median(
                steady_utilization.values()
            ),
            pipeline_efficiency=ideal / event_run.makespan_ms,
            ideal_balanced_pipeline_ms=ideal,
            useful_parallelism=(total_compute / critical_compute if critical_compute else 1.0),
            critical_path_fraction=(
                event_run.makespan_ms / event_run.serial_sum_ms
                if event_run.serial_sum_ms
                else 1.0
            ),
            slowest_stage=slowest,
            max_chunks_in_flight=maximum_chunks_in_flight,
            messages=messages,
            bytes=bytes_,
            critical_path_communication_ms=critical_communication,
            scheduler_overhead_ms=self.control_setup_ms,
            simulation_cpu_ms=event_run.scheduler_cpu_ms,
            serial_waits=serial_waits,
            event_run=event_run,
            network=accounting,
        )


@dataclass(frozen=True, slots=True)
class WorkerEnvelope:
    task_id: str
    request_id: str
    sequence: int
    object_versions: Mapping[str, int]
    payload: Any = field(compare=False)


@dataclass(frozen=True, slots=True)
class WorkerResult:
    task_id: str
    request_id: str
    sequence: int
    status: str
    payload: Any = field(compare=False)


class PersistentMicrocellWorker:
    """One long-lived queue worker with idempotency and deterministic cleanup."""

    _STOP = object()

    def __init__(
        self,
        worker_id: str,
        handler: Callable[[WorkerEnvelope], Any],
        *,
        maximum_queue: int = 128,
        cache_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        if not worker_id or maximum_queue <= 0:
            raise ValueError("persistent worker identity and queue bound are required")
        self.worker_id = worker_id
        self.handler = handler
        self.input_queue: queue.Queue[WorkerEnvelope | object] = queue.Queue(maximum_queue)
        self.output_queue: queue.Queue[WorkerResult] = queue.Queue(maximum_queue)
        self.cache = ImmutableObjectCache(maximum_bytes=cache_bytes)
        self._thread: threading.Thread | None = None
        self._closed = False
        self._completed: dict[str, WorkerResult] = {}
        self._latest_sequence: dict[str, int] = defaultdict(lambda: -1)
        self.task_count = 0
        self.duplicate_count = 0
        self.failure_count = 0

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("closed persistent worker cannot restart")
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name=f"wavefront-{self.worker_id}",
            daemon=True,
        )
        self._thread.start()

    def submit(self, envelope: WorkerEnvelope, *, timeout: float = 5.0) -> None:
        if self._thread is None:
            raise RuntimeError("persistent worker is not started")
        self.input_queue.put(envelope, timeout=timeout)

    def take(self, *, timeout: float = 5.0) -> WorkerResult:
        return self.output_queue.get(timeout=timeout)

    def _run(self) -> None:
        while True:
            item = self.input_queue.get()
            if item is self._STOP:
                return
            assert isinstance(item, WorkerEnvelope)
            cached = self._completed.get(item.task_id)
            if cached is not None:
                self.duplicate_count += 1
                self.output_queue.put(cached)
                continue
            latest = self._latest_sequence[item.request_id]
            if item.sequence <= latest:
                result = WorkerResult(
                    task_id=item.task_id,
                    request_id=item.request_id,
                    sequence=item.sequence,
                    status="STALE",
                    payload=None,
                )
                self.output_queue.put(result)
                continue
            if item.sequence != latest + 1:
                result = WorkerResult(
                    task_id=item.task_id,
                    request_id=item.request_id,
                    sequence=item.sequence,
                    status="LOST_PREDECESSOR",
                    payload={"expected_sequence": latest + 1},
                )
                self.output_queue.put(result)
                continue
            try:
                self.cache.validate_versions(item.object_versions)
                payload = self.handler(item)
                result = WorkerResult(
                    task_id=item.task_id,
                    request_id=item.request_id,
                    sequence=item.sequence,
                    status="PASS",
                    payload=payload,
                )
                self._completed[item.task_id] = result
                self._latest_sequence[item.request_id] = item.sequence
                self.task_count += 1
            except (KeyError, StaleObjectError) as exc:
                self.failure_count += 1
                result = WorkerResult(
                    task_id=item.task_id,
                    request_id=item.request_id,
                    sequence=item.sequence,
                    status="STALE_OBJECT",
                    payload={"type": type(exc).__name__, "message": str(exc)},
                )
            except BaseException as exc:
                self.failure_count += 1
                self.cache.cleanup_request(item.request_id)
                result = WorkerResult(
                    task_id=item.task_id,
                    request_id=item.request_id,
                    sequence=item.sequence,
                    status="FAIL",
                    payload={"type": type(exc).__name__, "message": str(exc)},
                )
            self.output_queue.put(result)

    def cleanup_request(self, request_id: str) -> None:
        self.cache.cleanup_request(request_id)
        self._latest_sequence.pop(request_id, None)
        self._completed = {
            task_id: result
            for task_id, result in self._completed.items()
            if result.request_id != request_id
        }

    def close(self, *, timeout: float = 5.0) -> None:
        if self._closed:
            return
        self._closed = True
        if self._thread is not None:
            self.input_queue.put(self._STOP)
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                raise TimeoutError("persistent worker did not stop deterministically")
            self._thread = None
        for request_id in list(self._latest_sequence):
            self.cleanup_request(request_id)


class HierarchicalTaskBatcher:
    """Coalesce logical tasks without creating one coordinator wait per task."""

    def __init__(self, *, leaf_batch: int = 32) -> None:
        if leaf_batch <= 0:
            raise ValueError("leaf batch must be positive")
        self.leaf_batch = leaf_batch

    def plan(self, tasks: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
        worker_tasks: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for task in tasks:
            worker_tasks[str(task.get("microcell"))].append(task)
        return self.plan_hierarchical(worker_tasks)

    def plan_hierarchical(
        self,
        worker_tasks: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> dict[str, int | float]:
        """Measure local planning separately from the serial coordinator path."""

        if not worker_tasks or any(not tasks for tasks in worker_tasks.values()):
            raise ValueError("hierarchical planning requires non-empty worker task lists")
        logical_count = 0
        physical_batches = 0
        metadata_bytes = 0
        worker_cpu: list[float] = []
        local_queue_operations = 0
        worker_summaries: list[tuple[str, int]] = []
        for worker_id, tasks in sorted(worker_tasks.items()):
            local_started = time.perf_counter_ns()
            buckets: dict[tuple[Any, ...], int] = defaultdict(int)
            for task in tasks:
                key = (
                    task.get("operation"),
                    task.get("shape"),
                    task.get("dtype"),
                    task.get("expert"),
                    task.get("weight_shard"),
                    task.get("route_bucket"),
                )
                buckets[key] += 1
                metadata_bytes += len(
                    json.dumps(task, sort_keys=True, separators=(",", ":"))
                )
            worker_batches = sum(
                math.ceil(count / self.leaf_batch) for count in buckets.values()
            )
            worker_cpu.append((time.perf_counter_ns() - local_started) / 1e6)
            logical_count += len(tasks)
            physical_batches += worker_batches
            local_queue_operations += worker_batches * 2
            worker_summaries.append((worker_id, worker_batches))

        # Only compact worker summaries cross the coordinator hot path.
        coordinator_started = time.perf_counter_ns()
        hierarchy_levels = math.ceil(math.log2(max(1, len(worker_summaries))))
        serial_decisions = len(worker_summaries) + hierarchy_levels
        coordinator_metadata_bytes = sum(
            len(worker_id.encode("utf-8")) + 8
            for worker_id, _batch_count in worker_summaries
        )
        coordinator_ms = (time.perf_counter_ns() - coordinator_started) / 1e6
        return {
            "logical_task_count": logical_count,
            "physical_kernel_count": physical_batches,
            "coalescing_ratio": logical_count / max(1, physical_batches),
            "serial_scheduling_decisions": serial_decisions,
            "queue_operations": local_queue_operations,
            "metadata_bytes": metadata_bytes,
            "coordinator_metadata_bytes": coordinator_metadata_bytes,
            "task_creation": 0,
            "critical_path_waits": hierarchy_levels + 1,
            "coordinator_cpu_ms": coordinator_ms,
            "worker_local_cpu_total_ms": sum(worker_cpu),
            "worker_local_cpu_critical_ms": max(worker_cpu),
            "worker_count": len(worker_summaries),
        }

    def plan_compact(
        self,
        worker_buckets: Mapping[
            str, Mapping[tuple[Any, ...], tuple[int, int]]
        ],
    ) -> dict[str, int | float]:
        """Plan prebucketed persistent-worker work without central task expansion.

        Each bucket value is ``(logical_count, total_metadata_bytes)``.
        Route-local workers form these counters while grouping assignments; the serial
        coordinator receives only one physical-batch count per worker.
        """

        if not worker_buckets or any(not buckets for buckets in worker_buckets.values()):
            raise ValueError("compact planning requires non-empty worker buckets")
        logical_count = 0
        physical_batches = 0
        metadata_bytes = 0
        worker_cpu: list[float] = []
        worker_summaries: list[tuple[str, int]] = []
        for worker_id, buckets in sorted(worker_buckets.items()):
            local_started = time.perf_counter_ns()
            worker_batches = 0
            for count, bucket_metadata_bytes in buckets.values():
                if count <= 0 or bucket_metadata_bytes <= 0:
                    raise ValueError("compact bucket counts and sizes must be positive")
                logical_count += count
                metadata_bytes += bucket_metadata_bytes
                worker_batches += math.ceil(count / self.leaf_batch)
            physical_batches += worker_batches
            worker_summaries.append((worker_id, worker_batches))
            worker_cpu.append((time.perf_counter_ns() - local_started) / 1e6)

        coordinator_started = time.perf_counter_ns()
        hierarchy_levels = math.ceil(math.log2(max(1, len(worker_summaries))))
        serial_decisions = len(worker_summaries) + hierarchy_levels
        coordinator_metadata_bytes = sum(
            len(worker_id.encode("utf-8")) + 8
            for worker_id, _batch_count in worker_summaries
        )
        coordinator_ms = (time.perf_counter_ns() - coordinator_started) / 1e6
        return {
            "logical_task_count": logical_count,
            "physical_kernel_count": physical_batches,
            "coalescing_ratio": logical_count / max(1, physical_batches),
            "serial_scheduling_decisions": serial_decisions,
            "queue_operations": physical_batches * 2,
            "metadata_bytes": metadata_bytes,
            "coordinator_metadata_bytes": coordinator_metadata_bytes,
            "task_creation": 0,
            "critical_path_waits": hierarchy_levels + 1,
            "coordinator_cpu_ms": coordinator_ms,
            "worker_local_cpu_total_ms": sum(worker_cpu),
            "worker_local_cpu_critical_ms": max(worker_cpu),
            "worker_count": len(worker_summaries),
        }


def build_fine_expert_tasks(
    *,
    request_id: str,
    block_id: str,
    chunk_id: int,
    microcell_id: int,
    layer: int,
    expert_ids: Sequence[int],
    split_degree: int,
    shard_service_ms: float,
    partial_payload_bytes: int,
    internal_network: NetworkProfile,
) -> tuple[EventTask, ...]:
    """Expand one exact routed layer into expert-shard and stable tree tasks."""

    if split_degree < 1 or split_degree & (split_degree - 1):
        raise ValueError("fine expert split degree must be a power of two")
    tasks: list[EventTask] = []
    leaves: list[str] = []
    for expert in expert_ids:
        for shard in range(split_degree):
            task_id = f"expert:l{layer}:e{expert}:s{shard}:q{chunk_id}"
            leaves.append(task_id)
            tasks.append(
                EventTask(
                    task_id=task_id,
                    kind=TaskKind.EXPERT,
                    resource_id=(
                        f"microcell:{microcell_id}:layer:{layer}:"
                        f"expert:{expert}:shard:{shard}"
                    ),
                    duration_ms=shard_service_ms,
                    priority=30,
                    metadata={
                        "request_id": request_id,
                        "block_id": block_id,
                        "chunk": chunk_id,
                        "layer": layer,
                        "expert": expert,
                        "shard": shard,
                    },
                )
            )
    level = leaves
    depth = 0
    while len(level) > 1:
        following: list[str] = []
        for index in range(0, len(level), 2):
            left = level[index]
            if index + 1 == len(level):
                following.append(left)
                continue
            right = level[index + 1]
            task_id = f"reduce:l{layer}:q{chunk_id}:d{depth}:n{index // 2}"
            tasks.append(
                EventTask(
                    task_id=task_id,
                    kind=TaskKind.REDUCTION,
                    resource_id=(
                        f"microcell:{microcell_id}:reduction:d{depth}:n{index // 2}"
                    ),
                    duration_ms=internal_network.service_ms(partial_payload_bytes),
                    dependencies=(left, right),
                    priority=40,
                    metadata={
                        "stable_order": [left, right],
                        "payload_bytes": partial_payload_bytes,
                        "depth": depth,
                    },
                )
            )
            following.append(task_id)
        level = following
        depth += 1
    return tuple(tasks)


__all__ = [
    "ATTNRES_SNAPSHOT_LAYERS",
    "AttnResObject",
    "DeterministicEventEngine",
    "EventRun",
    "EventTask",
    "ExpertTask",
    "HierarchicalTaskBatcher",
    "ImmutableObjectCache",
    "MicrocellServiceProfile",
    "MicrocellTask",
    "PersistentMicrocellWorker",
    "ReductionTask",
    "RequestScopedObject",
    "StaleObjectError",
    "TaskKind",
    "VerificationBlockSpec",
    "WavefrontChunk",
    "WavefrontModel",
    "WavefrontNetworkAccounting",
    "WavefrontResult",
    "WorkerEnvelope",
    "WorkerResult",
    "build_fine_expert_tasks",
    "current_boundary_payload",
    "hidden_payload",
    "make_chunks",
    "partition_rows",
]
