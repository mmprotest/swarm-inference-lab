"""Deterministic closed-loop E023 multi-request serving model."""

from __future__ import annotations

import heapq
import math
import re
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from swarm_inference.experiments.experiment_022.evaluator import (
    LATENT,
    ROUTE_METADATA_BYTES_PER_ROW,
    EndpointPolicy,
    PlacementEvaluator,
    target_chunks,
)
from swarm_inference.experiments.experiment_022.event_model import EventTask
from swarm_inference.experiments.experiment_022.models import (
    Inventory,
    ModelGraph,
    PartitionKind,
)
from swarm_inference.experiments.experiment_022.service import ResidentServiceModel

from .models import E023Plan, NetworkMode
from .resource_calendar import ResourceCalendars, transfer_resource_ids
from .routing import GroupExecutionSpec, select_fork_join_routes

TARGET_ROWS = 17
CONCURRENCY_LEVELS = (1, 8, 32, 64, 128)


@dataclass(frozen=True, slots=True)
class ScheduledTaskRecord:
    slot_id: int
    pass_index: int
    local_task_id: str
    category: str
    operation: str
    layer_id: int | None
    chunk_id: int | None
    node_id: str | None
    source_node_id: str | None
    destination_node_id: str | None
    payload_bytes: int
    resource_ids: tuple[str, ...]
    dependency_ready_ms: float
    start_ms: float
    finish_ms: float
    duration_ms: float
    queue_wait_ms: float
    logical_group_id: int | None = None
    selected_alternate: bool = False


@dataclass(frozen=True, slots=True)
class ForkJoinRecord:
    slot_id: int
    pass_index: int
    layer_id: int
    chunk_id: int
    group_arrival_ms: dict[int, float]
    selected_nodes: dict[int, str]
    mask: int


@dataclass(frozen=True, slots=True)
class PassResult:
    slot_id: int
    pass_index: int
    start_ms: float
    finish_ms: float
    latency_ms: float
    record_start: int
    record_stop: int
    fork_join_records: tuple[ForkJoinRecord, ...]
    compute_busy_ms_by_node: dict[str, float]
    compute_queue_wait_ms_by_node: dict[str, float]
    compute_operation_count_by_node: dict[str, int]
    tx_busy_ms_by_node: dict[str, float]
    rx_busy_ms_by_node: dict[str, float]
    link_busy_ms: dict[str, float]
    worker_compute_ms: float
    network_bytes: int
    transfer_count: int
    network_queue_wait_ms: float


@dataclass(frozen=True, slots=True)
class ServingRun:
    status: str
    network_mode: NetworkMode
    concurrency: int
    target_passes_measured: int
    target_rows_measured: int
    measurement_window_ms: float
    target_rows_per_second: float
    p50_pass_latency_ms: float
    p95_pass_latency_ms: float
    t0_ms: float
    t_start_ms: float
    t1_ms: float
    network_bytes: int
    transfer_count: int
    worker_compute_ms: float
    compute_queue_wait_ms: float
    network_queue_wait_ms: float
    maximum_compute_utilization: float
    maximum_tx_utilization: float
    maximum_rx_utilization: float
    maximum_link_utilization: float
    compute_busy_ms_by_node: dict[str, float]
    compute_queue_wait_ms_by_node: dict[str, float]
    compute_operation_count_by_node: dict[str, int]
    tx_busy_ms_by_node: dict[str, float]
    rx_busy_ms_by_node: dict[str, float]
    link_busy_ms: dict[str, float]
    participating_workers: tuple[str, ...]
    selected_masks: tuple[int, ...]
    primary_selection_count: int
    alternate_selection_count: int
    passes: tuple[PassResult, ...]
    records: tuple[ScheduledTaskRecord, ...]
    insufficient_extra_pass_rounds: int

    @property
    def replica_selection_rate(self) -> float:
        total = self.primary_selection_count + self.alternate_selection_count
        return self.alternate_selection_count / total if total else 0.0


@dataclass(frozen=True, slots=True)
class SinglePassRun:
    critical_path_ms: float
    total_worker_compute_ms: float
    total_network_ms: float
    network_bytes: int
    messages: int
    event_count: int
    participating_workers: tuple[str, ...]
    records: tuple[ScheduledTaskRecord, ...]
    fork_join_records: tuple[ForkJoinRecord, ...]
    selected_masks: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _ForkSpec:
    marker_id: str
    dependency_id: str
    layer_id: int
    chunk_id: int
    rows: int
    coordinator_node_id: str
    primary_nodes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PreparedTemplate:
    ordered_tasks: tuple[EventTask, ...]
    fork_specs: dict[str, _ForkSpec]


def _topological_task_order(tasks: list[EventTask]) -> tuple[EventTask, ...]:
    """Compile the placement DAG's deterministic heap order once per plan."""

    by_id = {task.task_id: task for task in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("serving template task IDs are not unique")
    known = set(by_id)
    successors: dict[str, list[str]] = {identifier: [] for identifier in known}
    remaining: dict[str, int] = {}
    for task in tasks:
        missing = set(task.dependency_ids).difference(known)
        if missing:
            raise ValueError(f"serving task {task.task_id} has missing dependencies")
        remaining[task.task_id] = len(task.dependency_ids)
        for dependency in task.dependency_ids:
            successors[dependency].append(task.task_id)
    ready = [identifier for identifier, count in remaining.items() if count == 0]
    heapq.heapify(ready)
    ordered: list[EventTask] = []
    while ready:
        identifier = heapq.heappop(ready)
        ordered.append(by_id[identifier])
        for successor in successors[identifier]:
            remaining[successor] -= 1
            if remaining[successor] == 0:
                heapq.heappush(ready, successor)
    if len(ordered) != len(tasks):
        raise ValueError("serving template contains a cycle")
    return tuple(ordered)


def measured_passes_per_slot(concurrency: int) -> int:
    if concurrency not in CONCURRENCY_LEVELS:
        raise ValueError("E023 concurrency is outside the frozen ladder")
    return max(2, math.ceil(128 / concurrency))


def _eligible_passes(passes: list[PassResult]) -> tuple[float, list[PassResult]]:
    warmup_finishes = [row.finish_ms for row in passes if row.pass_index == 1]
    if not warmup_finishes:
        raise ValueError("closed-loop workload did not complete two warm-up passes")
    t0 = max(warmup_finishes)
    eligible = [
        row for row in passes if row.pass_index >= 2 and row.start_ms >= t0
    ]
    return t0, eligible


class ServingEngine:
    """Schedule exact E022 target-pass DAGs on persistent interval resources."""

    def __init__(
        self,
        model: ModelGraph,
        inventory: Inventory,
        service: ResidentServiceModel,
        endpoint: EndpointPolicy,
        *,
        network_mode: NetworkMode,
    ) -> None:
        self.model = model
        self.inventory = inventory
        self.service = service
        self.endpoint = endpoint
        self.network_mode = network_mode
        self.nodes = inventory.node_map()
        self.evaluator = PlacementEvaluator(model, inventory, service, endpoint)
        self._template_cache: dict[str, _PreparedTemplate] = {}

    def _prepare(self, plan: E023Plan) -> _PreparedTemplate:
        cached = self._template_cache.get(plan.canonical_sha256)
        if cached is not None:
            return cached
        tasks = list(self.evaluator.build_tasks(plan.base_plan))
        replicas_by_layer: dict[int, list[Any]] = {}
        for replica in plan.replicas:
            replicas_by_layer.setdefault(replica.layer_id, []).append(replica)
        fork_specs: dict[str, _ForkSpec] = {}
        if replicas_by_layer:
            assignments = {
                row.layer_id: row for row in plan.base_plan.assignments
            }
            remove_ids: set[str] = set()
            replacements: dict[str, EventTask] = {}
            markers: list[EventTask] = []
            for layer_id in sorted(replicas_by_layer):
                assignment = assignments[layer_id]
                if (
                    assignment.partition_kind is not PartitionKind.WHOLE_EXPERT
                    or assignment.degree != 8
                ):
                    raise ValueError("flexible routing requires WHOLE_EXPERT:p8")
                for chunk_id, rows in enumerate(target_chunks(plan.base_plan.chunk_rows)):
                    prefix = f"c{chunk_id:02d}.l{layer_id:02d}"
                    marker_id = f"{prefix}.expert-optionality-fork"
                    dependency_id = f"{prefix}.latent-down-whole"
                    pattern = re.compile(
                        rf"^{re.escape(prefix)}\.expert-\d{{2}}(?:\.worker-protocol)?$"
                    )
                    for task in tasks:
                        if (
                            task.task_id.startswith(
                                f"{prefix}.expert-latent-fanout."
                            )
                            or task.task_id.startswith(f"{prefix}.expert-gather.")
                            or pattern.match(task.task_id)
                        ):
                            remove_ids.add(task.task_id)
                    reduction_id = f"{prefix}.expert-reduction"
                    reduction = next(task for task in tasks if task.task_id == reduction_id)
                    replacements[reduction_id] = replace(
                        reduction, dependency_ids=(marker_id,)
                    )
                    marker = EventTask(
                        task_id=marker_id,
                        resource_id=marker_id,
                        dependency_ids=(dependency_id,),
                        duration_ms=0.0,
                        category="fork_join",
                        layer_id=layer_id,
                        chunk_id=chunk_id,
                        operation="exact_expert_group_optionality",
                    )
                    markers.append(marker)
                    fork_specs[marker_id] = _ForkSpec(
                        marker_id=marker_id,
                        dependency_id=dependency_id,
                        layer_id=layer_id,
                        chunk_id=chunk_id,
                        rows=rows,
                        coordinator_node_id=assignment.coordinator_node_id,
                        primary_nodes=assignment.node_ids,
                    )
            tasks = [
                replacements.get(task.task_id, task)
                for task in tasks
                if task.task_id not in remove_ids
            ]
            tasks.extend(markers)
        prepared = _PreparedTemplate(_topological_task_order(tasks), fork_specs)
        self._template_cache[plan.canonical_sha256] = prepared
        return prepared

    def _fork_specs(
        self,
        plan: E023Plan,
        spec: _ForkSpec,
        ready_ms: float,
    ) -> tuple[GroupExecutionSpec, ...]:
        replicas = {
            row.logical_group_id: row
            for row in plan.replicas
            if row.layer_id == spec.layer_id
        }
        layer = self.model.layers[spec.layer_id]
        protocol = self.service.service_ms(layer, "worker_protocol", 1, spec.rows)
        input_bytes = spec.rows * LATENT * 4 + spec.rows * ROUTE_METADATA_BYTES_PER_ROW
        output_bytes = spec.rows * LATENT * 4
        values: list[GroupExecutionSpec] = []
        for group, primary in enumerate(spec.primary_nodes):
            replica = replicas.get(group)
            alternate = replica.alternate_node_id if replica is not None else None
            resident_nodes = (primary,) if alternate is None else (primary, alternate)
            compute: dict[str, float] = {}
            for node_id in resident_nodes:
                operation = (
                    "expert_whole_group"
                    if node_id == spec.coordinator_node_id
                    else "expert_whole_group_remote"
                )
                compute[node_id] = self.service.service_ms(
                    layer, operation, 8, spec.rows
                ) / self.nodes[node_id].compute_multiplier
            values.append(
                GroupExecutionSpec(
                    logical_group_id=group,
                    coordinator_node_id=spec.coordinator_node_id,
                    primary_node_id=primary,
                    alternate_node_id=alternate,
                    ready_ms=ready_ms,
                    input_bytes=input_bytes,
                    output_bytes=output_bytes,
                    worker_protocol_ms=protocol,
                    compute_ms_by_node=compute,
                )
            )
        return tuple(values)

    def _schedule_pass(
        self,
        plan: E023Plan,
        calendars: ResourceCalendars,
        records: list[ScheduledTaskRecord],
        *,
        slot_id: int,
        pass_index: int,
        pass_start_ms: float,
        retain_records: bool,
    ) -> PassResult:
        prepared = self._prepare(plan)
        completed: dict[str, float] = {}
        record_start = len(records)
        fork_records: list[ForkJoinRecord] = []
        compute_busy: dict[str, float] = {}
        compute_wait: dict[str, float] = {}
        compute_count: dict[str, int] = {}
        tx_busy: dict[str, float] = {}
        rx_busy: dict[str, float] = {}
        link_busy: dict[str, float] = {}
        worker_compute_ms = 0.0
        network_bytes = 0
        transfer_count = 0
        network_queue_wait_ms = 0.0

        def aggregate(
            *,
            category: str,
            node_id: str | None,
            source_node_id: str | None,
            destination_node_id: str | None,
            payload_bytes: int,
            duration_ms: float,
            queue_wait_ms: float,
        ) -> None:
            nonlocal worker_compute_ms
            nonlocal network_bytes
            nonlocal transfer_count
            nonlocal network_queue_wait_ms
            if category == "compute":
                assert node_id is not None
                compute_busy[node_id] = compute_busy.get(node_id, 0.0) + duration_ms
                compute_wait[node_id] = (
                    compute_wait.get(node_id, 0.0) + queue_wait_ms
                )
                compute_count[node_id] = compute_count.get(node_id, 0) + 1
                worker_compute_ms += duration_ms
                return
            assert source_node_id is not None and destination_node_id is not None
            tx_busy[source_node_id] = tx_busy.get(source_node_id, 0.0) + duration_ms
            rx_busy[destination_node_id] = (
                rx_busy.get(destination_node_id, 0.0) + duration_ms
            )
            link_id = f"{source_node_id}->{destination_node_id}"
            link_busy[link_id] = link_busy.get(link_id, 0.0) + duration_ms
            network_bytes += payload_bytes
            transfer_count += 1
            network_queue_wait_ms += queue_wait_ms
        for task in prepared.ordered_tasks:
            identifier = task.task_id
            dependency_ready = max(
                (completed[value] for value in task.dependency_ids),
                default=pass_start_ms,
            )
            dependency_ready = max(dependency_ready, pass_start_ms)
            if task.category == "fork_join":
                spec = prepared.fork_specs[identifier]
                choice = select_fork_join_routes(
                    self._fork_specs(plan, spec, dependency_ready),
                    calendars,
                    self.inventory,
                    self.network_mode,
                    commit=True,
                    metadata_prefix=(
                        f"s{slot_id:03d}.p{pass_index:03d}."
                        f"l{spec.layer_id:02d}.c{spec.chunk_id:02d}"
                    ),
                )
                replica_groups = {
                    row.logical_group_id
                    for row in plan.replicas
                    if row.layer_id == spec.layer_id
                }
                for index, operation in enumerate(choice.operations):
                    primary = spec.primary_nodes[operation.logical_group_id]
                    aggregate(
                        category=operation.category,
                        node_id=operation.node_id,
                        source_node_id=operation.source_node_id,
                        destination_node_id=operation.destination_node_id,
                        payload_bytes=operation.payload_bytes,
                        duration_ms=operation.duration_ms,
                        queue_wait_ms=operation.queue_wait_ms,
                    )
                    if retain_records:
                        records.append(ScheduledTaskRecord(
                            slot_id=slot_id,
                            pass_index=pass_index,
                            local_task_id=f"{identifier}.op-{index:02d}",
                            category=operation.category,
                            operation=operation.operation,
                            layer_id=spec.layer_id,
                            chunk_id=spec.chunk_id,
                            node_id=operation.node_id,
                            source_node_id=operation.source_node_id,
                            destination_node_id=operation.destination_node_id,
                            payload_bytes=operation.payload_bytes,
                            resource_ids=operation.resource_ids,
                            dependency_ready_ms=operation.dependency_ready_ms,
                            start_ms=operation.start_ms,
                            finish_ms=operation.finish_ms,
                            duration_ms=operation.duration_ms,
                            queue_wait_ms=operation.queue_wait_ms,
                            logical_group_id=operation.logical_group_id,
                            selected_alternate=(
                                operation.logical_group_id in replica_groups
                                and choice.selected_nodes[operation.logical_group_id]
                                != primary
                            ),
                        ))
                completed[identifier] = choice.completion_ms
                fork_records.append(
                    ForkJoinRecord(
                        slot_id=slot_id,
                        pass_index=pass_index,
                        layer_id=spec.layer_id,
                        chunk_id=spec.chunk_id,
                        group_arrival_ms=choice.group_arrival_ms,
                        selected_nodes=choice.selected_nodes,
                        mask=choice.mask,
                    )
                )
            else:
                if task.category == "compute":
                    resource_ids = (task.resource_id,)
                elif task.category == "network":
                    assert task.source_node_id is not None
                    assert task.destination_node_id is not None
                    resource_ids = transfer_resource_ids(
                        self.network_mode,
                        task.source_node_id,
                        task.destination_node_id,
                    )
                else:
                    raise ValueError(f"unexpected serving task category {task.category}")
                reservation_method = (
                    calendars.reserve_append_order
                    if self.network_mode is NetworkMode.LEGACY_DIRECTED_LINK
                    else calendars.reserve
                )
                reservation = reservation_method(
                    resource_ids,
                    earliest_ms=dependency_ready,
                    duration_ms=task.duration_ms,
                    metadata={
                        "id": f"s{slot_id:03d}.p{pass_index:03d}.{identifier}",
                        "slot_id": slot_id,
                        "pass_index": pass_index,
                        "operation": task.operation,
                    },
                )
                completed[identifier] = reservation.finish_ms
                aggregate(
                    category=task.category,
                    node_id=task.node_id,
                    source_node_id=task.source_node_id,
                    destination_node_id=task.destination_node_id,
                    payload_bytes=task.payload_bytes,
                    duration_ms=reservation.duration_ms,
                    queue_wait_ms=reservation.queue_wait_ms,
                )
                if retain_records:
                    records.append(ScheduledTaskRecord(
                        slot_id=slot_id,
                        pass_index=pass_index,
                        local_task_id=identifier,
                        category=task.category,
                        operation=task.operation,
                        layer_id=task.layer_id,
                        chunk_id=task.chunk_id,
                        node_id=task.node_id,
                        source_node_id=task.source_node_id,
                        destination_node_id=task.destination_node_id,
                        payload_bytes=task.payload_bytes,
                        resource_ids=resource_ids,
                        dependency_ready_ms=dependency_ready,
                        start_ms=reservation.start_ms,
                        finish_ms=reservation.finish_ms,
                        duration_ms=reservation.duration_ms,
                        queue_wait_ms=reservation.queue_wait_ms,
                    ))
        finish = max(completed.values())

        existing = {(row.layer_id, row.chunk_id) for row in fork_records}
        record_map = {
            row.local_task_id: row for row in records[record_start:]
        } if retain_records else {}
        for assignment in plan.base_plan.assignments:
            if (
                assignment.partition_kind is not PartitionKind.WHOLE_EXPERT
                or assignment.degree != 8
            ):
                continue
            for chunk_id, _rows in enumerate(target_chunks(plan.base_plan.chunk_rows)):
                if (assignment.layer_id, chunk_id) in existing:
                    continue
                prefix = f"c{chunk_id:02d}.l{assignment.layer_id:02d}"
                arrival: dict[int, float] = {}
                selected: dict[int, str] = {}
                for group, node_id in enumerate(assignment.node_ids):
                    if node_id == assignment.coordinator_node_id:
                        task_id = f"{prefix}.expert-{group:02d}"
                    else:
                        task_id = (
                            f"{prefix}.expert-gather.recv-{group:02d}."
                            f"{node_id}.{assignment.coordinator_node_id}"
                        )
                    if retain_records:
                        arrival[group] = record_map[task_id].finish_ms
                    else:
                        arrival[group] = completed[task_id]
                    selected[group] = node_id
                fork_records.append(
                    ForkJoinRecord(
                        slot_id=slot_id,
                        pass_index=pass_index,
                        layer_id=assignment.layer_id,
                        chunk_id=chunk_id,
                        group_arrival_ms=arrival,
                        selected_nodes=selected,
                        mask=0,
                    )
                )
        return PassResult(
            slot_id=slot_id,
            pass_index=pass_index,
            start_ms=pass_start_ms,
            finish_ms=finish,
            latency_ms=finish - pass_start_ms,
            record_start=record_start,
            record_stop=len(records),
            fork_join_records=tuple(
                sorted(fork_records, key=lambda row: (row.layer_id, row.chunk_id))
            ),
            compute_busy_ms_by_node=compute_busy,
            compute_queue_wait_ms_by_node=compute_wait,
            compute_operation_count_by_node=compute_count,
            tx_busy_ms_by_node=tx_busy,
            rx_busy_ms_by_node=rx_busy,
            link_busy_ms=link_busy,
            worker_compute_ms=worker_compute_ms,
            network_bytes=network_bytes,
            transfer_count=transfer_count,
            network_queue_wait_ms=network_queue_wait_ms,
        )

    def run_single_pass(self, plan: E023Plan) -> SinglePassRun:
        calendars = ResourceCalendars()
        records: list[ScheduledTaskRecord] = []
        result = self._schedule_pass(
            plan,
            calendars,
            records,
            slot_id=0,
            pass_index=0,
            pass_start_ms=0.0,
            retain_records=True,
        )
        compute = [row for row in records if row.category == "compute"]
        network = [row for row in records if row.category == "network"]
        workers = tuple(sorted({row.node_id for row in compute if row.node_id is not None}))
        replica_layers = {row.layer_id for row in plan.replicas}
        masks = tuple(
            row.mask
            for row in result.fork_join_records
            if row.layer_id in replica_layers
        )
        return SinglePassRun(
            critical_path_ms=result.finish_ms,
            total_worker_compute_ms=sum(row.duration_ms for row in compute),
            total_network_ms=sum(row.duration_ms for row in network),
            network_bytes=sum(row.payload_bytes for row in network),
            messages=len(network),
            event_count=len(records),
            participating_workers=workers,
            records=tuple(records),
            fork_join_records=result.fork_join_records,
            selected_masks=masks,
        )

    def run_closed_loop(
        self,
        plan: E023Plan,
        concurrency: int,
        *,
        retain_records: bool = False,
    ) -> ServingRun:
        if concurrency not in CONCURRENCY_LEVELS:
            raise ValueError("concurrency must use the frozen E023 ladder")
        calendars = ResourceCalendars()
        records: list[ScheduledTaskRecord] = []
        passes: list[PassResult] = []
        last_finish = [0.0 for _ in range(concurrency)]
        initial_total = 2 + measured_passes_per_slot(concurrency)

        def schedule_round(pass_index: int) -> None:
            for slot_id in range(concurrency):
                result = self._schedule_pass(
                    plan,
                    calendars,
                    records,
                    slot_id=slot_id,
                    pass_index=pass_index,
                    pass_start_ms=last_finish[slot_id],
                    retain_records=retain_records,
                )
                passes.append(result)
                last_finish[slot_id] = result.finish_ms

        for pass_index in range(initial_total):
            schedule_round(pass_index)
        t0, eligible = _eligible_passes(passes)
        extra_rounds = 0
        while len(eligible) < 64 and extra_rounds < 8:
            schedule_round(initial_total + extra_rounds)
            extra_rounds += 1
            t0, eligible = _eligible_passes(passes)

        status = "PASS" if len(eligible) >= 64 else "INSUFFICIENT_STEADY_STATE_SAMPLES"
        if not eligible:
            raise RuntimeError("closed-loop simulation produced no eligible pass")
        t1 = max(row.finish_ms for row in eligible)
        t_start = min(row.start_ms for row in eligible)
        window = t1 - t_start
        if window <= 0:
            raise RuntimeError("closed-loop measurement window is not positive")
        compute_busy: dict[str, float] = {}
        compute_wait: dict[str, float] = {}
        compute_count: dict[str, int] = {}
        tx_busy: dict[str, float] = {}
        rx_busy: dict[str, float] = {}
        link_busy: dict[str, float] = {}
        network_bytes = 0
        transfer_count = 0
        worker_compute_ms = 0.0
        network_queue_wait_ms = 0.0
        for pass_row in eligible:
            for node_id, value in pass_row.compute_busy_ms_by_node.items():
                compute_busy[node_id] = compute_busy.get(node_id, 0.0) + value
            for node_id, value in pass_row.compute_queue_wait_ms_by_node.items():
                compute_wait[node_id] = compute_wait.get(node_id, 0.0) + value
            for node_id, value in pass_row.compute_operation_count_by_node.items():
                compute_count[node_id] = compute_count.get(node_id, 0) + value
            for node_id, value in pass_row.tx_busy_ms_by_node.items():
                tx_busy[node_id] = tx_busy.get(node_id, 0.0) + value
            for node_id, value in pass_row.rx_busy_ms_by_node.items():
                rx_busy[node_id] = rx_busy.get(node_id, 0.0) + value
            for link_id, value in pass_row.link_busy_ms.items():
                link_busy[link_id] = link_busy.get(link_id, 0.0) + value
            network_bytes += pass_row.network_bytes
            transfer_count += pass_row.transfer_count
            worker_compute_ms += pass_row.worker_compute_ms
            network_queue_wait_ms += pass_row.network_queue_wait_ms
        selected_masks: list[int] = []
        primary_count = 0
        alternate_count = 0
        replica_keys = {(row.layer_id, row.logical_group_id) for row in plan.replicas}
        for pass_row in eligible:
            for fork in pass_row.fork_join_records:
                if any(layer == fork.layer_id for layer, _ in replica_keys):
                    selected_masks.append(fork.mask)
                assignment = plan.base_plan.assignments[fork.layer_id]
                for group, node_id in fork.selected_nodes.items():
                    if (fork.layer_id, group) not in replica_keys:
                        continue
                    if node_id == assignment.node_ids[group]:
                        primary_count += 1
                    else:
                        alternate_count += 1
        latencies = np.asarray([row.latency_ms for row in eligible], dtype=np.float64)
        throughput = TARGET_ROWS * len(eligible) / (window / 1000.0)

        def maximum_utilization(values: dict[str, float]) -> float:
            return max((busy / window for busy in values.values()), default=0.0)

        return ServingRun(
            status=status,
            network_mode=self.network_mode,
            concurrency=concurrency,
            target_passes_measured=len(eligible),
            target_rows_measured=TARGET_ROWS * len(eligible),
            measurement_window_ms=window,
            target_rows_per_second=throughput,
            p50_pass_latency_ms=float(np.quantile(latencies, 0.50, method="linear")),
            p95_pass_latency_ms=float(np.quantile(latencies, 0.95, method="linear")),
            t0_ms=t0,
            t_start_ms=t_start,
            t1_ms=t1,
            network_bytes=network_bytes,
            transfer_count=transfer_count,
            worker_compute_ms=worker_compute_ms,
            compute_queue_wait_ms=sum(compute_wait.values()),
            network_queue_wait_ms=network_queue_wait_ms,
            maximum_compute_utilization=maximum_utilization(compute_busy),
            maximum_tx_utilization=maximum_utilization(tx_busy),
            maximum_rx_utilization=maximum_utilization(rx_busy),
            maximum_link_utilization=maximum_utilization(link_busy),
            compute_busy_ms_by_node=compute_busy,
            compute_queue_wait_ms_by_node=compute_wait,
            compute_operation_count_by_node=compute_count,
            tx_busy_ms_by_node=tx_busy,
            rx_busy_ms_by_node=rx_busy,
            link_busy_ms=link_busy,
            participating_workers=tuple(sorted(compute_busy)),
            selected_masks=tuple(selected_masks),
            primary_selection_count=primary_count,
            alternate_selection_count=alternate_count,
            passes=tuple(passes),
            records=tuple(records),
            insufficient_extra_pass_rounds=extra_rounds,
        )


__all__ = [
    "CONCURRENCY_LEVELS",
    "TARGET_ROWS",
    "PassResult",
    "ScheduledTaskRecord",
    "ServingEngine",
    "ServingRun",
    "SinglePassRun",
    "measured_passes_per_slot",
]
