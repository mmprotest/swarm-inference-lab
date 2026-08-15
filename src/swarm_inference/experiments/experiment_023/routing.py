"""Exact non-clairvoyant mask enumeration for alternative expert residency."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Any

from swarm_inference.experiments.experiment_022.models import Inventory, NodeCapability

from .models import NetworkMode
from .resource_calendar import ResourceCalendars, reserve_transfer, transfer_resource_ids


@dataclass(frozen=True, slots=True)
class GroupExecutionSpec:
    logical_group_id: int
    coordinator_node_id: str
    primary_node_id: str
    alternate_node_id: str | None
    ready_ms: float
    input_bytes: int
    output_bytes: int
    worker_protocol_ms: float
    compute_ms_by_node: dict[str, float]

    def __post_init__(self) -> None:
        if not 0 <= self.logical_group_id < 8:
            raise ValueError("logical group ID must be in [0, 7]")
        expected = {self.primary_node_id}
        if self.alternate_node_id is not None:
            if self.alternate_node_id == self.primary_node_id:
                raise ValueError("alternate and primary nodes must differ")
            expected.add(self.alternate_node_id)
        if set(self.compute_ms_by_node) != expected:
            raise ValueError("compute service must be supplied for every resident copy")
        if min(self.compute_ms_by_node.values()) <= 0 or self.worker_protocol_ms <= 0:
            raise ValueError("routing service durations must be positive")


@dataclass(frozen=True, slots=True)
class RoutedOperation:
    logical_group_id: int
    operation: str
    category: str
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


@dataclass(frozen=True, slots=True)
class ForkJoinChoice:
    mask: int
    completion_ms: float
    total_queue_wait_ms: float
    selected_nodes: dict[int, str]
    group_arrival_ms: dict[int, float]
    operations: tuple[RoutedOperation, ...]


def _schedule_mask(
    specs: tuple[GroupExecutionSpec, ...],
    mask: int,
    replicated_groups: tuple[int, ...],
    calendars: ResourceCalendars,
    inventory: Inventory,
    mode: NetworkMode,
    *,
    metadata_prefix: str,
) -> ForkJoinChoice:
    bit_by_group = {group: bit for bit, group in enumerate(replicated_groups)}
    selected: dict[int, str] = {}
    arrival: dict[int, float] = {}
    operations: list[RoutedOperation] = []
    total_wait = 0.0
    for spec in sorted(specs, key=lambda value: value.logical_group_id):
        bit = bit_by_group.get(spec.logical_group_id)
        use_alternate = bit is not None and bool(mask & (1 << bit))
        node_id = (
            spec.alternate_node_id
            if use_alternate and spec.alternate_node_id is not None
            else spec.primary_node_id
        )
        selected[spec.logical_group_id] = node_id
        ready = spec.ready_ms
        common_metadata: dict[str, Any] = {
            "id": f"{metadata_prefix}.group-{spec.logical_group_id:02d}",
            "logical_group_id": spec.logical_group_id,
            "selected_node_id": node_id,
            "mask": mask,
        }
        if node_id != spec.coordinator_node_id:
            transfer = reserve_transfer(
                calendars,
                inventory,
                mode,
                source=spec.coordinator_node_id,
                destination=node_id,
                earliest_ms=ready,
                payload_bytes=spec.input_bytes,
                metadata={**common_metadata, "operation": "expert_group_input"},
            )
            operations.append(
                RoutedOperation(
                    logical_group_id=spec.logical_group_id,
                    operation="expert_group_input",
                    category="network",
                    node_id=None,
                    source_node_id=spec.coordinator_node_id,
                    destination_node_id=node_id,
                    payload_bytes=spec.input_bytes,
                    resource_ids=transfer.resource_ids,
                    dependency_ready_ms=ready,
                    start_ms=transfer.start_ms,
                    finish_ms=transfer.finish_ms,
                    duration_ms=transfer.duration_ms,
                    queue_wait_ms=transfer.queue_wait_ms,
                )
            )
            total_wait += transfer.queue_wait_ms
            ready = transfer.finish_ms
            protocol = calendars.reserve(
                (f"compute:{node_id}",),
                earliest_ms=ready,
                duration_ms=spec.worker_protocol_ms,
                metadata={**common_metadata, "operation": "worker_protocol"},
            )
            operations.append(
                RoutedOperation(
                    logical_group_id=spec.logical_group_id,
                    operation="worker_protocol:expert_whole_group",
                    category="compute",
                    node_id=node_id,
                    source_node_id=None,
                    destination_node_id=None,
                    payload_bytes=0,
                    resource_ids=protocol.resource_ids,
                    dependency_ready_ms=ready,
                    start_ms=protocol.start_ms,
                    finish_ms=protocol.finish_ms,
                    duration_ms=protocol.duration_ms,
                    queue_wait_ms=protocol.queue_wait_ms,
                )
            )
            total_wait += protocol.queue_wait_ms
            ready = protocol.finish_ms
        compute = calendars.reserve(
            (f"compute:{node_id}",),
            earliest_ms=ready,
            duration_ms=spec.compute_ms_by_node[node_id],
            metadata={**common_metadata, "operation": "expert_whole_group"},
        )
        operations.append(
            RoutedOperation(
                logical_group_id=spec.logical_group_id,
                operation="expert_whole_group",
                category="compute",
                node_id=node_id,
                source_node_id=None,
                destination_node_id=None,
                payload_bytes=0,
                resource_ids=compute.resource_ids,
                dependency_ready_ms=ready,
                start_ms=compute.start_ms,
                finish_ms=compute.finish_ms,
                duration_ms=compute.duration_ms,
                queue_wait_ms=compute.queue_wait_ms,
            )
        )
        total_wait += compute.queue_wait_ms
        ready = compute.finish_ms
        if node_id != spec.coordinator_node_id:
            transfer = reserve_transfer(
                calendars,
                inventory,
                mode,
                source=node_id,
                destination=spec.coordinator_node_id,
                earliest_ms=ready,
                payload_bytes=spec.output_bytes,
                metadata={**common_metadata, "operation": "expert_group_output"},
            )
            operations.append(
                RoutedOperation(
                    logical_group_id=spec.logical_group_id,
                    operation="expert_group_output",
                    category="network",
                    node_id=None,
                    source_node_id=node_id,
                    destination_node_id=spec.coordinator_node_id,
                    payload_bytes=spec.output_bytes,
                    resource_ids=transfer.resource_ids,
                    dependency_ready_ms=ready,
                    start_ms=transfer.start_ms,
                    finish_ms=transfer.finish_ms,
                    duration_ms=transfer.duration_ms,
                    queue_wait_ms=transfer.queue_wait_ms,
                )
            )
            total_wait += transfer.queue_wait_ms
            ready = transfer.finish_ms
        arrival[spec.logical_group_id] = ready
    return ForkJoinChoice(
        mask=mask,
        completion_ms=max(arrival.values()),
        total_queue_wait_ms=total_wait,
        selected_nodes=selected,
        group_arrival_ms=arrival,
        operations=tuple(operations),
    )


def _schedule_group(
    spec: GroupExecutionSpec,
    node_id: str,
    calendars: ResourceCalendars,
    inventory: Inventory,
    mode: NetworkMode,
    *,
    metadata_prefix: str,
    mask: int,
) -> tuple[float, float, tuple[RoutedOperation, ...]]:
    """Schedule one canonical logical slot on a speculative calendar."""

    operations: list[RoutedOperation] = []
    total_wait = 0.0
    ready = spec.ready_ms
    common_metadata: dict[str, Any] = {
        "id": f"{metadata_prefix}.group-{spec.logical_group_id:02d}",
        "logical_group_id": spec.logical_group_id,
        "selected_node_id": node_id,
        "mask": mask,
    }
    if node_id != spec.coordinator_node_id:
        transfer = reserve_transfer(
            calendars,
            inventory,
            mode,
            source=spec.coordinator_node_id,
            destination=node_id,
            earliest_ms=ready,
            payload_bytes=spec.input_bytes,
            metadata={**common_metadata, "operation": "expert_group_input"},
        )
        operations.append(
            RoutedOperation(
                logical_group_id=spec.logical_group_id,
                operation="expert_group_input",
                category="network",
                node_id=None,
                source_node_id=spec.coordinator_node_id,
                destination_node_id=node_id,
                payload_bytes=spec.input_bytes,
                resource_ids=transfer.resource_ids,
                dependency_ready_ms=ready,
                start_ms=transfer.start_ms,
                finish_ms=transfer.finish_ms,
                duration_ms=transfer.duration_ms,
                queue_wait_ms=transfer.queue_wait_ms,
            )
        )
        total_wait += transfer.queue_wait_ms
        ready = transfer.finish_ms
        protocol = calendars.reserve(
            (f"compute:{node_id}",),
            earliest_ms=ready,
            duration_ms=spec.worker_protocol_ms,
            metadata={**common_metadata, "operation": "worker_protocol"},
        )
        operations.append(
            RoutedOperation(
                logical_group_id=spec.logical_group_id,
                operation="worker_protocol:expert_whole_group",
                category="compute",
                node_id=node_id,
                source_node_id=None,
                destination_node_id=None,
                payload_bytes=0,
                resource_ids=protocol.resource_ids,
                dependency_ready_ms=ready,
                start_ms=protocol.start_ms,
                finish_ms=protocol.finish_ms,
                duration_ms=protocol.duration_ms,
                queue_wait_ms=protocol.queue_wait_ms,
            )
        )
        total_wait += protocol.queue_wait_ms
        ready = protocol.finish_ms
    compute = calendars.reserve(
        (f"compute:{node_id}",),
        earliest_ms=ready,
        duration_ms=spec.compute_ms_by_node[node_id],
        metadata={**common_metadata, "operation": "expert_whole_group"},
    )
    operations.append(
        RoutedOperation(
            logical_group_id=spec.logical_group_id,
            operation="expert_whole_group",
            category="compute",
            node_id=node_id,
            source_node_id=None,
            destination_node_id=None,
            payload_bytes=0,
            resource_ids=compute.resource_ids,
            dependency_ready_ms=ready,
            start_ms=compute.start_ms,
            finish_ms=compute.finish_ms,
            duration_ms=compute.duration_ms,
            queue_wait_ms=compute.queue_wait_ms,
        )
    )
    total_wait += compute.queue_wait_ms
    ready = compute.finish_ms
    if node_id != spec.coordinator_node_id:
        transfer = reserve_transfer(
            calendars,
            inventory,
            mode,
            source=node_id,
            destination=spec.coordinator_node_id,
            earliest_ms=ready,
            payload_bytes=spec.output_bytes,
            metadata={**common_metadata, "operation": "expert_group_output"},
        )
        operations.append(
            RoutedOperation(
                logical_group_id=spec.logical_group_id,
                operation="expert_group_output",
                category="network",
                node_id=None,
                source_node_id=node_id,
                destination_node_id=spec.coordinator_node_id,
                payload_bytes=spec.output_bytes,
                resource_ids=transfer.resource_ids,
                dependency_ready_ms=ready,
                start_ms=transfer.start_ms,
                finish_ms=transfer.finish_ms,
                duration_ms=transfer.duration_ms,
                queue_wait_ms=transfer.queue_wait_ms,
            )
        )
        total_wait += transfer.queue_wait_ms
        ready = transfer.finish_ms
    return ready, total_wait, tuple(operations)


def _enumerate_masks_persistent(
    specs: tuple[GroupExecutionSpec, ...],
    replicated_groups: tuple[int, ...],
    calendars: ResourceCalendars,
    inventory: Inventory,
    mode: NetworkMode,
    *,
    metadata_prefix: str,
) -> list[ForkJoinChoice]:
    """Enumerate all masks with immutable prefix sharing.

    Each leaf owns a distinct overlay calendar (and is therefore the required
    what-if clone for that mask), while masks with the same canonical prefix
    share the already-computed immutable prefix.  This changes no ordering or
    reservation semantics and avoids rebuilding that prefix up to 256 times.
    """

    ordered = tuple(sorted(specs, key=lambda value: value.logical_group_id))
    bit_by_group = {group: bit for bit, group in enumerate(replicated_groups)}
    candidates: list[ForkJoinChoice] = []

    def visit(
        index: int,
        branch: ResourceCalendars,
        mask: int,
        selected: dict[int, str],
        arrivals: dict[int, float],
        operations: tuple[RoutedOperation, ...],
        total_wait: float,
    ) -> None:
        if index == len(ordered):
            candidates.append(
                ForkJoinChoice(
                    mask=mask,
                    completion_ms=max(arrivals.values()),
                    total_queue_wait_ms=total_wait,
                    selected_nodes=selected,
                    group_arrival_ms=arrivals,
                    operations=operations,
                )
            )
            return
        spec = ordered[index]
        bit = bit_by_group.get(spec.logical_group_id)
        choices = [(spec.primary_node_id, mask)]
        if bit is not None:
            assert spec.alternate_node_id is not None
            choices.append((spec.alternate_node_id, mask | (1 << bit)))
        for node_id, updated_mask in choices:
            child = branch.fork()
            arrival, queue_wait, group_operations = _schedule_group(
                spec,
                node_id,
                child,
                inventory,
                mode,
                metadata_prefix=metadata_prefix,
                mask=updated_mask,
            )
            visit(
                index + 1,
                child,
                updated_mask,
                {**selected, spec.logical_group_id: node_id},
                {**arrivals, spec.logical_group_id: arrival},
                operations + group_operations,
                total_wait + queue_wait,
            )

    visit(0, calendars, 0, {}, {}, (), 0.0)
    return candidates


ScratchIntervals = dict[str, tuple[tuple[float, float], ...]]


def _scratch_local_start(
    intervals: tuple[tuple[float, float], ...],
    earliest_ms: float,
    duration_ms: float,
) -> float:
    candidate = earliest_ms
    for start, finish in intervals:
        if finish <= candidate:
            continue
        if candidate + duration_ms <= start:
            return candidate
        candidate = finish
    return candidate


def _scratch_resource_start(
    calendars: ResourceCalendars,
    intervals: ScratchIntervals,
    resource_id: str,
    earliest_ms: float,
    duration_ms: float,
) -> float:
    candidate = earliest_ms
    local = intervals.get(resource_id, ())
    base = calendars.calendar(resource_id)
    while True:
        updated = max(
            base.earliest_start(candidate, duration_ms),
            _scratch_local_start(local, candidate, duration_ms),
        )
        if updated == candidate:
            return candidate
        candidate = updated


def _scratch_reserve(
    calendars: ResourceCalendars,
    intervals: ScratchIntervals,
    resource_ids: tuple[str, ...],
    earliest_ms: float,
    duration_ms: float,
) -> tuple[float, float]:
    resources = tuple(dict.fromkeys(resource_ids))
    candidate = earliest_ms
    while True:
        updated = max(
            _scratch_resource_start(
                calendars,
                intervals,
                resource_id,
                candidate,
                duration_ms,
            )
            for resource_id in resources
        )
        if updated == candidate:
            break
        candidate = updated
    finish = candidate + duration_ms
    for resource_id in resources:
        current = intervals.get(resource_id, ())
        starts = [value[0] for value in current]
        index = bisect.bisect_left(starts, candidate)
        values = list(current)
        values.insert(index, (candidate, finish))
        intervals[resource_id] = tuple(values)
    return finish, candidate - earliest_ms


def _scratch_schedule_group(
    spec: GroupExecutionSpec,
    node_id: str,
    calendars: ResourceCalendars,
    intervals: ScratchIntervals,
    nodes: dict[str, NodeCapability],
    mode: NetworkMode,
) -> tuple[float, float]:
    ready = spec.ready_ms
    total_wait = 0.0
    if node_id != spec.coordinator_node_id:
        duration = nodes[spec.coordinator_node_id].peer(node_id).transfer_ms(
            spec.input_bytes
        )
        ready, wait = _scratch_reserve(
            calendars,
            intervals,
            transfer_resource_ids(mode, spec.coordinator_node_id, node_id),
            ready,
            duration,
        )
        total_wait += wait
        ready, wait = _scratch_reserve(
            calendars,
            intervals,
            (f"compute:{node_id}",),
            ready,
            spec.worker_protocol_ms,
        )
        total_wait += wait
    ready, wait = _scratch_reserve(
        calendars,
        intervals,
        (f"compute:{node_id}",),
        ready,
        spec.compute_ms_by_node[node_id],
    )
    total_wait += wait
    if node_id != spec.coordinator_node_id:
        duration = nodes[node_id].peer(spec.coordinator_node_id).transfer_ms(
            spec.output_bytes
        )
        ready, wait = _scratch_reserve(
            calendars,
            intervals,
            transfer_resource_ids(mode, node_id, spec.coordinator_node_id),
            ready,
            duration,
        )
        total_wait += wait
    return ready, total_wait


def _select_mask_fast(
    specs: tuple[GroupExecutionSpec, ...],
    calendars: ResourceCalendars,
    inventory: Inventory,
    mode: NetworkMode,
) -> tuple[int, float, float, int]:
    """Exhaustively enumerate every mask using compact speculative intervals."""

    ordered = tuple(sorted(specs, key=lambda value: value.logical_group_id))
    replicated = tuple(
        value.logical_group_id
        for value in ordered
        if value.alternate_node_id is not None
    )
    bit_by_group = {group: bit for bit, group in enumerate(replicated)}
    nodes = inventory.node_map()
    best: tuple[float, float, int] | None = None
    leaf_count = 0

    def visit(
        index: int,
        intervals: ScratchIntervals,
        mask: int,
        arrivals: tuple[float, ...],
        total_wait: float,
    ) -> None:
        nonlocal best, leaf_count
        if index == len(ordered):
            leaf_count += 1
            candidate = (max(arrivals), total_wait, mask)
            if best is None or candidate < best:
                best = candidate
            return
        spec = ordered[index]
        bit = bit_by_group.get(spec.logical_group_id)
        choices = ((spec.primary_node_id, mask),)
        if bit is not None:
            assert spec.alternate_node_id is not None
            choices += ((spec.alternate_node_id, mask | (1 << bit)),)
        for node_id, updated_mask in choices:
            child = dict(intervals)
            arrival, wait = _scratch_schedule_group(
                spec,
                node_id,
                calendars,
                child,
                nodes,
                mode,
            )
            visit(
                index + 1,
                child,
                updated_mask,
                (*arrivals, arrival),
                total_wait + wait,
            )

    visit(0, {}, 0, (), 0.0)
    assert best is not None
    return best[2], best[0], best[1], leaf_count


def select_fork_join_routes(
    specs: tuple[GroupExecutionSpec, ...],
    calendars: ResourceCalendars,
    inventory: Inventory,
    mode: NetworkMode,
    *,
    commit: bool = True,
    metadata_prefix: str = "fork",
) -> ForkJoinChoice:
    """Enumerate every copy mask using only current deterministic reservations."""

    if not specs:
        raise ValueError("fork-join routing requires at least one logical group")
    group_ids = [value.logical_group_id for value in specs]
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("fork-join logical groups must be unique")
    replicated = tuple(
        value.logical_group_id
        for value in sorted(specs, key=lambda row: row.logical_group_id)
        if value.alternate_node_id is not None
    )
    mask, predicted_completion, predicted_wait, leaf_count = _select_mask_fast(
        specs, calendars, inventory, mode
    )
    if leaf_count != 1 << len(replicated):
        raise AssertionError("routing did not enumerate every copy-selection mask")
    if not commit:
        return _schedule_mask(
            specs,
            mask,
            replicated,
            calendars.fork(),
            inventory,
            mode,
            metadata_prefix=metadata_prefix,
        )
    winner = _schedule_mask(
        specs,
        mask,
        replicated,
        calendars,
        inventory,
        mode,
        metadata_prefix=metadata_prefix,
    )
    scale = max(abs(predicted_completion), abs(winner.completion_ms), 1.0)
    if (
        abs(predicted_completion - winner.completion_ms) > 1e-12 * scale
        or abs(predicted_wait - winner.total_queue_wait_ms) > 1e-9
    ):
        raise AssertionError("compact mask enumeration diverged from committed schedule")
    return winner


__all__ = [
    "ForkJoinChoice",
    "GroupExecutionSpec",
    "RoutedOperation",
    "select_fork_join_routes",
]
