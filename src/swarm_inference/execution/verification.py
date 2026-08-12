"""Exact scheduling primitives for speculative target-verification blocks.

The objects in this module are deliberately independent of Kimi and CUDA.  A
model runtime can use the same deterministic assignment plan on CPU or device,
while retaining token-major reduction order for exact inference semantics.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class VerificationBlock:
    """A contiguous unit of known target work.

    Speculative decoding normally verifies ``candidate_count`` proposed tokens
    and obtains one bonus target token from the same forward pass.  Set
    ``include_bonus_token=False`` for callers which only need candidate rows.
    """

    session_id: str
    cache_position_start: int
    candidate_count: int
    include_bonus_token: bool = True

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("verification block session ID must not be empty")
        if self.cache_position_start < 0:
            raise ValueError("verification block cache position must be non-negative")
        if self.candidate_count < 1:
            raise ValueError("verification block must contain at least one candidate")

    @property
    def row_count(self) -> int:
        return self.candidate_count + int(self.include_bonus_token)

    @property
    def positions(self) -> tuple[int, ...]:
        return tuple(range(self.cache_position_start, self.cache_position_start + self.row_count))


@dataclass(frozen=True, slots=True)
class ExpertAssignment:
    """One token/slot assignment in deterministic expert-major order."""

    expert_id: int
    row: int
    slot: int
    work_index: int


@dataclass(frozen=True, slots=True)
class ExpertGroup:
    """Contiguous work belonging to one expert."""

    expert_id: int
    start: int
    count: int
    native_chunks: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GroupedExpertPlan:
    """Immutable exact gather/execute/scatter plan for routed rows."""

    rows: int
    topk: int
    assignments: tuple[ExpertAssignment, ...]
    groups: tuple[ExpertGroup, ...]
    row_slot_to_work: tuple[tuple[int, ...], ...]

    @property
    def total_assignments(self) -> int:
        return self.rows * self.topk

    @property
    def unique_experts(self) -> int:
        return len(self.groups)

    @property
    def native_call_count(self) -> int:
        return sum(len(group.native_chunks) for group in self.groups)


def partition_certified_batch(
    count: int,
    supported_batch_sizes: Iterable[int],
) -> tuple[int, ...]:
    """Greedily partition work into exact native batch sizes.

    The result is deterministic and fails closed when batch one is absent or a
    count cannot be represented.
    """

    if count <= 0:
        raise ValueError("expert batch count must be positive")
    supported = tuple(
        sorted({int(size) for size in supported_batch_sizes if int(size) > 0}, reverse=True)
    )
    if not supported or 1 not in supported:
        raise ValueError("certified batch sizes must include one")
    chunks: list[int] = []
    remaining = count
    while remaining:
        chunk = next((size for size in supported if size <= remaining), None)
        if chunk is None:
            raise ValueError(f"cannot partition batch {count} into {supported}")
        chunks.append(chunk)
        remaining -= chunk
    return tuple(chunks)


def build_grouped_expert_plan(
    route_ids: np.ndarray | Sequence[Sequence[int]],
    *,
    supported_batch_sizes: Iterable[int] = (1,),
) -> GroupedExpertPlan:
    """Build a stable expert-major plan while retaining token-major identity.

    Expert groups are ordered by expert ID.  Work within each group is ordered
    by token row and then router slot.  This makes dispatch reproducible across
    Python versions and allows reduction to restore the canonical slot order.
    """

    routes = np.asarray(route_ids)
    if routes.ndim != 2 or routes.shape[0] < 1 or routes.shape[1] < 1:
        raise ValueError("route IDs must be a non-empty [rows, topk] matrix")
    if not np.issubdtype(routes.dtype, np.integer):
        raise TypeError("route IDs must be integers")
    if np.any(routes < 0):
        raise ValueError("route IDs must be non-negative")
    rows, topk = (int(routes.shape[0]), int(routes.shape[1]))
    for row in range(rows):
        if len(set(int(value) for value in routes[row])) != topk:
            raise ValueError("each routed row must contain unique experts")
    supported = tuple(supported_batch_sizes)

    grouped: dict[int, list[tuple[int, int]]] = {}
    for row in range(rows):
        for slot in range(topk):
            grouped.setdefault(int(routes[row, slot]), []).append((row, slot))

    assignments: list[ExpertAssignment] = []
    groups: list[ExpertGroup] = []
    inverse = [[-1] * topk for _ in range(rows)]
    for expert_id in sorted(grouped):
        tasks = sorted(grouped[expert_id])
        start = len(assignments)
        for row, slot in tasks:
            work_index = len(assignments)
            assignments.append(
                ExpertAssignment(
                    expert_id=expert_id,
                    row=row,
                    slot=slot,
                    work_index=work_index,
                )
            )
            inverse[row][slot] = work_index
        groups.append(
            ExpertGroup(
                expert_id=expert_id,
                start=start,
                count=len(tasks),
                native_chunks=partition_certified_batch(len(tasks), supported),
            )
        )

    if any(index < 0 for row in inverse for index in row):
        raise RuntimeError("grouped expert plan omitted a token/slot assignment")
    return GroupedExpertPlan(
        rows=rows,
        topk=topk,
        assignments=tuple(assignments),
        groups=tuple(groups),
        row_slot_to_work=tuple(tuple(row) for row in inverse),
    )


def scatter_grouped_expert_outputs(
    grouped_outputs: np.ndarray,
    plan: GroupedExpertPlan,
) -> np.ndarray:
    """Restore grouped expert results to exact ``[row, slot, value]`` order."""

    outputs = np.asarray(grouped_outputs)
    if outputs.ndim != 2 or outputs.shape[0] != plan.total_assignments:
        raise ValueError("grouped expert outputs do not match the assignment plan")
    restored = np.empty((plan.rows, plan.topk, outputs.shape[1]), dtype=outputs.dtype)
    for assignment in plan.assignments:
        restored[assignment.row, assignment.slot] = outputs[assignment.work_index]
    return restored


def deterministic_scatter_reduce(
    grouped_outputs: np.ndarray,
    route_weights: np.ndarray,
    plan: GroupedExpertPlan,
) -> np.ndarray:
    """Scatter and reduce in canonical router-slot order using FP32."""

    weights = np.asarray(route_weights, dtype=np.float32)
    if weights.shape != (plan.rows, plan.topk):
        raise ValueError("route weights do not match the assignment plan")
    restored = np.asarray(scatter_grouped_expert_outputs(grouped_outputs, plan), dtype=np.float32)
    reduced = np.zeros((plan.rows, restored.shape[-1]), dtype=np.float32)
    for row in range(plan.rows):
        for slot in range(plan.topk):
            reduced[row] += restored[row, slot] * weights[row, slot]
    return reduced


def _percentile(values: Sequence[int], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def expert_reuse_statistics(
    route_ids: np.ndarray | Sequence[Sequence[int]],
) -> dict[str, object]:
    """Summarize within-block expert reuse without assuming a model family."""

    routes = np.asarray(route_ids)
    plan = build_grouped_expert_plan(routes)
    group_sizes = [group.count for group in plan.groups]
    adjacent_intersections: list[int] = []
    adjacent_unions: list[int] = []
    for row in range(1, plan.rows):
        previous = set(int(value) for value in routes[row - 1])
        current = set(int(value) for value in routes[row])
        adjacent_intersections.append(len(previous & current))
        adjacent_unions.append(len(previous | current))
    histogram: dict[str, int] = {}
    for size in group_sizes:
        key = str(size)
        histogram[key] = histogram.get(key, 0) + 1
    return {
        "rows": plan.rows,
        "topk": plan.topk,
        "total_assignments": plan.total_assignments,
        "unique_experts": plan.unique_experts,
        "unique_experts_per_assignment": plan.unique_experts / plan.total_assignments,
        "mean_assignments_per_touched_expert": plan.total_assignments / plan.unique_experts,
        "maximum_assignments_for_one_expert": max(group_sizes),
        "assignments_per_touched_expert_p50": _percentile(group_sizes, 0.50),
        "assignments_per_touched_expert_p95": _percentile(group_sizes, 0.95),
        "assignments_per_touched_expert_p99": _percentile(group_sizes, 0.99),
        "expert_group_size_histogram": histogram,
        "mean_adjacent_position_overlap": (
            float(np.mean(adjacent_intersections)) if adjacent_intersections else 0.0
        ),
        "mean_adjacent_position_jaccard": (
            float(
                np.mean(
                    [
                        intersection / union
                        for intersection, union in zip(
                            adjacent_intersections, adjacent_unions, strict=True
                        )
                    ]
                )
            )
            if adjacent_intersections
            else 0.0
        ),
    }


__all__ = [
    "ExpertAssignment",
    "ExpertGroup",
    "GroupedExpertPlan",
    "VerificationBlock",
    "build_grouped_expert_plan",
    "deterministic_scatter_reduce",
    "expert_reuse_statistics",
    "partition_certified_batch",
    "scatter_grouped_expert_outputs",
]
