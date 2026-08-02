"""Measured latency interpolation and critical-path costing for Experiment 008."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from statistics import mean
from typing import Any

from pydantic import Field

from swarm_inference.config.models import StrictModel
from swarm_inference.experiments.experiment_008.schemas import CostBreakdown


class UtilityObjective(StrEnum):
    MAXIMUM_DECODE_THROUGHPUT = "maximum_decode_throughput"
    MINIMUM_TIME_TO_FIRST_TOKEN = "minimum_time_to_first_token"
    MAXIMUM_MIXED_VERIFIED_THROUGHPUT = "maximum_mixed_verified_throughput"
    MINIMUM_PEAK_VRAM_SUBJECT_TO_LATENCY = "minimum_peak_vram_subject_to_latency"


class MeasuredKernelPoint(StrictModel):
    operation: str
    device: str
    shape: list[int]
    median_ms: float = Field(gt=0)
    p95_ms: float = Field(gt=0)
    effective_bandwidth_bytes_s: float = Field(ge=0)
    evidence_class: str = "MEASURED"


class TransferPoint(StrictModel):
    direction: str
    memory_kind: str
    payload_bytes: int = Field(gt=0)
    median_ms: float = Field(gt=0)
    p95_ms: float = Field(gt=0)
    effective_bandwidth_bytes_s: float = Field(gt=0)
    evidence_class: str = "MEASURED"


@dataclass(frozen=True, slots=True)
class CriticalTask:
    task_id: str
    duration_ms: float
    dependencies: tuple[str, ...] = ()


class CandidateEstimate(StrictModel):
    plan_id: str
    objective: UtilityObjective
    decode_tokens_per_second: float = Field(gt=0)
    time_to_first_token_ms: float = Field(gt=0)
    mixed_verified_tokens_per_second: float = Field(gt=0)
    interactive_p95_ms: float = Field(gt=0)
    peak_vram_bytes: int = Field(ge=0)
    peak_ram_bytes: int = Field(ge=0)
    pcie_bytes: int = Field(ge=0)
    cpu_utilisation_percent: float = Field(ge=0)
    gpu_utilisation_percent: float = Field(ge=0)
    enabled_techniques: list[str] = Field(default_factory=list)
    unsupported_techniques: list[str] = Field(default_factory=list)
    explanation: list[str] = Field(default_factory=list)


class CandidateSelection(StrictModel):
    selected_plan_id: str
    selected_utility: float
    ranking: list[dict[str, Any]]
    reason: str


def critical_path(tasks: list[CriticalTask]) -> tuple[float, list[str]]:
    """Return DAG completion time; independent tasks overlap instead of being summed."""

    by_id = {task.task_id: task for task in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("critical-path task IDs must be unique")
    visiting: set[str] = set()
    memo: dict[str, tuple[float, list[str]]] = {}

    def finish(task_id: str) -> tuple[float, list[str]]:
        if task_id in memo:
            return memo[task_id]
        if task_id in visiting:
            raise ValueError("critical-path graph contains a cycle")
        task = by_id.get(task_id)
        if task is None:
            raise ValueError(f"critical-path dependency {task_id} is not declared")
        if task.duration_ms < 0 or not math.isfinite(task.duration_ms):
            raise ValueError("critical-path task durations must be finite and non-negative")
        visiting.add(task_id)
        predecessors = [finish(dependency) for dependency in task.dependencies]
        visiting.remove(task_id)
        if predecessors:
            predecessor_time, predecessor_path = max(predecessors, key=lambda item: item[0])
        else:
            predecessor_time, predecessor_path = 0.0, []
        result = predecessor_time + task.duration_ms, [*predecessor_path, task_id]
        memo[task_id] = result
        return result

    if not tasks:
        return 0.0, []
    return max((finish(task.task_id) for task in tasks), key=lambda item: item[0])


class MeasuredCostModel:
    def __init__(
        self,
        kernels: list[MeasuredKernelPoint],
        transfers: list[TransferPoint],
    ) -> None:
        self.kernels = kernels
        self.transfers = transfers

    def kernel_ms(self, operation: str, device: str, shape: list[int]) -> float:
        candidates = [
            point
            for point in self.kernels
            if point.operation == operation and point.device == device
        ]
        if not candidates:
            raise ValueError(f"no measured kernel point for {operation} on {device}")
        target_work = max(math.prod(shape), 1)
        point = min(
            candidates,
            key=lambda item: (
                abs(math.log(max(math.prod(item.shape), 1) / target_work)),
                item.median_ms,
            ),
        )
        scale = target_work / max(math.prod(point.shape), 1)
        # Nearby measured shapes interpolate linearly in work; extrapolation is explicit.
        return point.median_ms * scale

    def transfer_ms(self, direction: str, memory_kind: str, payload_bytes: int) -> float:
        if payload_bytes <= 0:
            return 0.0
        candidates = [
            point
            for point in self.transfers
            if point.direction == direction and point.memory_kind == memory_kind
        ]
        if not candidates:
            raise ValueError(f"no measured {direction} {memory_kind} transfer point")
        point = min(candidates, key=lambda item: abs(item.payload_bytes - payload_bytes))
        fixed_ms = max(
            point.median_ms - point.payload_bytes / point.effective_bandwidth_bytes_s * 1000,
            0.0,
        )
        return fixed_ms + payload_bytes / point.effective_bandwidth_bytes_s * 1000

    def estimate(
        self,
        *,
        compute_tasks: list[tuple[str, str, list[int]]],
        transfer_tasks: list[tuple[str, str, int]],
        dequantization_ms: float,
        synchronization_ms: float,
        reduction_ms: float,
        cache_miss_ms: float,
        contention_ms: float,
        asynchronous: bool,
    ) -> CostBreakdown:
        compute_ms = sum(self.kernel_ms(*task) for task in compute_tasks)
        transfer_ms = sum(self.transfer_ms(*task) for task in transfer_tasks)
        serial_tail = synchronization_ms + reduction_ms + cache_miss_ms + contention_ms
        if asynchronous:
            tasks = [
                CriticalTask("compute", compute_ms),
                CriticalTask("transfer", transfer_ms),
                CriticalTask("dequantize", dequantization_ms, ("transfer",)),
                CriticalTask("tail", serial_tail, ("compute", "dequantize")),
            ]
            completion, path = critical_path(tasks)
        else:
            completion = compute_ms + transfer_ms + dequantization_ms + serial_tail
            path = ["transfer", "dequantize", "compute", "tail"]
        return CostBreakdown(
            transfer_ms=transfer_ms,
            dequantization_ms=dequantization_ms,
            compute_ms=compute_ms,
            synchronization_ms=synchronization_ms,
            reduction_ms=reduction_ms,
            cache_miss_ms=cache_miss_ms,
            contention_ms=contention_ms,
            completion_ms=completion,
            critical_path=path,
        )


def _utility(
    candidate: CandidateEstimate,
    baseline: CandidateEstimate,
    *,
    objective: UtilityObjective,
    maximum_interactive_p95_increase_fraction: float,
    maximum_other_regression_fraction: float,
) -> tuple[float, str | None]:
    if candidate.unsupported_techniques:
        return -math.inf, "candidate requests unsupported techniques"
    interactive_limit = baseline.interactive_p95_ms * (
        1 + maximum_interactive_p95_increase_fraction
    )
    if candidate.interactive_p95_ms > interactive_limit:
        return -math.inf, "interactive p95 constraint exceeded"
    floor = 1 - maximum_other_regression_fraction
    if (
        objective != UtilityObjective.MAXIMUM_DECODE_THROUGHPUT
        and candidate.decode_tokens_per_second < baseline.decode_tokens_per_second * floor
    ):
        return -math.inf, "decode non-regression constraint exceeded"
    if (
        objective != UtilityObjective.MINIMUM_TIME_TO_FIRST_TOKEN
        and candidate.time_to_first_token_ms
        > baseline.time_to_first_token_ms * (1 + maximum_other_regression_fraction)
    ):
        return -math.inf, "prefill non-regression constraint exceeded"
    if objective == UtilityObjective.MAXIMUM_DECODE_THROUGHPUT:
        return candidate.decode_tokens_per_second / baseline.decode_tokens_per_second - 1, None
    if objective == UtilityObjective.MINIMUM_TIME_TO_FIRST_TOKEN:
        return 1 - candidate.time_to_first_token_ms / baseline.time_to_first_token_ms, None
    if objective == UtilityObjective.MAXIMUM_MIXED_VERIFIED_THROUGHPUT:
        return (
            candidate.mixed_verified_tokens_per_second / baseline.mixed_verified_tokens_per_second
            - 1,
            None,
        )
    if candidate.time_to_first_token_ms > baseline.time_to_first_token_ms * 1.05:
        return -math.inf, "latency constraint for minimum-VRAM objective exceeded"
    if baseline.peak_vram_bytes == 0:
        return -math.inf, "baseline VRAM is unavailable"
    return 1 - candidate.peak_vram_bytes / baseline.peak_vram_bytes, None


def select_positive_utility(
    candidates: list[CandidateEstimate],
    *,
    baseline_plan_id: str,
    objective: UtilityObjective,
    maximum_interactive_p95_increase_fraction: float = 0.05,
    maximum_other_regression_fraction: float = 0.10,
) -> CandidateSelection:
    by_id = {candidate.plan_id: candidate for candidate in candidates}
    baseline = by_id.get(baseline_plan_id)
    if baseline is None:
        raise ValueError("baseline plan is absent from candidate set")
    ranking: list[dict[str, Any]] = []
    for candidate in candidates:
        utility, rejection = _utility(
            candidate,
            baseline,
            objective=objective,
            maximum_interactive_p95_increase_fraction=(maximum_interactive_p95_increase_fraction),
            maximum_other_regression_fraction=maximum_other_regression_fraction,
        )
        ranking.append(
            {
                "plan_id": candidate.plan_id,
                "utility": utility if math.isfinite(utility) else None,
                "eligible": rejection is None,
                "rejection_reason": rejection,
                "enabled_techniques": candidate.enabled_techniques,
            }
        )
    ranking.sort(
        key=lambda row: (
            bool(row["eligible"]),
            float(row["utility"]) if row["utility"] is not None else -math.inf,
        ),
        reverse=True,
    )
    eligible = [row for row in ranking if row["eligible"] and float(row["utility"]) > 0]
    selected = (
        eligible[0]
        if eligible
        else next(row for row in ranking if row["plan_id"] == baseline_plan_id)
    )
    return CandidateSelection(
        selected_plan_id=str(selected["plan_id"]),
        selected_utility=float(selected["utility"] or 0.0),
        ranking=ranking,
        reason=(
            "highest positive predicted utility satisfying workload constraints"
            if eligible
            else "baseline retained because no candidate earned positive utility"
        ),
    )


def prediction_quality(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    valid = [
        row
        for row in rows
        if isinstance(row.get("predicted_ms"), (int, float))
        and isinstance(row.get("measured_ms"), (int, float))
        and float(row["measured_ms"]) > 0
    ]
    if not valid:
        return {
            "observation_count": 0,
            "mean_absolute_percentage_error": None,
            "comparison_group_count": 0,
            "informative_pair_count": 0,
            "tied_prediction_pair_count": 0,
            "ranking_agreement_fraction": None,
        }
    errors = [
        abs(float(row["predicted_ms"]) - float(row["measured_ms"])) / float(row["measured_ms"])
        for row in valid
    ]
    pairs = 0
    agreements = 0
    tied_predictions = 0
    groups = {str(row.get("comparison_group", row.get("phase", "all"))) for row in valid}
    for group in groups:
        grouped = [
            row
            for row in valid
            if str(row.get("comparison_group", row.get("phase", "all"))) == group
        ]
        for left in range(len(grouped)):
            for right in range(left + 1, len(grouped)):
                predicted_delta = float(grouped[left]["predicted_ms"]) - float(
                    grouped[right]["predicted_ms"]
                )
                measured_delta = float(grouped[left]["measured_ms"]) - float(
                    grouped[right]["measured_ms"]
                )
                if math.isclose(predicted_delta, 0.0, rel_tol=1e-12, abs_tol=1e-12):
                    tied_predictions += 1
                    continue
                pairs += 1
                agreements += int(
                    math.isclose(measured_delta, 0.0, rel_tol=1e-12, abs_tol=1e-12)
                    or (predicted_delta < 0) == (measured_delta < 0)
                )
    return {
        "observation_count": len(valid),
        "mean_absolute_percentage_error": mean(errors),
        "comparison_group_count": len(groups),
        "informative_pair_count": pairs,
        "tied_prediction_pair_count": tied_predictions,
        "ranking_agreement_fraction": agreements / pairs if pairs else None,
    }


def planner_regret_fraction(
    measured_utility_by_plan: dict[str, float], selected_plan_id: str
) -> float:
    if selected_plan_id not in measured_utility_by_plan:
        raise ValueError("selected plan has no measured utility")
    best = max(measured_utility_by_plan.values())
    selected = measured_utility_by_plan[selected_plan_id]
    return (best - selected) / best if best > 0 else 0.0
