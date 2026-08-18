"""Canonical latency-constrained serving objective for Experiment 023."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from swarm_inference.experiments.experiment_022.models import Inventory

from .freeze import FROZEN_CONSTANTS
from .models import E023Plan, NetworkMode
from .replica_memory import abstract_node_cost
from .serving_engine import CONCURRENCY_LEVELS, ServingRun


class NoPrimarySLOEligibleConcurrency(RuntimeError):
    """Raised when a complete ladder contains no primary-SLO-eligible point."""


class _PlannerScoringContext(Protocol):
    inventory: Inventory

    def serving_runs(
        self,
        requests: list[tuple[E023Plan, NetworkMode, int]],
    ) -> list[ServingRun]: ...


class _ServingMeasurement(Protocol):
    status: str
    concurrency: int
    target_rows_per_second: float
    p50_pass_latency_ms: float
    p95_pass_latency_ms: float


@dataclass(frozen=True, slots=True)
class SLOScore:
    latency_budget_ms: float
    selected_concurrency: int
    target_rows_per_second: float
    p50_pass_latency_ms: float
    p95_pass_latency_ms: float
    abstract_node_cost: float
    rows_per_second_per_abstract_cost: float


def score_runs_under_latency_budget(
    runs: Mapping[int, _ServingMeasurement] | Sequence[_ServingMeasurement],
    *,
    latency_budget_ms: float,
    abstract_cost: float,
) -> SLOScore:
    """Score one complete frozen ladder under one hard latency budget."""

    by_concurrency = (
        dict(runs)
        if isinstance(runs, Mapping)
        else {run.concurrency: run for run in runs}
    )
    required = tuple(int(value) for value in CONCURRENCY_LEVELS)
    if set(by_concurrency) != set(required):
        raise RuntimeError(
            "MODEL_INVALID: primary SLO scorer requires exactly concurrency "
            + ",".join(str(value) for value in required)
        )
    if any(by_concurrency[value].status != "PASS" for value in required):
        raise RuntimeError("MODEL_INVALID: primary SLO serving ladder is incomplete")
    if abstract_cost <= 0.0:
        raise RuntimeError("MODEL_INVALID: primary SLO abstract cost is not positive")

    eligible = [
        by_concurrency[value]
        for value in required
        if by_concurrency[value].p95_pass_latency_ms <= latency_budget_ms
    ]
    if not eligible:
        raise NoPrimarySLOEligibleConcurrency(
            "NO_PRIMARY_SLO_ELIGIBLE_CONCURRENCY"
        )
    maximum = max(run.target_rows_per_second for run in eligible)
    near_maximum = [
        run
        for run in eligible
        if run.target_rows_per_second >= 0.995 * maximum
    ]
    selected = min(near_maximum, key=lambda run: run.concurrency)
    throughput = float(selected.target_rows_per_second)
    return SLOScore(
        latency_budget_ms=float(latency_budget_ms),
        selected_concurrency=int(selected.concurrency),
        target_rows_per_second=throughput,
        p50_pass_latency_ms=float(selected.p50_pass_latency_ms),
        p95_pass_latency_ms=float(selected.p95_pass_latency_ms),
        abstract_node_cost=float(abstract_cost),
        rows_per_second_per_abstract_cost=throughput / float(abstract_cost),
    )


def score_recorded_runs_under_primary_slo(
    runs: Mapping[int, _ServingMeasurement] | Sequence[_ServingMeasurement],
    *,
    u_strong_c1_p95_ms: float,
    abstract_cost: float,
) -> SLOScore:
    """Apply the canonical primary SLO rule to already-recorded run rows."""

    multiplier = float(FROZEN_CONSTANTS["primary_latency_multiplier"])
    return score_runs_under_latency_budget(
        runs,
        latency_budget_ms=multiplier * float(u_strong_c1_p95_ms),
        abstract_cost=abstract_cost,
    )


def score_plan_under_primary_slo(
    planner_context: _PlannerScoringContext,
    plan: E023Plan,
    *,
    u_strong_c1_p95_ms: float,
) -> SLOScore:
    """Evaluate and score a plan on the complete frozen SHARED_NIC ladder."""

    requests = [
        (plan, NetworkMode.SHARED_NIC, concurrency)
        for concurrency in CONCURRENCY_LEVELS
    ]
    runs = planner_context.serving_runs(requests)
    return score_recorded_runs_under_primary_slo(
        runs,
        u_strong_c1_p95_ms=u_strong_c1_p95_ms,
        abstract_cost=abstract_node_cost(plan, planner_context.inventory),
    )


__all__ = [
    "NoPrimarySLOEligibleConcurrency",
    "SLOScore",
    "score_plan_under_primary_slo",
    "score_recorded_runs_under_primary_slo",
    "score_runs_under_latency_budget",
]
