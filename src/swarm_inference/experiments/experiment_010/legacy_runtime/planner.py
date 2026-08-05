"""LEGACY_FROZEN planner retained only for Experiment 010 reproduction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from swarm_inference.experiments.experiment_010.schemas import (
    ExecutionStrategy,
    PhasePlan,
    PlannerCandidate,
    PlannerObjective,
    ServicePhase,
)
from swarm_inference.planner import (
    HeterogeneousPlanner,
    NodeRole,
    NonDegradationPolicy,
    RoleCandidate,
    UtilityNormalisation,
)
from swarm_inference.planner import (
    PlannerObjective as Experiment007Objective,
)
from swarm_inference.worker.abi import ResultClassification


@dataclass(frozen=True, slots=True)
class PlannerSelection:
    plan: PhasePlan
    ranking: tuple[PlannerCandidate, ...]
    regret: dict[str, Any] | None
    experiment_007_evaluations: tuple[dict[str, Any], ...]


_STRATEGY_ROLES = {
    ExecutionStrategy.LOCAL_WHOLE_EXPERT: NodeRole.CRITICAL_PATH_STAGE,
    ExecutionStrategy.REMOTE_WHOLE_EXPERT: NodeRole.MOE_EXPERT,
    ExecutionStrategy.EQUAL_MICROSHARDS: NodeRole.TENSOR_RANK,
    ExecutionStrategy.ASYMMETRIC_MICROSHARDS: NodeRole.TENSOR_RANK,
    ExecutionStrategy.COALESCED_MICROSHARDS: NodeRole.TENSOR_RANK,
    ExecutionStrategy.CACHE_ONLY: NodeRole.SHARD_CACHE,
    ExecutionStrategy.STORAGE_ONLY: NodeRole.SHARD_CACHE,
    ExecutionStrategy.BACKGROUND_INFERENCE: NodeRole.BACKGROUND_INFERENCE,
    ExecutionStrategy.VERIFICATION: NodeRole.INTEGRITY_AUDIT,
    ExecutionStrategy.IDLE: NodeRole.IDLE,
}

_OBJECTIVE_MAP = {
    PlannerObjective.MAX_DECODE_THROUGHPUT: Experiment007Objective.AGGREGATE_THROUGHPUT,
    PlannerObjective.MIN_TTFT: Experiment007Objective.INTERACTIVE_LATENCY,
    PlannerObjective.MAX_VERIFIED_AGGREGATE_THROUGHPUT: (
        Experiment007Objective.AGGREGATE_THROUGHPUT
    ),
    PlannerObjective.MIN_NETWORK_BYTES: Experiment007Objective.BALANCED,
    PlannerObjective.MIN_ENERGY_PER_VERIFIED_TOKEN: Experiment007Objective.ENERGY_EFFICIENCY,
    PlannerObjective.MAX_CAPACITY_SUBJECT_TO_LATENCY: Experiment007Objective.BALANCED,
}


class PositiveUtilityPlanner:
    """Fail-closed admission using measured/calibrated marginal utility.

    Experiment 007's role vocabulary and non-degradation principle are retained;
    this class adds Experiment 010's expert execution strategies, explicit lower
    confidence-bound gate, phase separation, and capacity-only exception.
    """

    def select(
        self,
        candidates: list[PlannerCandidate],
        *,
        phase: ServicePhase | str,
        objective: PlannerObjective | str,
        measured_utilities: dict[str, float] | None = None,
    ) -> PlannerSelection:
        selected_phase = ServicePhase(phase)
        selected_objective = PlannerObjective(objective)
        eligible_phase = [
            item
            for item in candidates
            if item.phase == selected_phase and item.objective == selected_objective
        ]
        if not eligible_phase:
            raise ValueError("planner has no candidates for the requested phase/objective")
        legacy = self._experiment_007_evaluations(eligible_phase, selected_objective)
        legacy_by_id = {item["candidate_id"]: item for item in legacy}
        evaluated = sorted(
            eligible_phase,
            key=lambda item: (
                self._admissible(item, legacy_by_id[item.candidate_id]),
                item.capacity_required,
                item.predicted_utility,
                item.candidate_id,
            ),
            reverse=True,
        )
        positive = [
            item
            for item in evaluated
            if item.strategy != ExecutionStrategy.IDLE
            and self._admissible(item, legacy_by_id[item.candidate_id])
        ]
        capacity = [
            item
            for item in evaluated
            if item.capacity_required
            and item.reliability_gate
            and item.correctness_gate
            and item.slo_gate
        ]
        capacity_exception = False
        if positive:
            winner = positive[0]
        elif capacity:
            winner = capacity[0]
            capacity_exception = True
        else:
            idle = [item for item in evaluated if item.strategy == ExecutionStrategy.IDLE]
            if not idle:
                raise ValueError("planner needs an idle candidate for fail-closed rejection")
            winner = idle[0]
        rejected = []
        for item in evaluated:
            if item.candidate_id == winner.candidate_id:
                continue
            rejected.append(
                {
                    "candidate_id": item.candidate_id,
                    "strategy": item.strategy.value,
                    "workers": item.workers,
                    "predicted_utility": item.predicted_utility,
                    "lower_confidence_bound": item.lower_confidence_bound,
                    "experiment_007_role": legacy_by_id[item.candidate_id]["role"],
                    "experiment_007_eligible": legacy_by_id[item.candidate_id]["eligible"],
                    "reasons": self._rejection_reasons(item, legacy_by_id[item.candidate_id]),
                }
            )
        explanation = list(winner.explanation)
        winner_legacy = legacy_by_id[winner.candidate_id]
        explanation.append(
            "Experiment 007 HeterogeneousPlanner evaluation: "
            f"role={winner_legacy['role']}, eligible={winner_legacy['eligible']}, "
            f"utility={winner_legacy['utility_score']:.6g}"
        )
        if capacity_exception:
            explanation.append(
                "admitted only because the worker is required to host the model; "
                "positive latency utility was not claimed"
            )
        elif winner.strategy == ExecutionStrategy.IDLE:
            explanation.append(
                "idle selected because no token-critical candidate had a positive lower bound"
            )
        else:
            explanation.append("selected with positive marginal-utility lower confidence bound")
        plan = PhasePlan(
            phase=selected_phase,
            objective=selected_objective,
            selected_candidate_id=winner.candidate_id,
            selected_strategy=winner.strategy,
            selected_workers=winner.workers,
            rejected=rejected,
            capacity_exception=capacity_exception,
            explanation=explanation,
        )
        regret = (
            planner_regret(measured_utilities, winner.candidate_id)
            if measured_utilities is not None
            else None
        )
        return PlannerSelection(
            plan=plan,
            ranking=tuple(evaluated),
            regret=regret,
            experiment_007_evaluations=tuple(legacy),
        )

    @staticmethod
    def _admissible(candidate: PlannerCandidate, legacy: dict[str, Any]) -> bool:
        if candidate.strategy == ExecutionStrategy.IDLE:
            return candidate.predicted_utility >= 0
        confidence_positive = (
            candidate.lower_confidence_bound is None or candidate.lower_confidence_bound > 0
        )
        return bool(
            candidate.predicted_utility > 0
            and confidence_positive
            and candidate.reliability_gate
            and candidate.correctness_gate
            and candidate.slo_gate
            and legacy["eligible"]
        )

    @staticmethod
    def _rejection_reasons(candidate: PlannerCandidate, legacy: dict[str, Any]) -> list[str]:
        reasons = []
        if candidate.predicted_utility <= 0:
            reasons.append("non-positive predicted marginal utility")
        if candidate.lower_confidence_bound is not None and candidate.lower_confidence_bound <= 0:
            reasons.append("utility lower confidence bound is non-positive")
        if not candidate.reliability_gate:
            reasons.append("reliability gate failed")
        if not candidate.correctness_gate:
            reasons.append("correctness gate failed")
        if not candidate.slo_gate:
            reasons.append("workload SLO gate failed")
        if not legacy["eligible"]:
            reasons.append(
                "Experiment 007 non-degradation gate: "
                + str(legacy.get("rejection_reason") or "ineligible")
            )
        return reasons or ["ranked below selected candidate"]

    @staticmethod
    def _experiment_007_evaluations(
        candidates: list[PlannerCandidate], objective: PlannerObjective
    ) -> list[dict[str, Any]]:
        """Evaluate Experiment 010 strategies through Experiment 007's planner.

        Experiment 010 retains its stricter positive lower-confidence-bound and
        capacity rules. This adapter makes the pre-existing role utility and
        non-degradation gates an additional, executable admission layer rather
        than merely copying their vocabulary.
        """

        local = next(
            (item for item in candidates if item.strategy == ExecutionStrategy.LOCAL_WHOLE_EXPERT),
            None,
        )
        baseline_latency = float(local.latency_ms or 1.0) if local else 1.0
        baseline_throughput = float(local.throughput or 1.0) if local else 1.0
        largest_transfer = max((item.network_bytes or 0 for item in candidates), default=0)
        largest_utility = max((abs(item.predicted_utility) for item in candidates), default=1.0)
        planner = HeterogeneousPlanner(
            policy=NonDegradationPolicy(),
            normalisation=UtilityNormalisation(
                verified_tps_scale=max(largest_utility, 1.0),
                interactive_latency_ms_scale=max(baseline_latency, 1e-9),
                transfer_bytes_scale=max(float(largest_transfer), 1.0),
                failure_cost_scale=1.0,
                verification_cost_scale=1.0,
                admission_risk_scale=1.0,
            ),
            maximum_regret_fraction=0.05,
        )
        rows: list[dict[str, Any]] = []
        for candidate in candidates:
            compatible = bool(
                candidate.reliability_gate and candidate.correctness_gate and candidate.slo_gate
            )
            uncertainty = (
                0.20
                if candidate.lower_confidence_bound is None
                else min(
                    1.0,
                    abs(candidate.predicted_utility - candidate.lower_confidence_bound)
                    / max(abs(candidate.predicted_utility), 1e-9),
                )
            )
            role = RoleCandidate(
                node_id=f"experiment-010-{candidate.candidate_id}",
                role=_STRATEGY_ROLES[candidate.strategy],
                expected_verified_token_gain=candidate.predicted_utility,
                predicted_p95_latency_delta_ms=(
                    max(float(candidate.latency_ms) - baseline_latency, 0.0)
                    if candidate.latency_ms is not None
                    else 0.0
                ),
                predicted_interactive_throughput_delta=(
                    float(candidate.throughput) - baseline_throughput
                    if candidate.throughput is not None
                    else candidate.predicted_utility
                ),
                predicted_memory_bytes=0,
                predicted_transfer_bytes=int(candidate.network_bytes or 0),
                predicted_failure_cost=0.0 if candidate.reliability_gate else 1.0,
                verification_cost=0.0 if candidate.correctness_gate else 1.0,
                admission_risk=0.0 if candidate.slo_gate else 1.0,
                confidence_fraction=uncertainty,
                compatible=compatible,
                compatibility_reason=(
                    None if compatible else "Experiment 010 correctness/reliability/SLO gate failed"
                ),
                classification=ResultClassification.PROJECTED_DEVICE_PROFILE,
                evidence={
                    "experiment": "010",
                    "candidate_id": candidate.candidate_id,
                    "source": "measured or calibrated fields retained on PlannerCandidate",
                },
            )
            evaluation = planner.evaluate_candidates(
                [role],
                objective=_OBJECTIVE_MAP[objective],
                baseline_p95_latency_ms=baseline_latency,
                baseline_interactive_throughput=baseline_throughput,
                maximum_memory_bytes=2**63 - 1,
            )[0]
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    **evaluation.model_dump(mode="json"),
                }
            )
        return rows


def planner_regret(
    measured_utilities: dict[str, float], selected_candidate_id: str
) -> dict[str, Any]:
    if selected_candidate_id not in measured_utilities:
        raise ValueError("selected candidate has no held-out measured utility")
    best_id, best_utility = max(measured_utilities.items(), key=lambda item: item[1])
    selected = measured_utilities[selected_candidate_id]
    regret = best_utility - selected
    denominator = abs(best_utility) if best_utility else 1.0
    fraction = max(0.0, regret / denominator)
    return {
        "selected_candidate_id": selected_candidate_id,
        "selected_measured_utility": selected,
        "best_candidate_id": best_id,
        "best_measured_utility": best_utility,
        "regret": regret,
        "regret_fraction": fraction,
        "passes": fraction <= 0.05,
    }


def worker_marginal_utility(
    cluster_with_worker: list[float], cluster_without_worker: list[float]
) -> dict[str, Any]:
    if not cluster_with_worker or not cluster_without_worker:
        raise ValueError("marginal utility requires both measured sample sets")
    if len(cluster_with_worker) != len(cluster_without_worker):
        raise ValueError("paired marginal utility sample counts differ")
    differences = [
        with_worker - without
        for with_worker, without in zip(cluster_with_worker, cluster_without_worker, strict=True)
    ]
    mean = sum(differences) / len(differences)
    if len(differences) == 1:
        half_width = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in differences) / (len(differences) - 1)
        half_width = 1.96 * (variance / len(differences)) ** 0.5
    return {
        "samples": differences,
        "mean_utility": mean,
        "confidence_interval_95": [mean - half_width, mean + half_width],
        "lower_confidence_bound_positive": mean - half_width > 0,
    }
