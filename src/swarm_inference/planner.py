"""Measured marginal-utility planner for heterogeneous worker roles."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import Field

from swarm_inference.config.models import StrictModel
from swarm_inference.worker.abi import ResultClassification


class NodeRole(StrEnum):
    CRITICAL_PATH_STAGE = "critical_path_stage"
    TENSOR_RANK = "tensor_rank"
    SPECULATIVE_DRAFT = "speculative_draft"
    MOE_EXPERT = "moe_expert"
    STAGE_REPLICA = "stage_replica"
    BACKGROUND_INFERENCE = "background_inference"
    INTEGRITY_AUDIT = "integrity_audit"
    SHARD_CACHE = "shard_cache"
    IDLE = "idle"


class PlannerObjective(StrEnum):
    INTERACTIVE_LATENCY = "interactive_latency"
    AGGREGATE_THROUGHPUT = "aggregate_throughput"
    BALANCED = "balanced"
    ENERGY_EFFICIENCY = "energy_efficiency"


class NonDegradationPolicy(StrictModel):
    maximum_interactive_p95_increase_fraction: float = Field(default=0.05, ge=0)
    maximum_interactive_throughput_decrease_fraction: float = Field(default=0.05, ge=0)


class UtilityNormalisation(StrictModel):
    """Explicit conversion scales; utility components are dimensionless."""

    verified_tps_scale: float = Field(gt=0)
    interactive_latency_ms_scale: float = Field(gt=0)
    transfer_bytes_scale: float = Field(gt=0)
    failure_cost_scale: float = Field(gt=0)
    verification_cost_scale: float = Field(gt=0)
    admission_risk_scale: float = Field(gt=0)
    energy_watts_scale: float = Field(default=100.0, gt=0)


class RoleCandidate(StrictModel):
    node_id: str
    role: NodeRole
    expected_verified_token_gain: float
    predicted_p95_latency_delta_ms: float
    predicted_interactive_throughput_delta: float
    predicted_memory_bytes: int = Field(ge=0)
    predicted_transfer_bytes: int = Field(ge=0)
    predicted_failure_cost: float = Field(ge=0)
    verification_cost: float = Field(ge=0)
    admission_risk: float = Field(ge=0)
    expected_power_watts: float = Field(default=0.0, ge=0)
    model_acquisition_seconds: float = Field(default=0.0, ge=0)
    model_conversion_seconds: float = Field(default=0.0, ge=0)
    model_load_seconds: float = Field(default=0.0, ge=0)
    warmup_seconds: float = Field(default=0.0, ge=0)
    failure_exposure: float = Field(default=0.0, ge=0)
    availability_requirement: float = Field(default=1.0, ge=0, le=1)
    confidence_fraction: float = Field(default=0.20, ge=0)
    compatible: bool = True
    compatibility_reason: str | None = None
    classification: ResultClassification
    evidence: dict[str, Any] = Field(default_factory=dict)


class UtilityComponents(StrictModel):
    expected_verified_token_gain: float
    interactive_latency_penalty: float
    transfer_cost: float
    failure_recovery_cost: float
    verification_cost: float
    admission_risk: float
    energy_cost: float
    objective_weights: dict[str, float]


class RoleEvaluation(StrictModel):
    node_id: str
    role: str
    predicted_verified_tps_gain: float
    predicted_p95_latency_delta_ms: float
    predicted_memory_bytes: int
    predicted_transfer_bytes: int
    predicted_failure_cost: float
    utility_score: float
    confidence_interval: tuple[float, float]
    eligible: bool
    rejection_reason: str | None
    objective: PlannerObjective
    components: UtilityComponents
    classification: ResultClassification


class PlannerDecision(StrictModel):
    node_id: str
    objective: PlannerObjective
    selected_role: NodeRole
    selected_utility: float
    ranking: list[RoleEvaluation]
    reason: str
    route_generation: int = Field(ge=0)
    decided_at_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class CanaryMeasurement(StrictModel):
    node_id: str
    role: NodeRole
    measured_verified_tps_gain: float
    measured_p95_latency_delta_ms: float
    measured_interactive_throughput_delta: float
    baseline_p95_latency_ms: float = Field(gt=0)
    baseline_interactive_throughput: float = Field(gt=0)
    measured_utility: float
    passed: bool
    classification: ResultClassification


def _objective_weights(objective: PlannerObjective) -> dict[str, float]:
    if objective == PlannerObjective.INTERACTIVE_LATENCY:
        return {"gain": 0.5, "latency": 2.0, "transfer": 0.5, "other": 1.0, "energy": 0.1}
    if objective == PlannerObjective.AGGREGATE_THROUGHPUT:
        return {"gain": 2.0, "latency": 0.5, "transfer": 0.5, "other": 1.0, "energy": 0.1}
    if objective == PlannerObjective.ENERGY_EFFICIENCY:
        return {"gain": 1.0, "latency": 0.5, "transfer": 0.5, "other": 1.0, "energy": 2.0}
    return {"gain": 1.0, "latency": 1.0, "transfer": 0.5, "other": 1.0, "energy": 0.25}


def evaluate_role(
    candidate: RoleCandidate,
    *,
    objective: PlannerObjective,
    normalisation: UtilityNormalisation,
    policy: NonDegradationPolicy,
    baseline_p95_latency_ms: float,
    baseline_interactive_throughput: float,
    maximum_memory_bytes: int,
) -> RoleEvaluation:
    if baseline_p95_latency_ms <= 0 or baseline_interactive_throughput <= 0:
        raise ValueError("baseline latency and throughput must be positive")
    weights = _objective_weights(objective)
    components = UtilityComponents(
        expected_verified_token_gain=(
            candidate.expected_verified_token_gain / normalisation.verified_tps_scale
        ),
        interactive_latency_penalty=(
            max(candidate.predicted_p95_latency_delta_ms, 0.0)
            / normalisation.interactive_latency_ms_scale
        ),
        transfer_cost=candidate.predicted_transfer_bytes / normalisation.transfer_bytes_scale,
        failure_recovery_cost=(candidate.predicted_failure_cost / normalisation.failure_cost_scale),
        verification_cost=candidate.verification_cost / normalisation.verification_cost_scale,
        admission_risk=candidate.admission_risk / normalisation.admission_risk_scale,
        energy_cost=candidate.expected_power_watts / normalisation.energy_watts_scale,
        objective_weights=weights,
    )
    score = (
        weights["gain"] * components.expected_verified_token_gain
        - weights["latency"] * components.interactive_latency_penalty
        - weights["transfer"] * components.transfer_cost
        - weights["other"]
        * (
            components.failure_recovery_cost
            + components.verification_cost
            + components.admission_risk
        )
        - weights["energy"] * components.energy_cost
    )
    if candidate.role == NodeRole.IDLE:
        score = 0.0
    p95_fraction = candidate.predicted_p95_latency_delta_ms / baseline_p95_latency_ms
    throughput_fraction = (
        candidate.predicted_interactive_throughput_delta / baseline_interactive_throughput
    )
    rejection: str | None = None
    if not candidate.compatible:
        rejection = candidate.compatibility_reason or "no compatible backend artifact"
    elif candidate.predicted_memory_bytes > maximum_memory_bytes:
        rejection = "insufficient measured memory"
    elif p95_fraction > policy.maximum_interactive_p95_increase_fraction:
        rejection = "predicted interactive p95 non-degradation limit exceeded"
    elif throughput_fraction < -policy.maximum_interactive_throughput_decrease_fraction:
        rejection = "predicted interactive throughput non-degradation limit exceeded"
    elif candidate.role != NodeRole.IDLE and score <= 0:
        rejection = "non-positive marginal utility"
    eligible = rejection is None
    uncertainty = abs(score) * candidate.confidence_fraction
    if not math.isfinite(uncertainty):
        raise ValueError("role utility uncertainty must be finite")
    return RoleEvaluation(
        node_id=candidate.node_id,
        role=candidate.role.value,
        predicted_verified_tps_gain=candidate.expected_verified_token_gain,
        predicted_p95_latency_delta_ms=candidate.predicted_p95_latency_delta_ms,
        predicted_memory_bytes=candidate.predicted_memory_bytes,
        predicted_transfer_bytes=candidate.predicted_transfer_bytes,
        predicted_failure_cost=candidate.predicted_failure_cost,
        utility_score=score,
        confidence_interval=(score - uncertainty, score + uncertainty),
        eligible=eligible,
        rejection_reason=rejection,
        objective=objective,
        components=components,
        classification=candidate.classification,
    )


class HeterogeneousPlanner:
    def __init__(
        self,
        *,
        policy: NonDegradationPolicy,
        normalisation: UtilityNormalisation,
        maximum_regret_fraction: float = 0.10,
    ) -> None:
        if maximum_regret_fraction < 0:
            raise ValueError("maximum regret fraction cannot be negative")
        self.policy = policy
        self.normalisation = normalisation
        self.maximum_regret_fraction = maximum_regret_fraction
        self.decisions: list[PlannerDecision] = []
        self.measurements: list[CanaryMeasurement] = []
        self._route_generations: dict[str, int] = {}
        self._adjustments: dict[tuple[str, NodeRole], tuple[float, float, float]] = {}

    def evaluate_candidates(
        self,
        candidates: list[RoleCandidate],
        *,
        objective: PlannerObjective,
        baseline_p95_latency_ms: float,
        baseline_interactive_throughput: float,
        maximum_memory_bytes: int,
    ) -> list[RoleEvaluation]:
        if not candidates:
            raise ValueError("planner requires at least one candidate")
        node_ids = {item.node_id for item in candidates}
        if len(node_ids) != 1:
            raise ValueError("one planner evaluation may describe only one node")
        adjusted: list[RoleCandidate] = []
        for candidate in candidates:
            update = self._adjustments.get((candidate.node_id, candidate.role))
            if update is None:
                adjusted.append(candidate)
            else:
                gain, latency, throughput = update
                adjusted.append(
                    candidate.model_copy(
                        update={
                            "expected_verified_token_gain": gain,
                            "predicted_p95_latency_delta_ms": latency,
                            "predicted_interactive_throughput_delta": throughput,
                            "confidence_fraction": max(candidate.confidence_fraction / 2, 0.02),
                        }
                    )
                )
        evaluations = [
            evaluate_role(
                item,
                objective=objective,
                normalisation=self.normalisation,
                policy=self.policy,
                baseline_p95_latency_ms=baseline_p95_latency_ms,
                baseline_interactive_throughput=baseline_interactive_throughput,
                maximum_memory_bytes=maximum_memory_bytes,
            )
            for item in adjusted
        ]
        return sorted(
            evaluations,
            key=lambda item: (item.eligible, item.utility_score, item.role == NodeRole.IDLE.value),
            reverse=True,
        )

    def select(
        self,
        candidates: list[RoleCandidate],
        *,
        objective: PlannerObjective,
        baseline_p95_latency_ms: float,
        baseline_interactive_throughput: float,
        maximum_memory_bytes: int,
    ) -> PlannerDecision:
        ranking = self.evaluate_candidates(
            candidates,
            objective=objective,
            baseline_p95_latency_ms=baseline_p95_latency_ms,
            baseline_interactive_throughput=baseline_interactive_throughput,
            maximum_memory_bytes=maximum_memory_bytes,
        )
        positive = [item for item in ranking if item.eligible and item.utility_score > 0]
        selected = positive[0] if positive else _idle_evaluation(ranking, objective)
        node_id = ranking[0].node_id
        generation = self._route_generations.get(node_id, 0) + 1
        self._route_generations[node_id] = generation
        decision = PlannerDecision(
            node_id=node_id,
            objective=objective,
            selected_role=NodeRole(selected.role),
            selected_utility=selected.utility_score,
            ranking=ranking,
            reason=(
                "highest positive measured marginal utility"
                if positive
                else "idle selected because every candidate role was non-positive or ineligible"
            ),
            route_generation=generation,
        )
        self.decisions.append(decision)
        return decision

    def update_after_canary(self, measurement: CanaryMeasurement) -> bool:
        """Update predictions and return whether the role remains admissible."""

        self.measurements.append(measurement)
        self._adjustments[(measurement.node_id, measurement.role)] = (
            measurement.measured_verified_tps_gain,
            measurement.measured_p95_latency_delta_ms,
            measurement.measured_interactive_throughput_delta,
        )
        p95_fraction = (
            measurement.measured_p95_latency_delta_ms / measurement.baseline_p95_latency_ms
        )
        throughput_fraction = (
            measurement.measured_interactive_throughput_delta
            / measurement.baseline_interactive_throughput
        )
        return (
            measurement.passed
            and measurement.measured_utility > 0
            and p95_fraction <= self.policy.maximum_interactive_p95_increase_fraction
            and throughput_fraction >= -self.policy.maximum_interactive_throughput_decrease_fraction
        )

    def monitor_and_reassign(
        self,
        measurement: CanaryMeasurement,
        candidates: list[RoleCandidate],
        *,
        objective: PlannerObjective,
        maximum_memory_bytes: int,
    ) -> PlannerDecision | None:
        if self.update_after_canary(measurement):
            return None
        adjusted = [
            item.model_copy(
                update={
                    "compatible": False,
                    "compatibility_reason": "removed after measured non-degradation violation",
                }
            )
            if item.role == measurement.role
            else item
            for item in candidates
        ]
        return self.select(
            adjusted,
            objective=objective,
            baseline_p95_latency_ms=measurement.baseline_p95_latency_ms,
            baseline_interactive_throughput=measurement.baseline_interactive_throughput,
            maximum_memory_bytes=maximum_memory_bytes,
        )


def _idle_evaluation(ranking: list[RoleEvaluation], objective: PlannerObjective) -> RoleEvaluation:
    idle = next((item for item in ranking if item.role == NodeRole.IDLE.value), None)
    if idle is not None:
        return idle.model_copy(update={"eligible": True, "rejection_reason": None})
    template = ranking[0]
    return RoleEvaluation(
        node_id=template.node_id,
        role=NodeRole.IDLE.value,
        predicted_verified_tps_gain=0.0,
        predicted_p95_latency_delta_ms=0.0,
        predicted_memory_bytes=0,
        predicted_transfer_bytes=0,
        predicted_failure_cost=0.0,
        utility_score=0.0,
        confidence_interval=(0.0, 0.0),
        eligible=True,
        rejection_reason=None,
        objective=objective,
        components=UtilityComponents(
            expected_verified_token_gain=0.0,
            interactive_latency_penalty=0.0,
            transfer_cost=0.0,
            failure_recovery_cost=0.0,
            verification_cost=0.0,
            admission_risk=0.0,
            energy_cost=0.0,
            objective_weights=_objective_weights(objective),
        ),
        classification=template.classification,
    )


def planner_regret(
    measured_utilities: dict[NodeRole, float], selected_role: NodeRole
) -> dict[str, float | str | bool]:
    if selected_role not in measured_utilities:
        raise ValueError("selected role has no measured utility")
    best_role, best_utility = max(measured_utilities.items(), key=lambda item: item[1])
    selected = measured_utilities[selected_role]
    regret = best_utility - selected
    fraction = regret / best_utility if best_utility > 0 else 0.0
    return {
        "best_role": best_role.value,
        "best_measured_utility": best_utility,
        "selected_role": selected_role.value,
        "selected_measured_utility": selected,
        "planner_regret": regret,
        "planner_regret_fraction": fraction,
        "passes": fraction <= 0.10,
    }
