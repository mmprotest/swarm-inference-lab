"""Hierarchical engine/topology/mechanism planning with evidence gates."""

from __future__ import annotations

from enum import StrEnum

from pydantic import ConfigDict, Field

from swarm_inference.config.models import StrictModel
from swarm_inference.engines.interfaces import (
    ClusterCapabilities,
    EngineSupportReport,
    ExecutionPlan,
    ExecutionRequest,
    MechanismEvidence,
)
from swarm_inference.engines.registry import ExecutionEngineRegistry
from swarm_inference.model.descriptor import ResolvedModelDescriptor


class ExperimentDisposition(StrEnum):
    REQUIRED = "REQUIRED"
    AVAILABLE_CONDITIONAL = "AVAILABLE_CONDITIONAL"
    REJECTED_DEFAULT = "REJECTED_DEFAULT"
    EVIDENCE_ONLY = "EVIDENCE_ONLY"


class MechanismPolicy(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mechanism: str
    disposition: ExperimentDisposition
    reason: str
    requires_positive_utility: bool = False


class CanonicalPlanningDecision(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    selected: ExecutionPlan
    candidates: tuple[ExecutionPlan, ...]
    rejected_plans: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    engine_support: tuple[EngineSupportReport, ...]
    planning_levels: tuple[str, ...] = (
        "model_variant",
        "execution_engine",
        "topology",
        "stage_fast_path",
        "moe_strategy",
        "optional_mechanisms",
    )


DEFAULT_MECHANISM_POLICIES: tuple[MechanismPolicy, ...] = (
    MechanismPolicy(
        mechanism="lossless_activation_compression",
        disposition=ExperimentDisposition.AVAILABLE_CONDITIONAL,
        reason="exact but negative measured utility on the frozen stage-ring topology",
        requires_positive_utility=True,
    ),
    MechanismPolicy(
        mechanism="speculation",
        disposition=ExperimentDisposition.AVAILABLE_CONDITIONAL,
        reason="exact mechanism requires topology-specific positive utility",
        requires_positive_utility=True,
    ),
    MechanismPolicy(
        mechanism="prefetch",
        disposition=ExperimentDisposition.AVAILABLE_CONDITIONAL,
        reason="mixed and rejected variants require fresh matched-executor evidence",
        requires_positive_utility=True,
    ),
    MechanismPolicy(
        mechanism="aggressive_paging",
        disposition=ExperimentDisposition.REJECTED_DEFAULT,
        reason="adaptive paging did not consistently beat stock execution",
    ),
    MechanismPolicy(
        mechanism="synchronous_cpu_expert_speed",
        disposition=ExperimentDisposition.REJECTED_DEFAULT,
        reason="corrected matched benchmark classified synchronous placement NOT_USEFUL",
    ),
    MechanismPolicy(
        mechanism="dense_tensor_microshards",
        disposition=ExperimentDisposition.AVAILABLE_CONDITIONAL,
        reason="capacity mechanism unless a physical speed gate is positive",
        requires_positive_utility=True,
    ),
    MechanismPolicy(
        mechanism="tensor_microshards",
        disposition=ExperimentDisposition.AVAILABLE_CONDITIONAL,
        reason="microsharding is capacity-first until a matching physical speed gate passes",
        requires_positive_utility=True,
    ),
    MechanismPolicy(
        mechanism="routing_aware_placement",
        disposition=ExperimentDisposition.AVAILABLE_CONDITIONAL,
        reason="routing-aware Colibri placement was positive only under matched measurements",
        requires_positive_utility=True,
    ),
    MechanismPolicy(
        mechanism="background_inference",
        disposition=ExperimentDisposition.AVAILABLE_CONDITIONAL,
        reason="background workers must add fixed-window service throughput without interference",
        requires_positive_utility=True,
    ),
)


class CanonicalPlanner:
    """Compose registered engines while preserving negative experiment results."""

    def __init__(
        self,
        registry: ExecutionEngineRegistry,
        *,
        policies: tuple[MechanismPolicy, ...] = DEFAULT_MECHANISM_POLICIES,
    ) -> None:
        self.registry = registry
        self.policies = {item.mechanism: item for item in policies}

    @staticmethod
    def _normalise_idle(
        plan: ExecutionPlan,
        cluster: ClusterCapabilities,
    ) -> ExecutionPlan:
        accounted = set(plan.worker_roles) | set(plan.idle_workers)
        missing = {
            worker.worker_id: "engine candidate assigned no positive-utility role"
            for worker in cluster.workers
            if worker.worker_id not in accounted
        }
        if not missing:
            return plan
        return plan.model_copy(update={"idle_workers": {**plan.idle_workers, **missing}})

    def _policy_rejections(
        self,
        plan: ExecutionPlan,
        request: ExecutionRequest,
        evidence: dict[str, MechanismEvidence],
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        forced = set(request.forced_mechanisms)
        for mechanism, enabled in plan.optional_mechanisms.items():
            if not enabled:
                continue
            policy = self.policies.get(mechanism)
            if policy is None:
                measurement = evidence.get(mechanism)
                if mechanism not in forced and (
                    measurement is None
                    or not measurement.exactness_passed
                    or measurement.measured_utility <= 0
                    or measurement.runtime_fingerprint != plan.execution_identity
                ):
                    reasons.append(
                        f"{mechanism}: unregistered optional mechanism has no exact "
                        "matching-runtime positive-utility evidence"
                    )
                continue
            if mechanism in forced:
                continue
            if policy.disposition in {
                ExperimentDisposition.REJECTED_DEFAULT,
                ExperimentDisposition.EVIDENCE_ONLY,
            }:
                reasons.append(f"{mechanism}: {policy.disposition.value}: {policy.reason}")
                continue
            if policy.requires_positive_utility:
                measurement = evidence.get(mechanism)
                if (
                    measurement is None
                    or not measurement.exactness_passed
                    or measurement.measured_utility <= 0
                    or measurement.runtime_fingerprint != plan.execution_identity
                ):
                    reasons.append(
                        f"{mechanism}: no exact matched-runtime positive-utility evidence"
                    )
        return tuple(reasons)

    @staticmethod
    def _evidence_adjusted_score(
        plan: ExecutionPlan,
        evidence: dict[str, MechanismEvidence],
    ) -> float:
        """Apply only exact utility deltas measured under this execution identity."""

        delta = sum(
            measurement.measured_utility
            for mechanism, enabled in plan.optional_mechanisms.items()
            if enabled
            and (measurement := evidence.get(mechanism)) is not None
            and measurement.exactness_passed
            and measurement.runtime_fingerprint == plan.execution_identity
        )
        return plan.score + delta

    @staticmethod
    def _matching_evidence(
        plan: ExecutionPlan,
        supplied: tuple[MechanismEvidence, ...],
    ) -> dict[str, MechanismEvidence]:
        """Use only measurements bound to this exact composed runtime identity.

        Explicitly supplied measurements supersede installer/runtime profile
        evidence, allowing a newer matched benchmark to reject a formerly
        positive profile without changing the engine implementation.
        """

        matched: dict[str, MechanismEvidence] = {}
        for item in (*plan.mechanism_evidence, *supplied):
            if item.runtime_fingerprint == plan.execution_identity:
                matched[item.mechanism] = item
        return matched

    async def plan(
        self,
        model: ResolvedModelDescriptor,
        cluster: ClusterCapabilities,
        request: ExecutionRequest,
        *,
        mechanism_evidence: tuple[MechanismEvidence, ...] = (),
    ) -> CanonicalPlanningDecision:
        competition = await self.registry.compete(model, cluster, request)
        candidates = tuple(self._normalise_idle(item, cluster) for item in competition.candidates)
        rejected: dict[str, tuple[str, ...]] = {}
        eligible: list[tuple[ExecutionPlan, dict[str, MechanismEvidence]]] = []
        for plan in candidates:
            evidence = self._matching_evidence(plan, mechanism_evidence)
            reasons = self._policy_rejections(plan, request, evidence)
            if reasons:
                rejected[plan.plan_id] = reasons
            else:
                eligible.append((plan, evidence))
        if not eligible:
            detail = "; ".join(
                f"{plan_id}: {', '.join(reasons)}" for plan_id, reasons in sorted(rejected.items())
            )
            raise RuntimeError(f"all composed plans violate experiment policy: {detail}")
        selected, _selected_evidence = max(
            eligible,
            key=lambda pair: (
                self._evidence_adjusted_score(pair[0], pair[1]),
                pair[0].predicted_decode_tokens_s,
                -pair[0].predicted_ttft_ms,
                pair[0].engine_id,
                pair[0].plan_id,
            ),
        )
        return CanonicalPlanningDecision(
            selected=selected,
            candidates=candidates,
            rejected_plans=rejected,
            engine_support=competition.support,
        )


__all__ = [
    "DEFAULT_MECHANISM_POLICIES",
    "CanonicalPlanner",
    "CanonicalPlanningDecision",
    "ExperimentDisposition",
    "MechanismEvidence",
    "MechanismPolicy",
]
