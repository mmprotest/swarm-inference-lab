"""Second-level, measured-utility planning for experts inside a stage."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from swarm_inference.config.models import StrictModel
from swarm_inference.engines.topology import TopologyDomain


class ExpertPolicy(StrEnum):
    AUTO = "auto"
    LOCAL = "local"
    WHOLE_REMOTE = "whole-remote"
    MICROSHARD_REMOTE = "microshard-remote"
    HYBRID = "hybrid"


class ExpertStrategy(StrEnum):
    LOCAL = "local"
    WHOLE_REMOTE = "whole-remote"
    MICROSHARD_REMOTE = "microshard-remote"
    HYBRID = "hybrid"


class ExpertUtilityInputs(StrictModel):
    measured_local_expert_ms: float = Field(ge=0)
    measured_remote_expert_ms: float = Field(ge=0)
    serialization_ms: float = Field(ge=0)
    network_transfer_ms: float = Field(ge=0)
    queue_delay_ms: float = Field(ge=0)
    reduction_ms: float = Field(ge=0)
    cache_hit_rate: float = Field(ge=0, le=1)
    cache_miss_cost_ms: float = Field(default=0, ge=0)
    memory_pressure_cost_ms: float = Field(default=0, ge=0)
    memory_relief_value_ms: float = Field(default=0, ge=0)
    failure_risk: float = Field(default=0, ge=0, le=1)
    expected_fallback_cost_ms: float = Field(default=0, ge=0)

    @property
    def expected_remote_cost_ms(self) -> float:
        cache_cost = (1.0 - self.cache_hit_rate) * self.cache_miss_cost_ms
        failure_cost = self.failure_risk * self.expected_fallback_cost_ms
        return (
            self.measured_remote_expert_ms
            + self.serialization_ms
            + self.network_transfer_ms
            + self.queue_delay_ms
            + self.reduction_ms
            + cache_cost
            + self.memory_pressure_cost_ms
            + failure_cost
        )

    @property
    def utility_ms(self) -> float:
        return (
            self.measured_local_expert_ms
            + self.memory_relief_value_ms
            - self.expected_remote_cost_ms
        )

    def breakdown(self) -> dict[str, float]:
        return {
            "measured_local_expert_ms": self.measured_local_expert_ms,
            "measured_remote_expert_ms": self.measured_remote_expert_ms,
            "serialization_ms": self.serialization_ms,
            "network_transfer_ms": self.network_transfer_ms,
            "queue_delay_ms": self.queue_delay_ms,
            "reduction_ms": self.reduction_ms,
            "cache_hit_rate": self.cache_hit_rate,
            "cache_miss_cost_ms": (1.0 - self.cache_hit_rate) * self.cache_miss_cost_ms,
            "memory_pressure_cost_ms": self.memory_pressure_cost_ms,
            "memory_relief_value_ms": self.memory_relief_value_ms,
            "failure_risk": self.failure_risk,
            "expected_fallback_cost_ms": self.failure_risk * self.expected_fallback_cost_ms,
            "expected_remote_cost_ms": self.expected_remote_cost_ms,
            "utility_ms": self.utility_ms,
        }


class ExpertStrategyCandidate(StrictModel):
    candidate_id: str
    strategy: ExpertStrategy
    worker_ids: list[str] = Field(default_factory=list)
    utility: ExpertUtilityInputs | None = None
    feasible: bool = True
    correctness_validated: bool = True
    model_identity_matches: bool = True
    quantization_identity_matches: bool = True
    memory_required_bytes: int = Field(default=0, ge=0)
    memory_available_bytes: int = Field(default=0, ge=0)
    worker_memory_required_bytes: dict[str, int] = Field(default_factory=dict)
    worker_memory_available_bytes: dict[str, int] = Field(default_factory=dict)
    explanation: list[str] = Field(default_factory=list)
    topology_domain: TopologyDomain = TopologyDomain.LOCAL_FAST

    @model_validator(mode="after")
    def validate_candidate(self) -> ExpertStrategyCandidate:
        if self.strategy == ExpertStrategy.LOCAL and self.worker_ids:
            raise ValueError("local expert candidates do not name remote workers")
        if self.strategy != ExpertStrategy.LOCAL and not self.worker_ids:
            raise ValueError("remote expert candidates require workers")
        if self.strategy != ExpertStrategy.LOCAL and self.utility is None:
            raise ValueError("remote expert candidates require measured utility inputs")
        if set(self.worker_memory_required_bytes) != set(self.worker_memory_available_bytes):
            raise ValueError("per-worker expert memory accounting keys do not match")
        if any(value < 0 for value in self.worker_memory_required_bytes.values()) or any(
            value < 0 for value in self.worker_memory_available_bytes.values()
        ):
            raise ValueError("per-worker expert memory accounting cannot be negative")
        return self

    @property
    def memory_fits(self) -> bool:
        if self.worker_memory_required_bytes:
            return all(
                required <= self.worker_memory_available_bytes[worker_id]
                for worker_id, required in self.worker_memory_required_bytes.items()
            )
        return self.memory_required_bytes <= self.memory_available_bytes

    @property
    def memory_rejections(self) -> list[str]:
        return [
            f"worker {worker_id} requires {required} bytes but only "
            f"{self.worker_memory_available_bytes[worker_id]} bytes are available"
            for worker_id, required in sorted(self.worker_memory_required_bytes.items())
            if required > self.worker_memory_available_bytes[worker_id]
        ]


class RejectedExpertStrategy(StrictModel):
    candidate_id: str
    strategy: ExpertStrategy
    reasons: list[str]
    utility_ms: float | None = None


class ExpertPlacementDecision(StrictModel):
    stage_id: int = Field(ge=0)
    layer_id: int = Field(ge=0)
    expert_id: int = Field(ge=0)
    policy: ExpertPolicy
    selected_candidate_id: str
    selected_strategy: ExpertStrategy
    selected_workers: list[str]
    measured_utility_ms: float
    capacity_required: bool = False
    local_fallback_permitted: bool = False
    forced_remote: bool = False
    explanation: list[str]
    rejected: list[RejectedExpertStrategy]


class StageExpertPlan(StrictModel):
    stage_id: int = Field(ge=0)
    policy: ExpertPolicy
    require_remote_experts: bool = False
    placements: list[ExpertPlacementDecision]
    rejected_strategy_count: int = Field(ge=0)

    @property
    def remote_placement_count(self) -> int:
        return sum(item.selected_strategy != ExpertStrategy.LOCAL for item in self.placements)


class ExpertUtilityPlanner:
    """Choose capacity or positive-utility expert placement after stage topology."""

    def choose(
        self,
        *,
        stage_id: int,
        layer_id: int,
        expert_id: int,
        candidates: list[ExpertStrategyCandidate],
        policy: ExpertPolicy | str = ExpertPolicy.AUTO,
        require_remote: bool = False,
        allow_local_fallback: bool = False,
    ) -> ExpertPlacementDecision:
        selected_policy = ExpertPolicy(policy)
        if not candidates:
            raise ValueError("expert planning requires at least one candidate")
        identifiers = [item.candidate_id for item in candidates]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("expert candidate IDs must be unique")
        local = [item for item in candidates if item.strategy == ExpertStrategy.LOCAL]
        local_feasible = any(
            item.feasible
            and item.correctness_validated
            and item.model_identity_matches
            and item.quantization_identity_matches
            and item.memory_fits
            for item in local
        )
        capacity_required = not local_feasible
        evaluated: list[tuple[ExpertStrategyCandidate, list[str]]] = []
        eligible: list[ExpertStrategyCandidate] = []
        for candidate in candidates:
            reasons: list[str] = []
            if not candidate.feasible:
                reasons.append("candidate is operationally infeasible")
            if not candidate.correctness_validated:
                reasons.append("exactness has not been validated")
            if not candidate.model_identity_matches:
                reasons.append("model fingerprint does not match")
            if not candidate.quantization_identity_matches:
                reasons.append("quantisation fingerprint does not match")
            if not candidate.memory_fits:
                reasons.extend(
                    candidate.memory_rejections
                    or [
                        f"requires {candidate.memory_required_bytes} bytes but only "
                        f"{candidate.memory_available_bytes} bytes are available"
                    ]
                )
            if require_remote and candidate.strategy == ExpertStrategy.LOCAL:
                reasons.append("forced-remote validation excludes local execution")
            if (
                candidate.strategy != ExpertStrategy.LOCAL
                and candidate.topology_domain == TopologyDomain.WAN
            ):
                reasons.append(
                    "fine-grained synchronous expert RPC is not admitted across a WAN "
                    "topology domain; place the expert inside a contiguous WAN stage"
                )
            if selected_policy == ExpertPolicy.LOCAL and candidate.strategy != ExpertStrategy.LOCAL:
                reasons.append("local policy excludes remote execution")
            if (
                selected_policy == ExpertPolicy.WHOLE_REMOTE
                and candidate.strategy != ExpertStrategy.WHOLE_REMOTE
            ):
                reasons.append("whole-remote policy excludes this strategy")
            if (
                selected_policy == ExpertPolicy.MICROSHARD_REMOTE
                and candidate.strategy != ExpertStrategy.MICROSHARD_REMOTE
            ):
                reasons.append("microshard-remote policy excludes this strategy")
            evaluated.append((candidate, reasons))
            if not reasons:
                eligible.append(candidate)
        if selected_policy == ExpertPolicy.AUTO and not require_remote:
            positive_remote = [
                item
                for item in eligible
                if item.strategy != ExpertStrategy.LOCAL
                and item.utility is not None
                and item.utility.utility_ms > 0
            ]
            if positive_remote:
                selected = max(
                    positive_remote,
                    key=lambda item: (
                        item.utility.utility_ms if item.utility else 0,
                        item.candidate_id,
                    ),
                )
            elif local_feasible:
                selected = next(
                    item
                    for item in eligible
                    if item.strategy == ExpertStrategy.LOCAL and item.memory_fits
                )
            else:
                remote = [item for item in eligible if item.strategy != ExpertStrategy.LOCAL]
                if not remote:
                    raise RuntimeError("no capacity-feasible exact expert strategy")
                selected = max(
                    remote,
                    key=lambda item: (
                        item.utility.utility_ms if item.utility else 0,
                        item.candidate_id,
                    ),
                )
        else:
            remote_or_policy = [
                item
                for item in eligible
                if not require_remote or item.strategy != ExpertStrategy.LOCAL
            ]
            if not remote_or_policy:
                raise RuntimeError("no exact expert strategy satisfies the requested policy")
            selected = max(
                remote_or_policy,
                key=lambda item: (
                    item.utility.utility_ms if item.utility is not None else 0,
                    item.candidate_id,
                ),
            )
        rejected: list[RejectedExpertStrategy] = []
        for candidate, reasons in evaluated:
            if candidate.candidate_id == selected.candidate_id:
                continue
            final_reasons = list(reasons)
            if not final_reasons:
                if (
                    selected_policy == ExpertPolicy.AUTO
                    and candidate.strategy != ExpertStrategy.LOCAL
                    and candidate.utility is not None
                    and candidate.utility.utility_ms <= 0
                    and local_feasible
                ):
                    final_reasons.append(
                        f"measured remote utility is non-positive ({candidate.utility.utility_ms:.6f} ms)"
                    )
                else:
                    final_reasons.append(
                        f"lower measured utility than selected candidate {selected.candidate_id}"
                    )
            rejected.append(
                RejectedExpertStrategy(
                    candidate_id=candidate.candidate_id,
                    strategy=candidate.strategy,
                    reasons=final_reasons,
                    utility_ms=(candidate.utility.utility_ms if candidate.utility else 0.0),
                )
            )
        utility = selected.utility.utility_ms if selected.utility is not None else 0.0
        explanation = list(selected.explanation)
        if selected.strategy == ExpertStrategy.LOCAL:
            explanation.append("local memory permits execution and remote utility is non-positive")
        elif capacity_required:
            explanation.append("remote placement is required to satisfy stage memory capacity")
        else:
            explanation.append(f"remote strategy has positive measured utility ({utility:.6f} ms)")
        if require_remote:
            explanation.append("forced-remote exactness validation is enabled")
        return ExpertPlacementDecision(
            stage_id=stage_id,
            layer_id=layer_id,
            expert_id=expert_id,
            policy=selected_policy,
            selected_candidate_id=selected.candidate_id,
            selected_strategy=selected.strategy,
            selected_workers=list(selected.worker_ids),
            measured_utility_ms=utility,
            capacity_required=capacity_required,
            local_fallback_permitted=(
                allow_local_fallback
                and not require_remote
                and selected.strategy != ExpertStrategy.LOCAL
                and local_feasible
            ),
            forced_remote=require_remote,
            explanation=explanation,
            rejected=rejected,
        )

    def plan_stage(
        self,
        *,
        stage_id: int,
        candidates_by_expert: dict[tuple[int, int], list[ExpertStrategyCandidate]],
        policy: ExpertPolicy | str = ExpertPolicy.AUTO,
        require_remote: bool = False,
        allow_local_fallback: bool = False,
    ) -> StageExpertPlan:
        selected_policy = ExpertPolicy(policy)
        placements = [
            self.choose(
                stage_id=stage_id,
                layer_id=layer_id,
                expert_id=expert_id,
                candidates=candidates,
                policy=selected_policy,
                require_remote=require_remote,
                allow_local_fallback=allow_local_fallback,
            )
            for (layer_id, expert_id), candidates in sorted(candidates_by_expert.items())
        ]
        if require_remote and not any(
            item.selected_strategy != ExpertStrategy.LOCAL for item in placements
        ):
            raise RuntimeError("forced-remote expert plan contains no remote placement")
        return StageExpertPlan(
            stage_id=stage_id,
            policy=selected_policy,
            require_remote_experts=require_remote,
            placements=placements,
            rejected_strategy_count=sum(len(item.rejected) for item in placements),
        )


def explain_expert_plan(plan: StageExpertPlan) -> dict[str, Any]:
    return plan.model_dump(mode="json")


__all__ = [
    "ExpertPlacementDecision",
    "ExpertPolicy",
    "ExpertStrategy",
    "ExpertStrategyCandidate",
    "ExpertUtilityInputs",
    "ExpertUtilityPlanner",
    "RejectedExpertStrategy",
    "StageExpertPlan",
    "explain_expert_plan",
]
