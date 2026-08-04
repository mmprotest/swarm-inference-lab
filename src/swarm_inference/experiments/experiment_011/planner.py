"""Measured, profile-name-agnostic strategy planner for Experiment 011."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PlannerInputs:
    bandwidth_bps: float | None
    one_way_latency_ms: float
    jitter_ms: float
    loss_probability: float
    queue_depth: int
    available_device_memory_bytes: int
    required_distributed_execution: bool


@dataclass(frozen=True, slots=True)
class PlannerCandidate:
    name: str
    execution_family: str
    stage_count: int
    partition_method: str
    compression_mode: str
    speculation_provider: str
    speculation_depth: int
    stage_compute_ns: tuple[int, ...]
    stage_weight_bytes: tuple[int, ...]
    kv_cache_bytes: tuple[int, ...]
    serial_boundaries: float
    payload_bytes_per_token: float
    compression_encode_ns: float = 0.0
    compression_decode_ns: float = 0.0
    compression_ratio: float = 1.0
    draft_acceptance_rate: float = 0.0
    rejected_speculative_compute_ns: float = 0.0
    reliability_multiplier: float = 1.0
    measured_throughput_tps: float | None = None
    exact: bool = True
    suitable: bool = True
    idle_node_excluded: bool = False


@dataclass(frozen=True, slots=True)
class CandidateScore:
    candidate: str
    feasible: bool
    selected: bool
    predicted_latency_ns: float | None
    predicted_throughput_tps: float | None
    measured_throughput_tps: float | None
    objective_ns: float | None
    exclusion_reasons: tuple[str, ...]
    components: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["exclusion_reasons"] = list(self.exclusion_reasons)
        return value


@dataclass(frozen=True, slots=True)
class PlannerDecision:
    selected_candidate: str | None
    selected_measured_throughput_tps: float | None
    scores: tuple[CandidateScore, ...]
    explanation: str
    inputs: PlannerInputs

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_candidate": self.selected_candidate,
            "selected_measured_throughput_tps": self.selected_measured_throughput_tps,
            "scores": [score.to_dict() for score in self.scores],
            "explanation": self.explanation,
            "inputs": asdict(self.inputs),
        }


class MeasuredStrategyPlanner:
    """Ranks exact feasible candidates from measured quantities only."""

    def evaluate(
        self, candidates: list[PlannerCandidate], *, inputs: PlannerInputs
    ) -> PlannerDecision:
        provisional: list[tuple[PlannerCandidate, CandidateScore]] = []
        for candidate in candidates:
            reasons = []
            if not candidate.exact:
                reasons.append("candidate is not exact")
            if not candidate.suitable:
                reasons.append("node or execution family is unsuitable")
            if candidate.idle_node_excluded:
                reasons.append("idle/unsuitable node excluded")
            if (
                inputs.required_distributed_execution
                and candidate.execution_family == "local_monolithic"
            ):
                reasons.append("request requires distributed ownership")
            if any(
                weight + kv > inputs.available_device_memory_bytes
                for weight, kv in zip(
                    candidate.stage_weight_bytes, candidate.kv_cache_bytes, strict=False
                )
            ):
                reasons.append("stage memory exceeds available device memory")
            if candidate.measured_throughput_tps is None or candidate.measured_throughput_tps <= 0:
                reasons.append("no valid measured throughput")
            feasible = not reasons
            components: dict[str, float] = {}
            predicted_latency_ns = None
            predicted_throughput = None
            objective = None
            if feasible:
                compute_ns = float(sum(candidate.stage_compute_ns))
                imbalance_ns = (
                    float(max(candidate.stage_compute_ns) - min(candidate.stage_compute_ns))
                    if candidate.stage_compute_ns
                    else 0.0
                )
                effective_payload = candidate.payload_bytes_per_token / max(
                    candidate.compression_ratio, 1.0
                )
                serialisation_ns = (
                    0.0
                    if inputs.bandwidth_bps is None
                    else effective_payload * 8.0 / inputs.bandwidth_bps * 1e9
                )
                network_latency_ns = (
                    candidate.serial_boundaries
                    * (inputs.one_way_latency_ms + inputs.jitter_ms / 2.0)
                    * 1e6
                )
                queue_penalty_ns = (
                    candidate.serial_boundaries * max(inputs.queue_depth - 1, 0) * 500.0
                )
                codec_ns = candidate.compression_encode_ns + candidate.compression_decode_ns
                rejected_ns = candidate.rejected_speculative_compute_ns
                failure_ns = (
                    inputs.loss_probability
                    * candidate.serial_boundaries
                    * (compute_ns + network_latency_ns)
                    * candidate.reliability_multiplier
                )
                predicted_latency_ns = (
                    compute_ns
                    + network_latency_ns
                    + serialisation_ns
                    + queue_penalty_ns
                    + codec_ns
                    + imbalance_ns * 0.10
                    + rejected_ns
                    + failure_ns
                )
                predicted_throughput = 1e9 / predicted_latency_ns
                measured_latency_ns = 1e9 / float(candidate.measured_throughput_tps)
                # The measured latency is the strongest term; the analytical
                # components make the decision auditable and penalise risks
                # omitted by a single successful observation.
                objective = measured_latency_ns + failure_ns + imbalance_ns * 0.05 + rejected_ns
                components = {
                    "compute_ns": compute_ns,
                    "network_latency_ns": network_latency_ns,
                    "serialisation_ns": serialisation_ns,
                    "queue_penalty_ns": queue_penalty_ns,
                    "codec_ns": codec_ns,
                    "stage_imbalance_penalty_ns": imbalance_ns * 0.10,
                    "rejected_speculative_compute_ns": rejected_ns,
                    "failure_risk_ns": failure_ns,
                    "measured_latency_ns": measured_latency_ns,
                }
            score = CandidateScore(
                candidate=candidate.name,
                feasible=feasible,
                selected=False,
                predicted_latency_ns=predicted_latency_ns,
                predicted_throughput_tps=predicted_throughput,
                measured_throughput_tps=candidate.measured_throughput_tps,
                objective_ns=objective,
                exclusion_reasons=tuple(reasons),
                components=components,
            )
            provisional.append((candidate, score))
        feasible_scores = [pair for pair in provisional if pair[1].feasible]
        selected_name = None
        selected_tps = None
        if feasible_scores:
            selected_candidate, _selected_score = min(
                feasible_scores,
                key=lambda pair: (
                    float(pair[1].objective_ns),
                    pair[0].serial_boundaries,
                    pair[0].payload_bytes_per_token,
                    pair[0].name,
                ),
            )
            selected_name = selected_candidate.name
            selected_tps = selected_candidate.measured_throughput_tps
            scores = tuple(
                CandidateScore(
                    **{
                        **asdict(score),
                        "selected": candidate.name == selected_name,
                    }
                )
                for candidate, score in provisional
            )
            explanation = (
                f"{selected_name} had the lowest measured-risk objective using bandwidth, RTT, "
                "jitter, loss, queue depth, stage compute/memory, KV memory, compression cost, "
                "draft acceptance, serial boundaries, payload, imbalance and reliability inputs."
            )
        else:
            scores = tuple(score for _, score in provisional)
            explanation = "No exact memory-feasible measured candidate satisfied the request."
        return PlannerDecision(
            selected_candidate=selected_name,
            selected_measured_throughput_tps=selected_tps,
            scores=scores,
            explanation=explanation,
            inputs=inputs,
        )
