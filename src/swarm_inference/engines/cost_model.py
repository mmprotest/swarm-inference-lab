"""Shared measured-cost and objective scoring helpers for execution engines."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import ConfigDict, Field, NonNegativeInt

from swarm_inference.config.models import StrictModel


class ObjectiveWeights(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ttft: float = Field(default=0.25, ge=0)
    decode_throughput: float = Field(default=0.30, ge=0)
    aggregate_throughput: float = Field(default=0.20, ge=0)
    memory_headroom: float = Field(default=0.15, ge=0)
    reliability: float = Field(default=0.10, ge=0)


class PlanCostInputs(StrictModel):
    """Comparable facts; missing measurements remain explicit in diagnostics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    measured_prefill_tokens_s: float | None = Field(default=None, gt=0)
    measured_decode_tokens_s: float | None = Field(default=None, gt=0)
    measured_compute_rate: float | None = Field(default=None, gt=0)
    queue_depth: NonNegativeInt = 0
    reliability: float = Field(default=1.0, ge=0, le=1)
    usable_memory_bytes: NonNegativeInt = 0
    required_memory_bytes: NonNegativeInt = 0
    resident_model_bytes: NonNegativeInt = 0
    artifact_transfer_bytes: NonNegativeInt = 0
    network_latency_ms: float = Field(default=0, ge=0)
    network_bandwidth_bytes_s: float | None = Field(default=None, gt=0)
    network_jitter_ms: float = Field(default=0, ge=0)
    serialization_ms_per_token: float = Field(default=0, ge=0)
    messages_per_token: float = Field(default=0, ge=0)
    bytes_per_token: float = Field(default=0, ge=0)
    serial_waits_per_token: float = Field(default=0, ge=0)
    cache_hit_rate: float | None = Field(default=None, ge=0, le=1)
    cache_miss_cost_ms: float = Field(default=0, ge=0)
    expert_reduction_ms: float = Field(default=0, ge=0)
    startup_cost_ms: float = Field(default=0, ge=0)
    compile_cost_ms: float = Field(default=0, ge=0)
    graph_capture_cost_ms: float = Field(default=0, ge=0)
    failure_replay_cost_ms: float = Field(default=0, ge=0)
    concurrency: int = Field(default=1, ge=1)
    replica_count: int = Field(default=1, ge=1)
    batching_factor: float = Field(default=1, ge=1)
    request_priority: int = 0

    @property
    def memory_headroom_ratio(self) -> float:
        if self.usable_memory_bytes <= 0:
            return 0.0
        return max(
            0.0,
            (self.usable_memory_bytes - self.required_memory_bytes)
            / self.usable_memory_bytes,
        )

    @property
    def acquisition_ms(self) -> float:
        missing = max(0, self.artifact_transfer_bytes - self.resident_model_bytes)
        if missing == 0:
            return 0.0
        if self.network_bandwidth_bytes_s is None:
            return float("inf")
        return missing / self.network_bandwidth_bytes_s * 1000

    @property
    def cache_penalty_ms(self) -> float:
        miss_fraction = 1.0 if self.cache_hit_rate is None else 1.0 - self.cache_hit_rate
        return miss_fraction * self.cache_miss_cost_ms


class ScoredUtility(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    score: float
    predicted_ttft_ms: float = Field(ge=0)
    predicted_decode_tokens_s: float = Field(ge=0)
    predicted_aggregate_tokens_s: float = Field(ge=0)
    components: dict[str, float]
    unmeasured_inputs: tuple[str, ...] = ()


def score_costs(
    inputs: PlanCostInputs,
    *,
    objective: Literal["speed", "throughput", "capacity", "balanced"],
    weights: ObjectiveWeights | None = None,
) -> ScoredUtility:
    selected_weights = weights or ObjectiveWeights()
    unmeasured: list[str] = []
    decode_rate = inputs.measured_decode_tokens_s or inputs.measured_compute_rate
    if decode_rate is None:
        decode_rate = 1.0
        unmeasured.append("decode_rate")
    prefill_rate = inputs.measured_prefill_tokens_s
    if prefill_rate is None:
        prefill_rate = decode_rate
        unmeasured.append("prefill_rate")
    base_token_ms = 1000 / max(decode_rate, 1e-9)
    per_token_ms = (
        base_token_ms * (1 + inputs.queue_depth)
        + inputs.network_latency_ms
        + inputs.network_jitter_ms
        + inputs.serialization_ms_per_token
        + inputs.cache_penalty_ms
        + inputs.expert_reduction_ms
        + inputs.failure_replay_cost_ms
    ) / max(inputs.reliability, 1e-9)
    predicted_decode = 1000 / max(per_token_ms, 1e-9)
    aggregate = (
        predicted_decode
        * min(inputs.concurrency, inputs.replica_count)
        * inputs.batching_factor
    )
    cold_costs = (
        inputs.startup_cost_ms
        + inputs.compile_cost_ms
        + inputs.graph_capture_cost_ms
    )
    acquisition = inputs.acquisition_ms
    if acquisition == float("inf"):
        unmeasured.append("acquisition_bandwidth")
        acquisition = 0.0
    predicted_ttft = 1000 / max(prefill_rate, 1e-9) + cold_costs + acquisition
    ttft_utility = 1 / (1 + predicted_ttft / 1000)
    decode_utility = predicted_decode / (1 + predicted_decode)
    aggregate_utility = aggregate / (1 + aggregate)
    components = {
        "ttft_utility": ttft_utility,
        "decode_utility": decode_utility,
        "aggregate_utility": aggregate_utility,
        "memory_headroom": inputs.memory_headroom_ratio,
        "reliability": inputs.reliability,
    }
    if objective == "speed":
        score = predicted_decode * ttft_utility
    elif objective == "throughput":
        score = aggregate
    elif objective == "capacity":
        score = inputs.memory_headroom_ratio * inputs.reliability
    else:
        total_weight = sum(selected_weights.model_dump().values())
        if total_weight <= 0:
            raise ValueError("balanced objective weights must have a positive sum")
        score = (
            selected_weights.ttft * ttft_utility
            + selected_weights.decode_throughput * decode_utility
            + selected_weights.aggregate_throughput * aggregate_utility
            + selected_weights.memory_headroom * inputs.memory_headroom_ratio
            + selected_weights.reliability * inputs.reliability
        ) / total_weight
    # Unmeasured estimates stay eligible but lose deterministic confidence to
    # an otherwise identical measured candidate.
    score *= 0.98 ** len(set(unmeasured))
    return ScoredUtility(
        score=score,
        predicted_ttft_ms=predicted_ttft,
        predicted_decode_tokens_s=predicted_decode,
        predicted_aggregate_tokens_s=aggregate,
        components=components,
        unmeasured_inputs=tuple(sorted(set(unmeasured))),
    )


def stable_plan_id(prefix: str, identity: dict[str, Any]) -> str:
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:20]}"


__all__ = [
    "ObjectiveWeights",
    "PlanCostInputs",
    "ScoredUtility",
    "score_costs",
    "stable_plan_id",
]
