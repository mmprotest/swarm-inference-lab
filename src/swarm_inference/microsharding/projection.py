"""Deterministic event-driven collective and hybrid parallel projections."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

ResultClassification = Literal[
    "logical_single_gpu_measurement",
    "logical_microsharding_correctness",
    "independent_rank_projection",
    "low_latency_cell_projection",
    "wan_projection",
    "real_moe_layer_measurement",
    "k3_checkpoint_projection",
]


@dataclass(frozen=True, slots=True)
class NetworkProfile:
    name: str
    one_way_latency_ms: float
    bandwidth_mbps: float | None
    jitter_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.one_way_latency_ms < 0:
            raise ValueError("network latency cannot be negative")
        if self.bandwidth_mbps is not None and self.bandwidth_mbps <= 0:
            raise ValueError("network bandwidth must be positive")
        if self.jitter_fraction < 0:
            raise ValueError("jitter fraction cannot be negative")


NETWORK_PROFILES: dict[str, NetworkProfile] = {
    "same_gpu_logical": NetworkProfile("same_gpu_logical", 0.0, None),
    "nvlink_class": NetworkProfile("nvlink_class", 0.01, 400_000),
    "thunderbolt_rdma_class": NetworkProfile("thunderbolt_rdma_class", 0.05, 80_000),
    "datacentre_ethernet": NetworkProfile("datacentre_ethernet", 0.1, 100_000),
    "home_lan_10gbe": NetworkProfile("home_lan_10gbe", 0.25, 10_000),
    "home_lan_1gbe": NetworkProfile("home_lan_1gbe", 1.0, 1_000),
    "nearby_residential": NetworkProfile("nearby_residential", 5.0, 500),
    "regional": NetworkProfile("regional", 20.0, 100),
    "international": NetworkProfile("international", 50.0, 100),
    "global_residential": NetworkProfile("global_residential", 100.0, 20),
}


@dataclass(frozen=True, slots=True)
class CollectiveEstimate:
    operation: str
    algorithm: str
    rank_count: int
    payload_bytes: int
    steps: int
    step_payload_bytes: float
    bytes_sent_per_rank: float
    aggregate_bytes: float
    base_latency_ms: float
    transfer_time_ms: float
    jitter_time_ms: float
    completion_time_ms: float
    critical_path: list[dict[str, float]]

    def payload(self) -> dict[str, Any]:
        return {
            "classification": "independent_rank_projection",
            "operation": self.operation,
            "algorithm": self.algorithm,
            "rank_count": self.rank_count,
            "payload_bytes": self.payload_bytes,
            "steps": self.steps,
            "step_payload_bytes": self.step_payload_bytes,
            "bytes_sent_per_rank": self.bytes_sent_per_rank,
            "aggregate_bytes": self.aggregate_bytes,
            "base_latency_ms": self.base_latency_ms,
            "transfer_time_ms": self.transfer_time_ms,
            "jitter_time_ms": self.jitter_time_ms,
            "completion_time_ms": self.completion_time_ms,
            "critical_path": self.critical_path,
        }


def _transfer_ms(byte_count: float, bandwidth_mbps: float | None) -> float:
    if byte_count <= 0 or bandwidth_mbps is None:
        return 0.0
    return byte_count * 8 / (bandwidth_mbps * 1_000_000) * 1_000


def collective_shape(
    *,
    operation: str,
    algorithm: str,
    rank_count: int,
    payload_bytes: int,
) -> tuple[int, float, float, float]:
    """Return steps, bytes/step, bytes/rank, and aggregate wire bytes."""

    if rank_count <= 0 or payload_bytes < 0:
        raise ValueError("rank_count must be positive and payload bytes non-negative")
    if rank_count == 1:
        return 0, 0.0, 0.0, 0.0
    log_steps = math.ceil(math.log2(rank_count))
    if operation == "barrier":
        steps = log_steps if algorithm != "ring" else rank_count - 1
        return steps, 0.0, 0.0, 0.0
    if operation == "all_reduce_sum":
        if algorithm == "ring":
            steps = 2 * (rank_count - 1)
            step_bytes = payload_bytes / rank_count
            per_rank = steps * step_bytes
            return steps, step_bytes, per_rank, per_rank * rank_count
        if algorithm == "binary_tree":
            steps = 2 * log_steps
            per_rank = 2 * payload_bytes
            return steps, float(payload_bytes), per_rank, 2.0 * (rank_count - 1) * payload_bytes
        if algorithm == "recursive_doubling":
            steps = log_steps
            per_rank = steps * payload_bytes
            return steps, float(payload_bytes), per_rank, per_rank * rank_count
        if algorithm == "leader_gather_broadcast":
            steps = 2 * (rank_count - 1)
            per_rank = 2 * payload_bytes
            return steps, float(payload_bytes), per_rank, 2.0 * (rank_count - 1) * payload_bytes
    if operation in {"all_gather", "reduce_scatter_sum"}:
        phase_multiplier = 1
        if algorithm == "ring":
            steps = phase_multiplier * (rank_count - 1)
            step_bytes = payload_bytes / rank_count
            per_rank = steps * step_bytes
            return steps, step_bytes, per_rank, per_rank * rank_count
        if algorithm in {"binary_tree", "recursive_doubling"}:
            steps = log_steps
            step_bytes = float(payload_bytes)
            per_rank = steps * step_bytes
            return steps, step_bytes, per_rank, per_rank * rank_count
        if algorithm == "leader_gather_broadcast":
            steps = rank_count - 1
            per_rank = float(payload_bytes)
            return steps, float(payload_bytes), per_rank, (rank_count - 1) * payload_bytes
    if operation == "all_to_all":
        steps = rank_count - 1
        step_bytes = payload_bytes / rank_count
        per_rank = steps * step_bytes
        return steps, step_bytes, per_rank, per_rank * rank_count
    if operation in {"broadcast", "gather_to_leader"}:
        steps = log_steps if algorithm in {"binary_tree", "recursive_doubling"} else rank_count - 1
        # A tree's critical path sends the full payload at every level.  Total
        # aggregate traffic still has one transfer per non-root rank.
        return steps, float(payload_bytes), float(payload_bytes), (rank_count - 1) * payload_bytes
    if operation == "distributed_argmax":
        # Candidate reduction followed by winner broadcast.
        if algorithm == "ring":
            steps = 2 * (rank_count - 1)
        elif algorithm in {"binary_tree", "recursive_doubling"}:
            steps = 2 * log_steps
        else:
            steps = 2 * (rank_count - 1)
        per_rank = 2.0 * payload_bytes
        return steps, float(payload_bytes), per_rank, 2.0 * (rank_count - 1) * payload_bytes
    raise ValueError(f"unsupported collective operation/algorithm: {operation}/{algorithm}")


def estimate_collective(
    *,
    operation: str,
    algorithm: str,
    rank_count: int,
    payload_bytes: int,
    network: NetworkProfile,
    seed: int = 1,
    straggler_delay_ms: float = 0.0,
) -> CollectiveEstimate:
    steps, step_bytes, per_rank, aggregate = collective_shape(
        operation=operation,
        algorithm=algorithm,
        rank_count=rank_count,
        payload_bytes=payload_bytes,
    )
    random_generator = random.Random(seed)
    critical_path: list[dict[str, float]] = []
    jitter_total = 0.0
    base_total = 0.0
    transfer_total = 0.0
    for step in range(steps):
        jitter = random_generator.uniform(
            -network.jitter_fraction * network.one_way_latency_ms,
            network.jitter_fraction * network.one_way_latency_ms,
        )
        latency = max(network.one_way_latency_ms + jitter, 0.0)
        transfer = _transfer_ms(step_bytes, network.bandwidth_mbps)
        base_total += network.one_way_latency_ms
        jitter_total += latency - network.one_way_latency_ms
        transfer_total += transfer
        critical_path.append(
            {
                "step": float(step),
                "latency_ms": latency,
                "transfer_ms": transfer,
                "completion_delta_ms": latency + transfer,
            }
        )
    completion = base_total + jitter_total + transfer_total + max(straggler_delay_ms, 0.0)
    return CollectiveEstimate(
        operation=operation,
        algorithm=algorithm,
        rank_count=rank_count,
        payload_bytes=payload_bytes,
        steps=steps,
        step_payload_bytes=step_bytes,
        bytes_sent_per_rank=per_rank,
        aggregate_bytes=aggregate,
        base_latency_ms=base_total,
        transfer_time_ms=transfer_total,
        jitter_time_ms=jitter_total,
        completion_time_ms=completion,
        critical_path=critical_path,
    )


@dataclass(slots=True)
class ProjectionEvent:
    event_type: str
    timestamp_ms: float
    rank_id: str | None = None
    layer_id: int | None = None
    collective_id: str | None = None
    step: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return {
            "classification": "independent_rank_projection",
            "event_type": self.event_type,
            "timestamp_ms": self.timestamp_ms,
            "rank_id": self.rank_id,
            "layer_id": self.layer_id,
            "collective_id": self.collective_id,
            "step": self.step,
            "details": self.details,
        }


@dataclass(frozen=True, slots=True)
class CollectiveWork:
    collective_id: str
    operation: str
    algorithm: str
    payload_bytes: int
    rank_ids: tuple[str, ...]
    phase: str


@dataclass(slots=True)
class ProjectionResult:
    completion_time_ms: float
    compute_completion_time_ms: float
    collective_time_ms: float
    events: list[ProjectionEvent]
    slowest_rank: str
    resource_mode: str

    def payload(self) -> dict[str, Any]:
        return {
            "classification": "independent_rank_projection",
            "completion_time_ms": self.completion_time_ms,
            "compute_completion_time_ms": self.compute_completion_time_ms,
            "collective_time_ms": self.collective_time_ms,
            "slowest_rank": self.slowest_rank,
            "resource_mode": self.resource_mode,
            "events": [item.payload() for item in self.events],
        }


class EventDrivenProjector:
    """Replay measured rank compute and collective work on a virtual clock."""

    def __init__(self, *, seed: int = 1) -> None:
        self.seed = seed

    def project_layer(
        self,
        *,
        layer_id: int,
        rank_compute_ms: dict[str, float],
        collectives: Iterable[CollectiveWork],
        network: NetworkProfile,
        resource_mode: Literal["independent", "shared"] = "independent",
        straggler_delays_ms: dict[str, float] | None = None,
        failure_rank: str | None = None,
        rejoin_delay_ms: float = 0.0,
    ) -> ProjectionResult:
        if not rank_compute_ms:
            raise ValueError("rank compute timings cannot be empty")
        if any(value < 0 for value in rank_compute_ms.values()):
            raise ValueError("rank compute timings cannot be negative")
        stragglers = straggler_delays_ms or {}
        events: list[ProjectionEvent] = []
        completions: dict[str, float] = {}
        shared_clock = 0.0
        for rank_id, duration in rank_compute_ms.items():
            start = 0.0 if resource_mode == "independent" else shared_clock
            events.append(
                ProjectionEvent("rank_compute_start", start, rank_id=rank_id, layer_id=layer_id)
            )
            completion = start + duration + max(stragglers.get(rank_id, 0.0), 0.0)
            if rank_id == failure_rank:
                failure_time = start + duration / 2
                events.append(
                    ProjectionEvent(
                        "rank_failure", failure_time, rank_id=rank_id, layer_id=layer_id
                    )
                )
                completion += max(rejoin_delay_ms, 0.0)
                events.append(
                    ProjectionEvent("rank_rejoin", completion, rank_id=rank_id, layer_id=layer_id)
                )
            events.append(
                ProjectionEvent(
                    "rank_compute_complete", completion, rank_id=rank_id, layer_id=layer_id
                )
            )
            completions[rank_id] = completion
            if resource_mode == "shared":
                shared_clock = completion
        slowest_rank = max(completions, key=completions.__getitem__)
        clock = max(completions.values())
        compute_completion = clock
        collective_total = 0.0
        for collective_index, work in enumerate(collectives):
            if set(work.rank_ids) != set(rank_compute_ms):
                raise ValueError("collective ranks differ from the layer compute group")
            estimate = estimate_collective(
                operation=work.operation,
                algorithm=work.algorithm,
                rank_count=len(work.rank_ids),
                payload_bytes=work.payload_bytes,
                network=network,
                seed=self.seed + collective_index,
            )
            events.append(
                ProjectionEvent(
                    "collective_step_start",
                    clock,
                    layer_id=layer_id,
                    collective_id=work.collective_id,
                    step=0 if estimate.steps else None,
                    details={"phase": work.phase, "algorithm": work.algorithm},
                )
            )
            for step_index, step in enumerate(estimate.critical_path):
                events.append(
                    ProjectionEvent(
                        "collective_transfer_start",
                        clock,
                        layer_id=layer_id,
                        collective_id=work.collective_id,
                        step=step_index,
                        details={"payload_bytes": estimate.step_payload_bytes},
                    )
                )
                clock += step["completion_delta_ms"]
                events.append(
                    ProjectionEvent(
                        "collective_transfer_complete",
                        clock,
                        layer_id=layer_id,
                        collective_id=work.collective_id,
                        step=step_index,
                    )
                )
                events.append(
                    ProjectionEvent(
                        "collective_step_complete",
                        clock,
                        layer_id=layer_id,
                        collective_id=work.collective_id,
                        step=step_index,
                    )
                )
            collective_total += estimate.completion_time_ms
            events.append(
                ProjectionEvent(
                    "collective_complete",
                    clock,
                    layer_id=layer_id,
                    collective_id=work.collective_id,
                    details=estimate.payload(),
                )
            )
        events.append(ProjectionEvent("layer_complete", clock, layer_id=layer_id))
        return ProjectionResult(
            completion_time_ms=clock,
            compute_completion_time_ms=compute_completion,
            collective_time_ms=collective_total,
            events=sorted(events, key=lambda item: item.timestamp_ms),
            slowest_rank=slowest_rank,
            resource_mode=resource_mode,
        )

    def project_pipeline_hop(
        self,
        *,
        start_time_ms: float,
        payload_bytes: int,
        network: NetworkProfile,
        source_stage: int,
        destination_stage: int,
    ) -> tuple[float, list[ProjectionEvent]]:
        duration = network.one_way_latency_ms + _transfer_ms(payload_bytes, network.bandwidth_mbps)
        end = start_time_ms + duration
        return end, [
            ProjectionEvent(
                "pipeline_hop_start",
                start_time_ms,
                details={
                    "source_stage": source_stage,
                    "destination_stage": destination_stage,
                    "payload_bytes": payload_bytes,
                },
            ),
            ProjectionEvent(
                "pipeline_hop_complete",
                end,
                details={
                    "source_stage": source_stage,
                    "destination_stage": destination_stage,
                    "payload_bytes": payload_bytes,
                },
            ),
        ]


def validate_projector() -> dict[str, Any]:
    projector = EventDrivenProjector(seed=17)
    one = projector.project_layer(
        layer_id=0,
        rank_compute_ms={"rank-0": 3.5},
        collectives=[],
        network=NETWORK_PROFILES["same_gpu_logical"],
    )
    two_network = NetworkProfile("analytical", 1.0, 1_000, 0.0)
    two_payload = 1_000_000
    two = estimate_collective(
        operation="all_reduce_sum",
        algorithm="ring",
        rank_count=2,
        payload_bytes=two_payload,
        network=two_network,
    )
    expected_two = 2 * (1.0 + _transfer_ms(two_payload / 2, 1_000))
    four = estimate_collective(
        operation="all_reduce_sum",
        algorithm="ring",
        rank_count=4,
        payload_bytes=400,
        network=NetworkProfile("latency-only", 0.25, None),
    )
    independent = projector.project_layer(
        layer_id=0,
        rank_compute_ms={f"rank-{rank}": float(rank + 1) for rank in range(4)},
        collectives=[],
        network=NETWORK_PROFILES["same_gpu_logical"],
        resource_mode="independent",
    )
    shared = projector.project_layer(
        layer_id=0,
        rank_compute_ms={f"rank-{rank}": float(rank + 1) for rank in range(4)},
        collectives=[],
        network=NETWORK_PROFILES["same_gpu_logical"],
        resource_mode="shared",
    )
    checks = {
        "one_rank_no_collective": math.isclose(one.completion_time_ms, 3.5),
        "two_rank_analytical_all_reduce": math.isclose(
            two.completion_time_ms, expected_two, rel_tol=1e-12
        ),
        "four_rank_ring_steps": four.steps == 6,
        "four_rank_ring_latency": math.isclose(four.completion_time_ms, 1.5),
        "independent_resource_uses_maximum": math.isclose(independent.completion_time_ms, 4.0),
        "shared_resource_serialises": math.isclose(shared.completion_time_ms, 10.0),
        "deterministic_seed": estimate_collective(
            operation="all_reduce_sum",
            algorithm="ring",
            rank_count=4,
            payload_bytes=400,
            network=NetworkProfile("jitter", 1.0, 1_000, 0.2),
            seed=3,
        ).completion_time_ms
        == estimate_collective(
            operation="all_reduce_sum",
            algorithm="ring",
            rank_count=4,
            payload_bytes=400,
            network=NetworkProfile("jitter", 1.0, 1_000, 0.2),
            seed=3,
        ).completion_time_ms,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "two_rank_expected_ms": expected_two,
        "two_rank_actual_ms": two.completion_time_ms,
        "four_rank_steps": four.steps,
        "shared_completion_ms": shared.completion_time_ms,
        "independent_completion_ms": independent.completion_time_ms,
    }


def break_even_latency_ms(
    *,
    whole_layer_latency_ms: float,
    rank_compute_latency_ms: float,
    rank_count: int,
    payload_bytes_per_collective: int,
    collective_count: int,
    algorithm: str,
    bandwidth_mbps: float | None,
) -> float:
    """Solve the largest non-negative one-way latency that still improves latency."""

    zero_latency = estimate_collective(
        operation="all_reduce_sum",
        algorithm=algorithm,
        rank_count=rank_count,
        payload_bytes=payload_bytes_per_collective,
        network=NetworkProfile("zero", 0.0, bandwidth_mbps),
    )
    available = (
        whole_layer_latency_ms
        - rank_compute_latency_ms
        - (collective_count * zero_latency.completion_time_ms)
    )
    steps = zero_latency.steps * collective_count
    if available <= 0 or steps <= 0:
        return 0.0
    return available / steps


def minimum_bandwidth_mbps(
    *,
    whole_layer_latency_ms: float,
    rank_compute_latency_ms: float,
    rank_count: int,
    payload_bytes_per_collective: int,
    collective_count: int,
    algorithm: str,
    one_way_latency_ms: float,
) -> float | None:
    steps, step_bytes, _, _ = collective_shape(
        operation="all_reduce_sum",
        algorithm=algorithm,
        rank_count=rank_count,
        payload_bytes=payload_bytes_per_collective,
    )
    available = (
        whole_layer_latency_ms
        - rank_compute_latency_ms
        - (collective_count * steps * one_way_latency_ms)
    )
    if available <= 0:
        return None
    total_step_bytes = collective_count * steps * step_bytes
    return total_step_bytes * 8 / (available / 1_000) / 1_000_000


def synchronous_group_decision(
    *,
    existing_compute_ms: list[float],
    candidate_compute_ms: float,
    memory_feasibility_gain: float,
    added_collective_ms: float,
) -> dict[str, Any]:
    """Enforce that a weak rank joins only when predicted marginal benefit is positive."""

    if not existing_compute_ms or any(value <= 0 for value in existing_compute_ms):
        raise ValueError("existing rank service times must be positive")
    if candidate_compute_ms <= 0:
        raise ValueError("candidate service time must be positive")
    before = max(existing_compute_ms)
    after = max([*existing_compute_ms, candidate_compute_ms]) + max(added_collective_ms, 0.0)
    compute_gain = before - after
    marginal_benefit = compute_gain + memory_feasibility_gain
    join = marginal_benefit > 0
    return {
        "classification": "independent_rank_projection",
        "join_synchronous_tensor_group": join,
        "marginal_benefit": marginal_benefit,
        "group_latency_before_ms": before,
        "group_latency_after_ms": after,
        "scheduler_rule": (
            "do not place a rank in a synchronous collective group when predicted "
            "marginal benefit is non-positive"
        ),
        "recommended_role": (
            "synchronous_tensor_group" if join else "separate_expert_or_background_pipeline"
        ),
    }
