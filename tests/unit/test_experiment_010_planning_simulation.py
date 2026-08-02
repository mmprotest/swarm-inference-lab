from __future__ import annotations

from typing import Any

from swarm_inference.experiments.experiment_010.batching import (
    BatchingPolicy,
    RoutedRequest,
    batching_summary,
    make_routing_batches,
)
from swarm_inference.experiments.experiment_010.planner import PositiveUtilityPlanner
from swarm_inference.experiments.experiment_010.schemas import (
    ExecutionStrategy,
    PlannerCandidate,
    PlannerObjective,
    ServicePhase,
)
from swarm_inference.simulation.expert_model import (
    calibrate_expert_simulator,
    deterministic_calibration_split,
)


def _candidate(
    candidate_id: str,
    strategy: ExecutionStrategy,
    utility: float,
    *,
    phase: ServicePhase = ServicePhase.DECODE,
    capacity_required: bool = False,
) -> PlannerCandidate:
    return PlannerCandidate(
        candidate_id=candidate_id,
        phase=phase,
        strategy=strategy,
        workers=[] if strategy == ExecutionStrategy.LOCAL_WHOLE_EXPERT else ["worker"],
        objective=PlannerObjective.MAX_DECODE_THROUGHPUT,
        predicted_utility=utility,
        lower_confidence_bound=utility - 0.01,
        capacity_required=capacity_required,
        explanation=[f"{candidate_id} evidence"],
    )


def _select(candidates: list[PlannerCandidate], phase: ServicePhase = ServicePhase.DECODE) -> Any:
    return PositiveUtilityPlanner().select(
        candidates,
        phase=phase,
        objective=PlannerObjective.MAX_DECODE_THROUGHPUT,
    )


def test_prefill_decode_plan_separation() -> None:
    candidates = [
        _candidate(
            "prefill-micro",
            ExecutionStrategy.COALESCED_MICROSHARDS,
            0.4,
            phase=ServicePhase.PREFILL,
        ),
        _candidate(
            "prefill-idle",
            ExecutionStrategy.IDLE,
            0,
            phase=ServicePhase.PREFILL,
        ),
        _candidate("decode-local", ExecutionStrategy.LOCAL_WHOLE_EXPERT, 0.3),
        _candidate("decode-idle", ExecutionStrategy.IDLE, 0),
    ]
    prefill = _select(candidates, ServicePhase.PREFILL).plan
    decode = _select(candidates, ServicePhase.DECODE).plan
    assert prefill.selected_strategy == ExecutionStrategy.COALESCED_MICROSHARDS
    assert decode.selected_strategy == ExecutionStrategy.LOCAL_WHOLE_EXPERT


def test_expert_union_deduplication() -> None:
    requests = [
        RoutedRequest("a", 0, (1, 2, 3), 100),
        RoutedRequest("b", 10, (2, 3, 4), 100),
        RoutedRequest("c", 20, (1, 3, 4), 100),
    ]
    summary = batching_summary(
        make_routing_batches(
            requests,
            policy=BatchingPolicy.EXPERT_OVERLAP,
            maximum_batch_size=3,
            maximum_queue_delay_ns=100,
        )
    )
    assert summary["expert_selections_before_union"] == 9
    assert summary["unique_experts_after_union"] == 4
    assert summary["deduplication_ratio"] > 0.5


def test_batch_queue_delay_limit() -> None:
    requests = [
        RoutedRequest(f"r{index}", index * 100, (index % 3, (index + 1) % 3), 10)
        for index in range(8)
    ]
    batches = make_routing_batches(
        requests,
        policy=BatchingPolicy.EXPERT_OVERLAP,
        maximum_batch_size=4,
        maximum_queue_delay_ns=250,
    )
    assert max(batch.metrics()["queue_delay_ns_max"] for batch in batches) <= 250


def test_planner_can_select_idle() -> None:
    result = _select(
        [
            _candidate("remote", ExecutionStrategy.REMOTE_WHOLE_EXPERT, -0.2),
            _candidate("idle", ExecutionStrategy.IDLE, 0),
        ]
    )
    assert result.plan.selected_strategy == ExecutionStrategy.IDLE


def test_planner_can_select_whole_expert() -> None:
    result = _select(
        [
            _candidate("whole", ExecutionStrategy.REMOTE_WHOLE_EXPERT, 0.2),
            _candidate("idle", ExecutionStrategy.IDLE, 0),
        ]
    )
    assert result.plan.selected_strategy == ExecutionStrategy.REMOTE_WHOLE_EXPERT


def test_planner_can_select_microshard() -> None:
    result = _select(
        [
            _candidate("micro", ExecutionStrategy.ASYMMETRIC_MICROSHARDS, 0.3),
            _candidate("whole", ExecutionStrategy.REMOTE_WHOLE_EXPERT, 0.1),
            _candidate("idle", ExecutionStrategy.IDLE, 0),
        ]
    )
    assert result.plan.selected_strategy == ExecutionStrategy.ASYMMETRIC_MICROSHARDS


def test_planner_rejects_negative_utility() -> None:
    result = _select(
        [
            _candidate("negative", ExecutionStrategy.EQUAL_MICROSHARDS, -0.01),
            _candidate("idle", ExecutionStrategy.IDLE, 0),
        ]
    )
    rejected = {row["candidate_id"]: row for row in result.plan.rejected}
    assert "non-positive predicted marginal utility" in rejected["negative"]["reasons"]


def test_planner_capacity_exception() -> None:
    result = _select(
        [
            _candidate(
                "capacity",
                ExecutionStrategy.REMOTE_WHOLE_EXPERT,
                -0.1,
                capacity_required=True,
            ),
            _candidate("idle", ExecutionStrategy.IDLE, 0),
        ]
    )
    assert result.plan.selected_candidate_id == "capacity"
    assert result.plan.capacity_exception is True


def _simulator_rows() -> list[dict[str, Any]]:
    rows = []
    for index in range(12):
        compute = 500_000 + index * 130_000
        transport = 75_000 + (index % 4) * 30_000
        serialisation = 20_000 + (index % 3) * 5_000
        total = 80_000 + 1.25 * compute + 0.8 * transport + 0.5 * serialisation
        rows.append(
            {
                "configuration_id": f"config-{index:02d}",
                "workload_id": "fixed-replay",
                "worker_compute_ns": compute,
                "tcp_transport_ns": transport,
                "serialisation_ns": serialisation,
                "measured_total_ns": total,
                "measured_throughput": 1e9 / total,
                "measured_p95_latency_ms": total / 1e6,
                "verified_tokens": 1,
            }
        )
    return rows


def test_simulator_calibration_split() -> None:
    calibration, validation = deterministic_calibration_split(_simulator_rows())
    calibration_ids = {row["configuration_id"] for row in calibration}
    validation_ids = {row["configuration_id"] for row in validation}
    assert calibration_ids
    assert validation_ids
    assert calibration_ids.isdisjoint(validation_ids)


def test_simulator_heldout_prediction() -> None:
    model, validation = calibrate_expert_simulator(_simulator_rows())
    assert model.validation_configuration_ids
    assert max(row["throughput_error_fraction"] for row in validation) <= 0.10
    assert set(model.calibration_configuration_ids).isdisjoint(model.validation_configuration_ids)


def test_simulator_ranking_agreement() -> None:
    model, _ = calibrate_expert_simulator(_simulator_rows())
    assert model.validation["plan_ranking_agreement"] >= 0.8
    assert model.validation["plan_ranking_agreement_pass"] is True


def test_simulator_regret() -> None:
    model, _ = calibrate_expert_simulator(_simulator_rows())
    assert model.validation["planner_regret"] <= 0.05
    assert model.validation["planner_regret_pass"] is True
