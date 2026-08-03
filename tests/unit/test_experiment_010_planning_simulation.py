from __future__ import annotations

from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_010.batching import (
    BatchingPolicy,
    RoutedRequest,
    batching_summary,
    make_routing_batches,
)
from swarm_inference.experiments.experiment_010.colibri_workloads import (
    _counter_delta,
    prefill_context_supported,
)
from swarm_inference.experiments.experiment_010.memory_analysis import (
    amdahl_gate,
    page_fault_candidate_validity,
    prefetch_idle_window_budget,
    reuse_distance_curve,
)
from swarm_inference.experiments.experiment_010.phase10_analysis import _plan
from swarm_inference.experiments.experiment_010.planner import PositiveUtilityPlanner
from swarm_inference.experiments.experiment_010.runner import (
    FULL_RUN_PREREQUISITES,
    assess_full_run_completeness,
)
from swarm_inference.experiments.experiment_010.schemas import (
    ExecutionStrategy,
    Experiment010Mode,
    PlannerCandidate,
    PlannerObjective,
    ServicePhase,
)
from swarm_inference.simulation.expert_model import (
    calibrate_expert_simulator,
    deterministic_calibration_split,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


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


def test_level_b_current_run_required() -> None:
    assert "level_b_current_workload" in FULL_RUN_PREREQUISITES
    reproduction = (
        REPOSITORY_ROOT
        / "experiments"
        / "010_hardware_in_loop_virtual_swarm_closure"
        / "reproduce.ps1"
    ).read_text(encoding="utf-8")
    assert "008_single_host_adaptive_moe_saturation\\reproduce.ps1" in reproduction
    assert "experiment-010-correction-work\\phase-14\\level-b-current" in reproduction
    assert '"-Configuration", "A"' in reproduction


def test_full_run_incomplete_when_level_b_missing() -> None:
    prerequisites = {name: True for name in FULL_RUN_PREREQUISITES}
    prerequisites["level_b_current_workload"] = False
    result = assess_full_run_completeness(
        mode=Experiment010Mode.FULL,
        prerequisites=prerequisites,
        reasons={"level_b_current_workload": "current Level B model path is unavailable"},
    )
    assert result["status"] == "INCOMPLETE_FULL_RUN"
    assert result["full_complete"] is False
    assert result["missing_prerequisites"] == [
        {
            "prerequisite": "level_b_current_workload",
            "complete": False,
            "reason": "current Level B model path is unavailable",
        }
    ]


def test_full_run_completeness() -> None:
    result = assess_full_run_completeness(
        mode=Experiment010Mode.FULL,
        prerequisites={name: True for name in FULL_RUN_PREREQUISITES},
    )
    assert result["status"] == "FULL_COMPLETE"
    assert result["full_complete"] is True
    assert result["missing_prerequisites"] == []


def test_reuse_distance_candidates_follow_measured_thresholds(tmp_path) -> None:
    trace = tmp_path / "route.trace"
    trace.write_text(
        "0 0 0 1:0.5 2:0.5\n"
        "1 0 0 3:0.5 1:0.5\n"
        "2 0 0 2:0.5 1:0.5\n",
        encoding="utf-8",
    )
    rows, summary = reuse_distance_curve([trace], expert_bytes=64)
    assert summary["threshold_slots"]["p50"] == 2
    assert summary["candidate_slots"] == [1, 2, 3, 64]
    assert all(row["candidate_basis"].startswith("measured_reuse") for row in rows)


def test_page_fault_gate_rejects_nonresident_cache_hits() -> None:
    result = page_fault_candidate_validity(
        resident_cache_hits=4,
        nonresident_cache_hits=9,
        pagefile_read_bytes=None,
        commit_pressure_fraction=0.5,
    )
    assert result["valid_performance_candidate"] is False
    assert "predominantly nonresident" in result["invalidation_reasons"][0]


def test_decode_prefetch_must_fit_idle_window() -> None:
    result = prefetch_idle_window_budget(
        phase="decode",
        layer_id=3,
        available_idle_window_ns=1_000_000,
        effective_bandwidth_bytes_per_second=1_000_000,
        proposed_prefetch_bytes=2_000,
        subsequently_consumed_bytes=2_000,
        demand_read_interference_ns=0,
        eviction_bytes=0,
    )
    assert result["maximum_prefetch_bytes"] == 1_000
    assert result["accepted"] is False


def test_amdahl_gate_rejects_microbenchmark_only_gain() -> None:
    result = amdahl_gate(
        optimization="compression",
        baseline_end_to_end_ns=1_000,
        baseline_affected_ns=100,
        optimized_affected_ns=50,
        optimized_end_to_end_ns=1_000,
    )
    assert result["measured_kernel_gain"] == 2.0
    assert result["measured_end_to_end_gain"] == 1.0
    assert result["accepted"] is False


def test_worker_counter_delta_uses_documented_zero_initial_value() -> None:
    assert _counter_delta({"logical_cache_hits": 17}, {}, "logical_cache_hits") == 17
    assert (
        _counter_delta(
            {"logical_cache_hits": 3},
            {"logical_cache_hits": 100},
            "logical_cache_hits",
        )
        == 3
    )
    assert _counter_delta({}, {}, "logical_cache_hits") is None


def test_prefill_context_capability_does_not_infer_32k_from_workspace() -> None:
    assert prefill_context_supported(context_length=8192, advertised_context_limit=4096)
    assert not prefill_context_supported(context_length=32768, advertised_context_limit=4096)
    assert prefill_context_supported(context_length=32768, advertised_context_limit=32768)


def test_measured_phase_plan_rejects_faster_inexact_candidate() -> None:
    plan = _plan(
        phase="decode",
        objective="max_decode_throughput",
        candidates=[
            {
                "configuration": "local",
                "decode_tokens_per_second": 5.0,
                "eligible": True,
            },
            {
                "configuration": "fast_inexact",
                "decode_tokens_per_second": 8.0,
                "eligible": False,
            },
        ],
        metric="decode_tokens_per_second",
        maximize=True,
    )
    assert plan["selected_candidate"] == "local"
