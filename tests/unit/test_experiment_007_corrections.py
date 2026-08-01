from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from swarm_inference.config.experiment_007_corrections import (
    load_experiment_007_corrections_config,
)
from swarm_inference.experiments.experiment_007_background_correction import (
    LaneState,
    MeasurementWindow,
    RequestObservation,
    TokenCompletion,
    WorkloadFixture,
    _candidate_token_ids,
    _new_tokens,
    aggregate_fixed_window_results,
    combined_throughput,
    count_window_tokens,
    lane_metrics,
    parse_sse_json_lines,
    workload_fixture_hash,
)
from swarm_inference.experiments.experiment_007_corrected_planner import (
    corrected_planner_points,
    evaluate_corrected_planner,
    split_calibration_and_held_out,
)
from swarm_inference.experiments.experiment_007_corrections import _corrected_status
from swarm_inference.experiments.experiment_007_moe_correction import (
    CANONICAL_EXECUTOR_ID,
    MoEExecutionPlan,
    _canonical_routing_corpus_hash,
    _stable_cosine_similarity,
    controlled_coverage_indices,
    measure_repeated,
    placement_dispatch_metrics,
    select_cpu_experts_corrected,
    valid_positive_expert_result,
    validate_matched_plans,
)


def _plan(
    *, cpu_ids: set[int], dtype: str = "bfloat16", executor: str = CANONICAL_EXECUTOR_ID
) -> MoEExecutionPlan:
    return MoEExecutionPlan(
        layer_id=24,
        model_revision="revision",
        router_backend="cuda",
        shared_expert_backend="not_present_in_qwen3_30b_a3b",
        expert_backend_by_id={index: "cpu" if index in cpu_ids else "cuda" for index in range(4)},
        expert_format_by_id={
            index: dtype if index in cpu_ids else "bfloat16" for index in range(4)
        },
        batch_size=1,
        token_count=10_000,
        top_k=2,
        dtype="bfloat16",
        execution_profile="test",
        executor_id=executor,
    )


def test_correction_config_is_strict(repository_root: Path) -> None:
    config = load_experiment_007_corrections_config(
        repository_root / "configs" / "experiments" / "experiment_007_corrections.yaml"
    )
    assert config.cpu_expert.routing_tokens == 10_000
    assert config.cpu_expert.maximum_variability_epochs == 3
    assert config.background.measurement_seconds == 120
    assert config.background.repeats == 3


def test_large_cosine_diagnostic_is_bounded_and_uses_stable_reduction() -> None:
    reference = torch.linspace(-4.0, 4.0, 2_000_000, dtype=torch.float32)
    candidate = reference.clone()
    cosine = _stable_cosine_similarity(reference, candidate, chunk_elements=65_536)
    assert cosine == pytest.approx(1.0, abs=1e-12)
    assert -1.0 <= cosine <= 1.0


def test_routing_corpus_hash_is_canonical_across_mapping_order() -> None:
    first = {
        "hidden": torch.arange(16, dtype=torch.bfloat16).reshape(4, 4),
        "selected": torch.tensor([[2, 1], [0, 3]], dtype=torch.long),
    }
    second = {"selected": first["selected"].clone(), "hidden": first["hidden"].clone()}
    expected = _canonical_routing_corpus_hash(
        first,
        model_revision="revision",
        layer_id=24,
        seed=7007,
    )
    assert expected == _canonical_routing_corpus_hash(
        second,
        model_revision="revision",
        layer_id=24,
        seed=7007,
    )


def test_variability_retry_keeps_failed_epoch_evidence() -> None:
    class FakeExecutor:
        def __init__(self) -> None:
            # Warm-up, five unstable measurements, warm-up, three stable measurements,
            # then the captured validation observation.
            self.values = iter((1.0, 1.0, 3.0, 1.0, 3.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0))

        def execute(self, plan: object, prepared: object, *, capture_output: bool) -> object:
            del plan, prepared, capture_output
            value = next(self.values)
            return type("Observation", (), {"timings": {"total_layer_time_ms": value}})()

    stats, _validation = measure_repeated(  # type: ignore[arg-type]
        FakeExecutor(),
        object(),
        object(),
        warmup_iterations=1,
        minimum_repeats=3,
        maximum_repeats=5,
        maximum_epochs=2,
        maximum_cv=0.10,
    )
    assert stats.measurement_epochs == 2
    assert stats.total_measured_repeats == 8
    assert len(stats.unstable_epoch_coefficients_of_variation) == 1
    assert stats.coefficient_of_variation == pytest.approx(0.0)


def test_matched_moe_plans_use_one_executor_and_only_change_placement() -> None:
    baseline = _plan(cpu_ids=set())
    hybrid = _plan(cpu_ids={1})
    validate_matched_plans(baseline, hybrid)
    assert baseline.executor_id == hybrid.executor_id == CANONICAL_EXECUTOR_ID
    assert baseline.token_count == hybrid.token_count
    assert baseline.top_k == hybrid.top_k
    assert baseline.dtype == hybrid.dtype
    assert baseline.expert_backend_by_id[1] == "cuda"
    assert hybrid.expert_backend_by_id[1] == "cpu"


def test_different_executor_invalidates_matched_moe_benchmark() -> None:
    with pytest.raises(ValueError, match="executor IDs"):
        validate_matched_plans(_plan(cpu_ids=set()), _plan(cpu_ids={1}, executor="other"))


def test_bf16_matched_plan_rejects_offloaded_dtype_difference() -> None:
    baseline = _plan(cpu_ids=set())
    hybrid = _plan(cpu_ids={1}, dtype="float16")
    with pytest.raises(ValueError, match="format differs"):
        validate_matched_plans(baseline, hybrid)


def test_dispatch_count_and_fraction_are_computed_from_frozen_router_trace() -> None:
    selected = torch.tensor([[0, 1], [1, 2], [3, 1]], dtype=torch.long)
    metrics = placement_dispatch_metrics(selected, [1, 3])
    assert metrics["expected_cpu_dispatch_count"] == 4
    assert metrics["expected_cpu_dispatch_fraction"] == pytest.approx(4 / 6)
    assert metrics["unique_cpu_experts_selected"] == 2


def test_zero_call_placement_is_visible_and_cannot_pass() -> None:
    selected = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    metrics = placement_dispatch_metrics(selected, [3])
    assert metrics["expected_cpu_dispatch_count"] == 0
    row = {
        "benchmark_mode": "natural_routing",
        "weight_format": "bfloat16",
        "cpu_expert_calls": 0,
        "cpu_dispatch_fraction": 0.0,
        "gpu_memory_saved_bytes": 1024,
        "output_correctness_passed": True,
        "matched_baseline_used": True,
        "throughput_retained_fraction": 2.0,
        "coefficient_of_variation": 0.01,
    }
    assert not valid_positive_expert_result(
        row, minimum_dispatch_fraction=0.01, minimum_retained_fraction=0.70
    )


def test_quantised_and_unstable_expert_rows_cannot_replace_bf16_primary() -> None:
    row = {
        "benchmark_mode": "natural_routing",
        "weight_format": "int8",
        "cpu_expert_calls": 100,
        "cpu_dispatch_fraction": 0.05,
        "gpu_memory_saved_bytes": 1024,
        "output_correctness_passed": True,
        "matched_baseline_used": True,
        "throughput_retained_fraction": 0.9,
        "coefficient_of_variation": 0.01,
    }
    assert not valid_positive_expert_result(
        row, minimum_dispatch_fraction=0.01, minimum_retained_fraction=0.70
    )
    row["weight_format"] = "bfloat16"
    row["coefficient_of_variation"] = 0.11
    assert not valid_positive_expert_result(
        row, minimum_dispatch_fraction=0.01, minimum_retained_fraction=0.70
    )


def test_placement_policies_are_deterministic_and_use_frequency() -> None:
    counts = {index: index for index in range(8)}
    assert select_cpu_experts_corrected(
        "coldest_experts_on_cpu", count=2, num_experts=8, routing_counts=counts, seed=7
    ) == [0, 1]
    assert select_cpu_experts_corrected(
        "hottest_experts_on_cpu", count=2, num_experts=8, routing_counts=counts, seed=7
    ) == [6, 7]
    first = select_cpu_experts_corrected(
        "random_experts_on_cpu", count=4, num_experts=8, routing_counts=counts, seed=7
    )
    second = select_cpu_experts_corrected(
        "random_experts_on_cpu", count=4, num_experts=8, routing_counts=counts, seed=7
    )
    assert first == second


def test_controlled_coverage_selects_real_trace_rows_without_forcing_router() -> None:
    selected = torch.tensor([[0, 1], [2, 1], [3, 2], [0, 3]], dtype=torch.long)
    indices = controlled_coverage_indices(selected, [2, 3], minimum_calls=1)
    assert indices
    covered = selected.index_select(0, torch.tensor(indices)).flatten().tolist()
    assert 2 in covered and 3 in covered


def _request(
    *,
    lane: str,
    success: bool = True,
    times: tuple[float, ...] = (9.0, 10.0, 15.0, 20.0),
) -> RequestObservation:
    return RequestObservation(
        lane=lane,  # type: ignore[arg-type]
        request_id=f"{lane}-request",
        fixture_id=f"{lane}-fixture",
        prompt_hash="hash",
        prompt_token_ids=[1, 2],
        requested_output_tokens=len(times),
        admitted_monotonic=8.0,
        started_monotonic=8.5,
        completed_monotonic=21.0,
        queue_delay_ms=500.0,
        success=success,
        error=None,
        token_events=[
            TokenCompletion(token_id=index, completion_monotonic=value, sequence_index=index)
            for index, value in enumerate(times)
        ],
        artifact_hash="artifact",
        artifact_revision="revision",
    )


def test_fixed_window_excludes_pre_and_post_window_tokens() -> None:
    window = MeasurementWindow(0.0, 10.0, 20.0, 30.0)
    counts = count_window_tokens([_request(lane="gpu")], window)
    assert counts == {
        "tokens_before_window": 1,
        "tokens_inside_window": 2,
        "tokens_after_window": 1,
    }
    metrics = lane_metrics("gpu", [_request(lane="gpu")], window, LaneState())
    assert metrics["verified_output_tokens"] == 2
    assert metrics["verified_tokens_per_second"] == pytest.approx(0.2)


def test_failed_request_tokens_are_not_verified() -> None:
    window = MeasurementWindow(0.0, 10.0, 20.0, 30.0)
    counts = count_window_tokens([_request(lane="cpu", success=False)], window)
    assert counts["tokens_inside_window"] == 0


def test_combined_throughput_uses_shared_window_formula() -> None:
    assert combined_throughput(1_200, 600, 120.0) == 15.0
    with pytest.raises(ValueError, match="positive"):
        combined_throughput(1, 1, 0)


def _window_row(
    *,
    arm: str,
    repeat: int,
    gpu_tps: float,
    cpu_tps: float,
    combined_tps: float,
    p95: float,
) -> dict[str, object]:
    return {
        "traffic_mode": "closed_loop",
        "gpu_concurrency": 4,
        "cpu_concurrency": 0 if arm == "gpu_only" else 2,
        "open_loop_arrival_rate_rps": None,
        "arm": arm,
        "repeat": repeat,
        "measurement_window_seconds": 120.0,
        "gpu_verified_tps": gpu_tps,
        "cpu_verified_tps": cpu_tps,
        "combined_verified_tps": combined_tps,
        "gpu_latency_p95_ms": p95,
        "gpu_verified_output_tokens": gpu_tps * 120,
        "cpu_verified_output_tokens": cpu_tps * 120,
        "denominator_kind": "shared_fixed_measurement_window",
    }


def test_repeat_aggregation_and_background_acceptance_use_medians() -> None:
    rows = []
    for repeat in range(3):
        rows.append(
            _window_row(
                arm="gpu_only",
                repeat=repeat,
                gpu_tps=100.0,
                cpu_tps=0.0,
                combined_tps=100.0,
                p95=100.0,
            )
        )
        rows.append(
            _window_row(
                arm="gpu_plus_cpu",
                repeat=repeat,
                gpu_tps=98.0,
                cpu_tps=20.0,
                combined_tps=118.0,
                p95=103.0,
            )
        )
    result = aggregate_fixed_window_results(
        rows,
        minimum_combined_gain_fraction=0.10,
        maximum_gpu_p95_increase_fraction=0.05,
        maximum_gpu_throughput_decrease_fraction=0.05,
    )
    assert len(result) == 1
    assert result[0]["repeats"] == 3
    assert result[0]["combined_gain_fraction"] == pytest.approx(0.18)
    assert result[0]["gpu_throughput_change_fraction"] == pytest.approx(-0.02)
    assert result[0]["gpu_p95_latency_change_fraction"] == pytest.approx(0.03)
    assert result[0]["positive_contribution_pass"] is True


def test_sse_parser_requires_json_objects_and_stops_at_done() -> None:
    rows = parse_sse_json_lines(
        [b'data: {"output_ids":[1]}\n', b'data: {"output_ids":[1,2]}\n', b"data: [DONE]\n"]
    )
    assert rows[-1]["output_ids"] == [1, 2]


def test_llamacpp_multi_token_stream_event_is_incremental() -> None:
    candidates, cumulative = _candidate_token_ids({"tokens": [17, 18]}, "cpu")
    assert cumulative is False
    assert _new_tokens([9], candidates, cumulative=cumulative) == [17, 18]


def test_sglang_output_ids_remain_cumulative() -> None:
    candidates, cumulative = _candidate_token_ids({"output_ids": [9, 17]}, "gpu")
    assert cumulative is True
    assert _new_tokens([9], candidates, cumulative=cumulative) == [17]


def _planner_moe_rows() -> list[dict[str, object]]:
    rows = []
    for count in (1, 2, 4, 8, 16):
        rows.append(
            {
                "arm": "hybrid_gpu_cpu",
                "weight_format": "bfloat16",
                "benchmark_mode": "natural_routing",
                "executor_id": CANONICAL_EXECUTOR_ID,
                "matched_baseline_used": True,
                "placement_policy": "hottest_experts_on_cpu",
                "cpu_expert_count": count,
                "cpu_dispatch_fraction": count / 128,
                "cpu_expert_calls": count * 100,
                "gpu_memory_saved_bytes": count * 1000,
                "baseline_gpu_expert_weight_bytes": 128_000,
                "throughput_retained_fraction": 0.9 - count / 200,
                "output_correctness_passed": True,
            }
        )
    return rows


def _planner_background_rows() -> list[dict[str, object]]:
    return [
        {
            "traffic_mode": "closed_loop",
            "fixed_window_formula_used": True,
            "gpu_concurrency": gpu,
            "cpu_concurrency": cpu,
            "combined_gain_fraction": 0.2 - gpu / 200,
            "gpu_throughput_change_fraction": -0.01,
            "gpu_p95_latency_change_fraction": 0.01,
        }
        for gpu in (1, 4, 16)
        for cpu in (1, 2, 4)
    ]


def test_planner_calibration_and_held_out_are_disjoint() -> None:
    points = corrected_planner_points(
        _planner_moe_rows(),  # type: ignore[arg-type]
        _planner_background_rows(),  # type: ignore[arg-type]
        minimum_expert_retained_fraction=0.70,
    )
    calibration, held_out = split_calibration_and_held_out(points)
    assert {item.point_id for item in calibration}.isdisjoint({item.point_id for item in held_out})
    calibration_rows, held_out_rows, regret, _model = evaluate_corrected_planner(
        points, maximum_regret_fraction=0.10
    )
    assert calibration_rows and held_out_rows
    assert all(row["observed_before_prediction"] is False for row in held_out_rows)
    assert regret[0]["held_out_observations_used_for_selection"] is False


def test_planner_rejects_superseded_metrics() -> None:
    background = _planner_background_rows()
    background[0]["fixed_window_formula_used"] = False
    with pytest.raises(ValueError, match="superseded"):
        corrected_planner_points(
            _planner_moe_rows(),  # type: ignore[arg-type]
            background,  # type: ignore[arg-type]
            minimum_expert_retained_fraction=0.70,
        )


def test_corrected_status_distinguishes_negative_from_invalid() -> None:
    base = {
        "cpu_expert_matched_baseline_status": "PASS",
        "cpu_expert_active_dispatch_status": "PASS",
        "cpu_expert_memory_offload_status": "PASS",
        "background_fixed_window_status": "PASS",
        "background_token_accounting_status": "PASS",
        "planner_held_out_evaluation_status": "PASS",
        "planner_regret_status": "PASS",
        "cpu_expert_positive_performance_status": "NOT_USEFUL",
        "background_positive_contribution_status": "NOT_USEFUL",
    }
    assert _corrected_status(base) == "PARTIAL_PASS"
    assert _corrected_status({**base, "background_positive_contribution_status": "PASS"}) == "PASS"
    assert _corrected_status({**base, "background_token_accounting_status": "FAIL"}) == "FAIL"


def test_original_correction_document_exposes_both_superseded_labels(
    repository_root: Path,
) -> None:
    source = (repository_root / "docs" / "experiment-007-benchmark-corrections.md").read_text(
        encoding="utf-8"
    )
    assert "superseded_unmatched_cpu_expert_result" in source
    assert "superseded_fixed_job_background_result" in source
    assert "189.387861%" in source


def test_correction_launcher_preserves_environment_and_cleans_up(repository_root: Path) -> None:
    source = (repository_root / "scripts" / "run_experiment_007_corrections.ps1").read_text(
        encoding="utf-8"
    )
    lowered = source.lower()
    assert "--no-sync" in lowered
    assert "uv sync" not in lowered
    assert "finally {" in source
    for parameter in (
        "$OriginalRun",
        "$SkipExpertFix",
        "$SkipBackgroundFix",
        "$Smoke",
        "$Resume",
        "$Profile",
        "$OutputRoot",
        "$KeepServers",
    ):
        assert parameter in source
    assert "exit $ExitCode" in source


def test_workload_fixture_is_immutable_for_matched_arms() -> None:
    fixture = WorkloadFixture("fixture", (1, 2, 3), 64, "hash")
    baseline = [fixture]
    paired = [fixture]
    assert baseline == paired
    assert baseline[0].prompt_token_ids == paired[0].prompt_token_ids
    assert workload_fixture_hash(baseline) == workload_fixture_hash(paired)
    changed = [WorkloadFixture("fixture", (1, 2, 3), 128, "hash")]
    assert workload_fixture_hash(baseline) != workload_fixture_hash(changed)


def test_report_source_starts_with_required_status_labels(repository_root: Path) -> None:
    source = (
        repository_root
        / "src"
        / "swarm_inference"
        / "experiments"
        / "experiment_007_correction_reporting.py"
    ).read_text(encoding="utf-8")
    for label in (
        "CPU expert benchmark validity:",
        "CPU expert memory offload:",
        "CPU expert positive performance:",
        "Background benchmark validity:",
        "Background positive contribution:",
        "Planner held-out evaluation:",
        "Corrected Experiment 007:",
    ):
        assert label in source


def test_superseded_json_contract_is_unambiguous() -> None:
    payload = {
        "label": "superseded_fixed_job_background_result",
        "historical_only": True,
    }
    encoded = json.dumps(payload)
    assert "superseded_fixed_job_background_result" in encoded
    assert payload["historical_only"] is True
