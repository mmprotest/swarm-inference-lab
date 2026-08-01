from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def test_completed_experiment_007_correction_artifact() -> None:
    configured = os.environ.get("SWARM_EXPERIMENT_007_CORRECTIONS_RUN")
    if not configured:
        pytest.skip("set SWARM_EXPERIMENT_007_CORRECTIONS_RUN for artifact validation")
    run = Path(configured)
    required = {
        "config.requested.yaml",
        "config.resolved.yaml",
        "environment.json",
        "git.json",
        "original_experiment_reference.json",
        "superseded_results.json",
        "moe_routing_corpus_manifest.json",
        "moe_routing_histogram.csv",
        "moe_execution_plans.json",
        "moe_matched_results.csv",
        "moe_timing_breakdown.csv",
        "moe_correctness.csv",
        "moe_memory_results.csv",
        "moe_controlled_coverage.csv",
        "background_window_results.csv",
        "background_token_events.jsonl",
        "background_gpu_metrics.csv",
        "background_cpu_metrics.csv",
        "background_combined_metrics.csv",
        "background_correctness.csv",
        "planner_calibration.csv",
        "planner_held_out_results.csv",
        "planner_regret.csv",
        "summary.json",
        "report.html",
    }
    missing = sorted(name for name in required if not (run / name).is_file())
    assert not missing
    empty = sorted(name for name in required if (run / name).stat().st_size == 0)
    assert not empty
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    for key in (
        "cpu_expert_matched_baseline_status",
        "cpu_expert_active_dispatch_status",
        "cpu_expert_memory_offload_status",
        "cpu_expert_positive_performance_status",
        "background_fixed_window_status",
        "background_token_accounting_status",
        "background_positive_contribution_status",
        "planner_held_out_evaluation_status",
        "planner_regret_status",
        "corrected_experiment_007_status",
    ):
        assert key in summary
    assert summary["corrected_experiment_007_status"] in {"PASS", "PARTIAL_PASS"}
    assert summary["superseded_metrics_used_by_planner"] is False
    charts = {
        "cpu_expert_calls_by_placement.png",
        "cpu_dispatch_fraction.png",
        "matched_expert_latency.png",
        "matched_expert_throughput_retained.png",
        "gpu_memory_saved_by_expert_count.png",
        "expert_timing_breakdown.png",
        "fixed_window_gpu_throughput.png",
        "fixed_window_cpu_throughput.png",
        "fixed_window_combined_throughput.png",
        "gpu_p95_interference.png",
        "gpu_throughput_interference.png",
        "combined_gain.png",
        "planner_prediction_vs_actual.png",
        "planner_held_out_regret.png",
    }
    assert charts == {path.name for path in (run / "charts").glob("*.png")}
