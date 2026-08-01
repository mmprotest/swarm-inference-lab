from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest


def _run() -> Path:
    value = os.environ.get("SWARM_EXPERIMENT_007_RUN")
    if not value:
        pytest.skip("set SWARM_EXPERIMENT_007_RUN to audit a completed Experiment 007 run")
    path = Path(value)
    if not path.is_dir():
        pytest.fail(f"Experiment 007 run does not exist: {path}")
    return path


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_experiment_007_complete_real_evidence() -> None:
    run = _run()
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["execution_mode"] == "heterogeneous-single-host-real-model"
    assert summary["universal_worker_abi_status"] == "PASS"
    assert summary["sglang_backend_status"] == "PASS"
    assert summary["cpu_rank_backend_status"] == "PASS"
    assert summary["llamacpp_backend_status"] == "PASS"
    assert summary["mixed_backend_correctness_status"] == "PASS"
    assert summary["planner_non_degradation_status"] == "PASS"
    assert summary["planner_regret_status"] == "PASS"
    required = {
        "sglang_baseline.csv",
        "mixed_backend_results.csv",
        "backend_boundary_metrics.csv",
        "speculative_results.csv",
        "cpu_expert_results.csv",
        "background_results.csv",
        "arm64_build.json",
        "arm64_protocol_results.json",
        "planner_regret.csv",
        "network_projection.csv",
        "availability_economics.csv",
        "contribution_frontier.csv",
        "correctness.json",
        "report.html",
    }
    assert all((run / name).is_file() for name in required)
    sglang = _csv(run / "sglang_baseline.csv")
    assert {(row["workload"], int(row["concurrency"])) for row in sglang} >= {
        ("short", 1),
        ("short", 4),
        ("short", 16),
        ("short", 64),
        ("long", 1),
        ("long", 4),
        ("long", 16),
    }
    mixed = _csv(run / "mixed_backend_results.csv")
    negative = next(row for row in mixed if row["route"] == "cuda-cpu-cuda-cpu")
    assert negative["exact_greedy_token_identity"] == "True"
    assert negative["forced_critical_path_classification"] in {"harmful", "useful"}
    speculative = _csv(run / "speculative_results.csv")
    assert {(row["weight_format"], int(row["draft_length"])) for row in speculative} == {
        (weight, length) for weight in ("Q8_0", "Q4_K_M") for length in (1, 2, 4, 8)
    }
    assert all(row["all_exact"] == "True" for row in speculative)
    frontier = _csv(run / "contribution_frontier.csv")
    pi = next(row for row in frontier if row["device_profile"] == "raspberry_pi_5_class")
    assert pi["classification"] == "projected_device_profile"
    assert pi["raspberry_pi_performance"] == "unproven"


def test_experiment_007_positive_claim_matches_measured_gate() -> None:
    run = _run()
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    measured_positive = any(summary["positive_roles"].values())
    assert (summary["positive_cpu_contribution_status"] == "PASS") == measured_positive
    if not measured_positive:
        assert summary["overall_status"] == "PARTIAL_PASS"
