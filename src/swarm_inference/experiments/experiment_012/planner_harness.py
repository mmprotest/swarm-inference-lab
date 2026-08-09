"""H012-007 held-out evaluation of the evidence-calibrated topology planner."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_012.baseline_harness import (
    _sha256_file,
    _write_json,
)
from swarm_inference.experiments.experiment_012.branch_factor_harness import run_h012_005
from swarm_inference.experiments.experiment_012.topology_planner import (
    CANDIDATE_BRANCH_FACTORS,
    calibrate_from_summary,
    choose_profiles,
)
from swarm_inference.microworker_protocol import NETWORK_PROFILES

HYPOTHESIS_ID = "H012-007"
HOLDOUT_PROFILES = (
    "holdout_2ms_shaped",
    "holdout_10ms_shaped",
    "holdout_40ms_shaped",
)


def _load_trials(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def evaluate_planner(
    *,
    output_directory: Path,
    summary: dict[str, Any],
    decision_record: dict[str, Any],
    measured_trials: int,
) -> dict[str, Any]:
    """Evaluate sealed decisions against controls after all runs complete."""

    decisions = {row["network_profile"]: row for row in decision_record["decisions"]}
    evaluations: list[dict[str, Any]] = []
    for profile_name in HOLDOUT_PROFILES:
        cells = {
            int(row["branch_factor"]): row
            for row in summary["summaries"]
            if row["network_profile"] == profile_name and int(row["worker_count"]) == 128
        }
        if set(cells) != set(CANDIDATE_BRANCH_FACTORS):
            raise ValueError(f"incomplete held-out controls for {profile_name}")
        selected_b = int(decisions[profile_name]["selected_branch_factor"])
        selected_ms = float(cells[selected_b]["end_to_end_latency_p50_ms"])
        best_b, best_cell = min(
            cells.items(), key=lambda item: (item[1]["end_to_end_latency_p50_ms"], item[0])
        )
        best_ms = float(best_cell["end_to_end_latency_p50_ms"])
        evaluations.append(
            {
                "network_profile": profile_name,
                "selected_branch_factor": selected_b,
                "best_observed_branch_factor": best_b,
                "selected_observed_latency_p50_ms": selected_ms,
                "best_observed_latency_p50_ms": best_ms,
                "latency_regret_fraction": selected_ms / best_ms - 1.0,
                "within_ten_percent": selected_ms <= 1.10 * best_ms,
            }
        )

    trials = _load_trials(output_directory / "raw" / "trials.jsonl")
    measured = [row for row in trials if not row["warmup"]]
    expected_measured_count = (
        len(HOLDOUT_PROFILES) * len(CANDIDATE_BRANCH_FACTORS) * measured_trials
    )
    controls_correct = len(measured) == expected_measured_count and all(
        row["status"] == "ok"
        and row["correctness"]
        and row["actual"] == row["expected"]
        and row["actual_digest"] == row["expected_digest"]
        for row in measured
    )
    traces_structural = all(
        row["root_leaf_rpc_count"] == 0
        and row["root_direct_degree"] <= row["branch_factor"]
        and row["total_messages"] == 2 * row["worker_count"]
        and row["observed_network_depth"] == row["hierarchy_depth"]
        for row in measured
    )
    decisions_precede_measurement = bool(measured) and int(decision_record["sealed_unix_ns"]) < min(
        int(row["measured_unix_ns"]) for row in measured
    )
    regrets = [float(row["latency_regret_fraction"]) for row in evaluations]
    criteria = {
        "endpoint_choices_differ": (
            decisions["holdout_2ms_shaped"]["selected_branch_factor"] == 8
            and decisions["holdout_40ms_shaped"]["selected_branch_factor"] == 32
        ),
        "every_profile_within_ten_percent": all(row["within_ten_percent"] for row in evaluations),
        "mean_regret_at_most_five_percent": statistics.fmean(regrets) <= 0.05,
        "all_fixed_controls_correct": controls_correct,
        "traces_preserve_hierarchy": traces_structural,
        "all_three_trials_per_cell_succeed": (
            len(measured) == expected_measured_count and not summary["failed_trials"]
        ),
        "decisions_sealed_before_measurement": decisions_precede_measurement,
        "calibration_hash_retained": bool(decision_record["calibration_sha256"]),
    }
    return {
        "schema_version": "1.0",
        "hypothesis_id": HYPOTHESIS_ID,
        "result": "PASS" if all(criteria.values()) else "FAIL",
        "criteria": criteria,
        "evaluations": evaluations,
        "mean_latency_regret_fraction": statistics.fmean(regrets),
        "maximum_latency_regret_fraction": max(regrets),
        "measured_trial_count": len(measured),
        "expected_measured_trial_count": expected_measured_count,
        "decision_record_sha256": _sha256_file(
            output_directory / "planner-decisions-prebenchmark.json"
        ),
    }


def run_h012_007(
    *,
    output_directory: Path,
    calibration_summary_path: Path,
    warmup_trials: int = 1,
    measured_trials: int = 3,
    payload_bytes: int = 256,
    operation_deadline_s: float = 30.0,
    startup_deadline_s: float = 180.0,
) -> dict[str, Any]:
    calibration = calibrate_from_summary(calibration_summary_path)
    calibration_path = output_directory / "planner-calibration.json"
    _write_json(calibration_path, calibration)
    decisions = choose_profiles(
        calibration,
        tuple(NETWORK_PROFILES[name] for name in HOLDOUT_PROFILES),
    )
    decision_record = {
        "schema_version": "1.0",
        "hypothesis_id": HYPOTHESIS_ID,
        "sealed_before_benchmark": True,
        "sealed_unix_ns": time.time_ns(),
        "calibration_sha256": _sha256_file(calibration_path),
        "decisions": decisions,
    }
    _write_json(output_directory / "planner-decisions-prebenchmark.json", decision_record)

    summary = run_h012_005(
        output_directory=output_directory,
        worker_counts=(128,),
        branch_factors=CANDIDATE_BRANCH_FACTORS,
        profiles=HOLDOUT_PROFILES,
        payload_bytes=payload_bytes,
        warmup_trials=warmup_trials,
        measured_trials=measured_trials,
        operation_deadline_s=operation_deadline_s,
        startup_deadline_s=startup_deadline_s,
        hypothesis_id=HYPOTHESIS_ID,
        additional_source_files=(
            Path(__file__).resolve(),
            Path(__file__).with_name("topology_planner.py"),
        ),
    )
    evaluation = evaluate_planner(
        output_directory=output_directory,
        summary=summary,
        decision_record=decision_record,
        measured_trials=measured_trials,
    )
    _write_json(output_directory / "planner-evaluation.json", evaluation)
    return {"benchmark": summary, "planner_evaluation": evaluation}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Experiment 012 H012-007")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-summary", type=Path, required=True)
    parser.add_argument("--warmup-trials", type=int, default=1)
    parser.add_argument("--measured-trials", type=int, default=3)
    parser.add_argument("--payload-bytes", type=int, default=256)
    parser.add_argument("--operation-deadline-s", type=float, default=30.0)
    parser.add_argument("--startup-deadline-s", type=float, default=180.0)
    args = parser.parse_args(argv)
    result = run_h012_007(
        output_directory=args.output.resolve(),
        calibration_summary_path=args.calibration_summary.resolve(),
        warmup_trials=args.warmup_trials,
        measured_trials=args.measured_trials,
        payload_bytes=args.payload_bytes,
        operation_deadline_s=args.operation_deadline_s,
        startup_deadline_s=args.startup_deadline_s,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["planner_evaluation"]["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["HOLDOUT_PROFILES", "HYPOTHESIS_ID", "evaluate_planner", "run_h012_007"]
