"""Independently audit Experiment 017 calculations, evidence, and deliverables."""

# The audit verifies deliberately typeset multiplication signs in the report.
# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from PIL import Image

BLOCKS = [1, 2, 4, 7, 12, 16]


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _close(left: float, right: float, *, tolerance: float = 1e-9) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(self, name: str, condition: bool, detail: str) -> None:
        self.checks.append(
            {"name": name, "status": "PASS" if condition else "FAIL", "detail": detail}
        )

    @property
    def passed(self) -> bool:
        return all(check["status"] == "PASS" for check in self.checks)


def audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    artifact = root / "artifacts" / "experiment-017"
    results = artifact / "results"
    report_path = root / "docs" / "experiments" / "EXPERIMENT_017_REPORT.md"
    summary = _json(artifact / "summary.json")
    tracker = _json(artifact / "target-tracker.json")
    tests = _json(artifact / "test-results.json")
    factor = _json(artifact / "physical" / "factor-reference.json")
    chart_validation = _json(artifact / "charts" / "validation.json")
    exact = _csv(results / "exact-combined.csv")
    short = _csv(results / "kda-short-window.csv")
    precision = _csv(results / "precision-matrix.csv")
    economics = _csv(results / "economics.csv")
    report = report_path.read_text(encoding="utf-8")
    checks = Audit()

    required = [
        "summary.json",
        "target-tracker.json",
        "environment.json",
        "model-metadata.json",
        "run-seeds.json",
        "source-manifest.json",
        "commands.txt",
        "test-results.json",
        "failure-log.json",
        "results/baseline.csv",
        "results/kda-short-window.csv",
        "results/state-traffic.csv",
        "results/factorized-verification.csv",
        "results/replay-state.csv",
        "results/expert-backends.csv",
        "results/exact-combined.csv",
        "results/precision-matrix.csv",
        "results/oracle-by-block.csv",
        "results/economics.csv",
        "physical/real-kda-layer-results.json",
        "physical/gpu-samples.csv",
        "cuda/manifest.json",
        *[
            f"charts/chart-{index:02d}-{name}.png"
            for index, name in enumerate(
                [
                    "oracle-progress",
                    "kda-budget-to-5",
                    "kda-layer-decomposition",
                    "state-traffic",
                    "expert-service",
                    "speed-quality-pareto",
                    "block-size-oracle",
                ],
                start=1,
            )
        ],
    ]
    missing = [name for name in required if not (artifact / name).is_file()]
    checks.check("required_artifacts", not missing, f"missing={missing}")
    checks.check("report_exists", report_path.is_file(), str(report_path))

    expected_constants = {
        "baseline_exp015_tok_s": 2.1838203221,
        "baseline_exp016_tok_s": 2.6691,
        "goal_tok_s": 5.0,
        "strong_goal_tok_s": 5.3382,
        "baseline_exp016_ms_per_accepted": 374.65,
        "goal_ms_per_accepted": 200.0,
        "baseline_exp016_block7_ms": 2997.24,
        "goal_block7_ms": 1600.0,
        "baseline_exp016_kda_ms": 1559.4,
        "baseline_exp016_non_kda_ms": 1437.84,
    }
    checks.check(
        "immutable_tracker_constants",
        all(_close(float(tracker[key]), value) for key, value in expected_constants.items()),
        "all mandated constants match",
    )
    budget = tracker["goal_block7_ms"] - tracker["baseline_exp016_non_kda_ms"]
    checks.check(
        "initial_kda_budget",
        _close(budget, tracker["initial_required_kda_ms_if_non_kda_frozen"])
        and _close(
            tracker["baseline_exp016_kda_ms"] / budget,
            tracker["initial_required_additional_kda_speedup"],
        ),
        f"budget={budget:.2f}, speedup={tracker['initial_required_additional_kda_speedup']:.6f}",
    )

    exact_blocks = [int(row["block_size"]) for row in exact]
    checks.check("exact_block_sweep", exact_blocks == BLOCKS, str(exact_blocks))
    checks.check("short_window_pair_count", len(short) == 12, f"rows={len(short)}")
    tracker_rows = tracker["results"]
    calculations_ok = len(tracker_rows) == 6
    for row in tracker_rows:
        accepted = int(row["accepted_tokens"])
        target_ms = float(row["target_pass_ms"])
        kda_ms = float(row["measured_kda_ms"])
        non_kda_ms = float(row["measured_non_kda_ms"])
        oracle = accepted * 1000.0 / target_ms
        calculations_ok &= _close(oracle, float(row["oracle_tok_s_per_user"]))
        calculations_ok &= _close(target_ms / accepted, float(row["ms_per_accepted_token"]))
        calculations_ok &= _close(kda_ms + non_kda_ms, target_ms)
        calculations_ok &= _close(
            accepted * 200.0 - non_kda_ms,
            float(row["required_kda_ms_if_non_kda_frozen"]),
        )
        calculations_ok &= _close(
            accepted * 1000.0 / non_kda_ms,
            float(row["free_kda_oracle"]),
        )
    checks.check("tracker_recalculation", calculations_ok, "six rows recomputed from raw fields")

    best = max(exact, key=lambda row: float(row["oracle_tok_s_per_user"]))
    best_oracle = float(best["oracle_tok_s_per_user"])
    best_speedup = best_oracle / tracker["baseline_exp016_tok_s"]
    expected_outcome = (
        "PASS_STRONG"
        if best_oracle >= 5.3382 and best_speedup >= 2.0
        else "PASS"
        if best_oracle >= 5.0 and best_speedup >= 1.8733
        else "STRONG_PARTIAL"
        if best_oracle >= 4.0037 and best_speedup >= 1.5
        else "WEAK"
        if best_oracle >= 3.3364 and best_speedup >= 1.25
        else "FAIL"
    )
    primary = summary["primary_result"]
    checks.check(
        "primary_result_selection",
        int(primary["block_size"]) == int(best["block_size"])
        and _close(float(primary["oracle_tok_s_per_user"]), best_oracle)
        and primary["outcome"] == expected_outcome,
        f"block={best['block_size']}, oracle={best_oracle:.10f}, expected={expected_outcome}",
    )
    checks.check(
        "baseline_reproduction_gate",
        summary["baseline_reproduction"]["pass"]
        and abs(float(summary["baseline_reproduction"]["target_pass_deviation_percent"])) <= 3.0,
        f"deviation={summary['baseline_reproduction']['target_pass_deviation_percent']:.6f}%",
    )

    factor_ok = factor["status"] == "PASS" and len(factor["rows"]) == 6
    for row in factor["rows"]:
        factor_ok &= int(row["block_tokens"]) in BLOCKS
        factor_ok &= float(row["output_metrics"]["relative_l2_error"]) <= 2e-5
        factor_ok &= float(row["state_metrics"]["relative_l2_error"]) <= 2e-5
        factor_ok &= float(row["replay_metrics"]["relative_l2_error"]) <= 2e-5
        factor_ok &= int(row["output_metrics"]["nan_count"]) == 0
        factor_ok &= int(row["state_metrics"]["inf_count"]) == 0
    checks.check("factor_and_replay_correctness", factor_ok, "all blocks <= 2e-5; no NaN/Inf")
    checks.check(
        "physical_exact_controls",
        bool(summary["correctness"]["short_window_bit_identity"])
        and bool(summary["correctness"]["expert_controls_pass"]),
        "window and expert controls exact",
    )
    bf16 = next(row for row in precision if row["mode"] == "bf16-state-fp32-update")
    checks.check(
        "approximate_separation",
        float(bf16["relative_l2_error"]) > 0.003
        and bf16["oracle_tok_s_per_user"] == ""
        and summary["approximate_result"]["outcome"] == "APPROX_FAIL",
        f"BF16 rel-L2={bf16['relative_l2_error']}; throughput absent",
    )

    economics_ok = len(economics) == 3
    for row in economics:
        aggregate = float(row["tok_s_per_user"]) * float(row["retained_user_slots"])
        gpu_hours = float(row["gpu_equivalent_count"]) * 1_000_000 / (aggregate * 3600)
        economics_ok &= _close(aggregate, float(row["aggregate_tok_s"]), tolerance=1e-8)
        economics_ok &= _close(
            gpu_hours,
            float(row["gpu_hours_per_1m_output_tokens"]),
            tolerance=1e-8,
        )
        economics_ok &= _close(
            gpu_hours * float(row["gpu_hourly_price_usd_inherited"]),
            float(row["projected_infrastructure_usd_per_1m_output_tokens"]),
            tolerance=1e-8,
        )
    checks.check(
        "economic_recalculation", economics_ok, "aggregate, GPU-hours, and cost recomputed"
    )

    checks.check(
        "tests",
        tests["status"] == "PASS"
        and tests["targeted"]["passed"] == tests["targeted"]["tests"]
        and tests["targeted"]["passed"] >= 53
        and tests["targeted"]["failures"] == 0
        and tests["full_repository"]["passed"]
        + tests["full_repository"]["skipped"]
        == tests["full_repository"]["tests"]
        and tests["full_repository"]["passed"] >= 1183
        and tests["full_repository"]["skipped"] == 13
        and tests["full_repository"]["failures"] == 0,
        (
            f"targeted {tests['targeted']['passed']} passed; full "
            f"{tests['full_repository']['passed']} passed, "
            f"{tests['full_repository']['skipped']} skipped, 0 failed"
        ),
    )

    manifest = _json(artifact / "source-manifest.json")
    hash_failures: list[str] = []
    for entry in manifest["files"]:
        path = Path(entry["path"])
        resolved = path if path.is_absolute() else root / path
        if not resolved.is_file() or _sha256(resolved) != entry["sha256"]:
            hash_failures.append(entry["path"])
    checks.check("source_manifest_hashes", not hash_failures, f"mismatches={hash_failures}")

    chart_ok = len(chart_validation["charts"]) == 7
    for chart in chart_validation["charts"]:
        image_path = artifact / "charts" / chart["file"]
        with Image.open(image_path) as image:
            chart_ok &= image.width >= 1200 and image.height >= 700
        chart_ok &= chart["status"] == "PASS"
    checks.check("chart_validation", chart_ok, "seven nonempty, high-resolution charts")

    headings_ok = all(f"## {index}." in report for index in range(1, 21))
    required_phrases = [
        "162.16 ms",
        "9.62×",
        "3.0994 tok/s/user",
        "1183 passed",
        "NO, KDA OPTIMIZATION PATH FALSIFIED",
    ]
    checks.check(
        "report_structure_and_claims",
        headings_ok and all(phrase in report for phrase in required_phrases),
        "20 numbered sections and decisive values present",
    )

    receipt = {
        "schema_version": "experiment-017-independent-audit-v1",
        "status": "PASS" if checks.passed else "FAIL",
        "check_count": len(checks.checks),
        "checks": checks.checks,
    }
    output = artifact / "validation.json"
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    receipt = audit(arguments.root)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
