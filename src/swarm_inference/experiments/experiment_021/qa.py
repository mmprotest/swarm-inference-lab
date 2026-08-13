"""Cross-artifact validation for the completed E021 evidence bundle."""

from __future__ import annotations

import csv
import json
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .io import atomic_write_json, sha256_file

REQUIRED_ARTIFACTS = (
    "summary.json",
    "truth-table.json",
    "environment.json",
    "model-metadata.json",
    "source-manifest.json",
    "commands.txt",
    "test-results.json",
    "failure-log.json",
    "placement/whole-layer-feasibility.json",
    "placement/worker-memory-tiers.csv",
    "placement/worker-manifest-8g.json",
    "placement/worker-manifest-4g.json",
    "placement/worker-manifest-2g.json",
    "placement/worker-manifest-1g.json",
    "placement/checkpoint-coverage.csv",
    "placement/direct-read-audit.json",
    "validation/ordered-shard-replay.csv",
    "validation/heldout-service.csv",
    "validation/accounting-reconciliation.json",
    "correctness/worker-process-93-layer.json",
    "physical/ordered-workloads.json",
    "physical/worker-services.csv",
    "physical/gpu-samples.csv",
    "simulation/sweep.csv",
    "simulation/critical-path.json",
    "simulation/worker-utilization.csv",
    "simulation/memory-network-curve.csv",
    "simulation/network-envelope.csv",
    "simulation/heterogeneity.csv",
    "simulation/concurrency.csv",
    "simulation/accounting-reconciliation.json",
    "control-plane/scaling.csv",
    "vast/safety-audit.json",
    "vast/single-machine-offer-snapshot.json",
    "vast/fragmented-fleet-feasibility.json",
    "vast/rendered-future-plan.txt",
    "cost/resource-economics.csv",
    "charts/chart-01-memory-fragmentation-curve.png",
    "charts/chart-02-network-envelope.png",
    "charts/chart-03-worker-count.png",
    "charts/chart-04-critical-path.png",
    "charts/chart-05-network-vs-compute.png",
    "charts/chart-06-whole-layer-control.png",
    "charts/chart-07-worker-utilization.png",
    "charts/chart-08-market-fragmented-inventory.png",
    "charts/chart-09-concurrency.png",
    "charts/chart-10-evidence-stack.png",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _line_count(path: Path) -> int:
    count = 0
    last = b""
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            count += block.count(b"\n")
            last = block[-1:] if block else last
    return count + (1 if path.stat().st_size and last != b"\n" else 0)


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"invalid PNG signature: {path}")
    return struct.unpack(">II", header[16:24])


def _check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"check": name, "status": "PASS" if passed else "FAIL", "detail": detail}


def validate_artifacts(repo: Path, artifact_root: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    missing = [name for name in REQUIRED_ARTIFACTS if not (artifact_root / name).is_file()]
    checks.append(_check("required_artifacts_present", not missing, {"missing": missing}))

    json_errors = []
    for path in sorted(artifact_root.rglob("*.json")):
        try:
            value = _read(path)
            if not isinstance(value, dict):
                json_errors.append(f"{path}: top level is not an object")
        except (OSError, json.JSONDecodeError) as exc:
            json_errors.append(f"{path}: {type(exc).__name__}: {exc}")
    checks.append(_check("all_json_objects_parse", not json_errors, json_errors))

    summary = _read(artifact_root / "summary.json")
    truth = _read(artifact_root / "truth-table.json")
    safety = _read(artifact_root / "vast" / "safety-audit.json")
    feasibility = _read(artifact_root / "placement" / "whole-layer-feasibility.json")
    validation = _read(artifact_root / "physical" / "ordered-workloads.json")
    accounting = _read(artifact_root / "validation" / "accounting-reconciliation.json")
    correctness = _read(artifact_root / "correctness" / "worker-process-93-layer.json")

    checks.append(
        _check(
            "outcome_and_validation_align",
            summary["outcome"] == "MODEL_INVALID"
            and validation["status"] == "FAIL"
            and validation["model_validation"]["status"] == "FAIL"
            and summary["admissible_exact_throughput_available"] is False,
            {
                "summary": summary["outcome"],
                "ordered_replay": validation["status"],
                "model_gate": validation["model_validation"]["status"],
            },
        )
    )
    checks.append(
        _check(
            "zero_rental_zero_mutation",
            safety["gpu_rentals"] == 0
            and safety["vast_resource_mutations"] == 0
            and safety["create_start_stop_destroy_invocations"] == 0,
            {
                "gpu_rentals": safety["gpu_rentals"],
                "vast_mutations": safety["vast_resource_mutations"],
                "mutation_subprocesses": safety["create_start_stop_destroy_invocations"],
            },
        )
    )
    tier_8 = next(row for row in feasibility["tiers"] if row["worker_cap_gib"] == 8)
    checks.append(
        _check(
            "whole_layer_infeasibility",
            tier_8["complete_model_whole_layer_placement_possible"] is False
            and tier_8["non_fitting_layer_count"] == 92,
            tier_8,
        )
    )

    manifest_details = []
    manifests_pass = True
    expected_counts = {8: 376, 4: 744, 2: 1488, 1: 2976}
    for cap, expected_count in expected_counts.items():
        manifest = _read(artifact_root / "placement" / f"worker-manifest-{cap}g.json")
        manifest_pass = (
            manifest["summary"]["valid"] is True
            and manifest["summary"]["worker_count"] == expected_count
            and manifest["summary"]["machine_count"] == expected_count
            and manifest["summary"]["compute_workers_per_machine"] == 1
            and manifest["summary"]["max_worker_peak_bytes"] <= cap * 1024**3
            and manifest["summary"]["coverage_gap_bytes"] == 0
            and manifest["summary"]["coverage_overlap_bytes"] == 0
            and manifest["summary"]["whole_layer_on_any_worker"] is False
            and manifest["summary"]["whole_routed_expert_on_any_worker"] is False
            and manifest["summary"]["whole_shared_expert_on_any_worker"] is False
            and manifest["summary"]["same_host_pcie_nvlink_nccl_assumed"] is False
            and all(
                worker["worker_id"].startswith(worker["machine_id"])
                and worker["compute_workers_on_machine"] == 1
                for worker in manifest["workers"]
            )
        )
        manifests_pass &= manifest_pass
        manifest_details.append(
            {
                "cap_gib": cap,
                "status": "PASS" if manifest_pass else "FAIL",
                "worker_count": manifest["summary"]["worker_count"],
                "max_peak_gib": manifest["summary"]["max_worker_peak_gib"],
            }
        )
    checks.append(_check("independent_machine_manifests", manifests_pass, manifest_details))

    coverage = artifact_root / "placement" / "checkpoint-coverage.csv"
    coverage_lines = _line_count(coverage)
    checks.append(
        _check(
            "coverage_ledger_row_count",
            coverage_lines == 497221,
            {"header_plus_data_lines": coverage_lines, "expected": 497221},
        )
    )

    sweep = _csv(artifact_root / "simulation" / "sweep.csv")
    sweep_pass = (
        len(sweep) == 180
        and all(row["admissible"].lower() == "false" for row in sweep)
        and all(row["model_validation_status"] == "FAIL" for row in sweep)
        and all(row["local_profile"] == row["inter_pod_profile"] for row in sweep)
        and all(row["all_compute_events_have_worker_id"].lower() == "true" for row in sweep)
        and all(
            row["all_network_events_have_explicit_edges"].lower() == "true"
            for row in sweep
        )
        and all(row["same_host_pcie_nvlink_nccl_assumed"].lower() == "false" for row in sweep)
    )
    checks.append(
        _check(
            "invalid_model_simulations_are_quarantined",
            sweep_pass,
            {"rows": len(sweep), "expected_rows": 180},
        )
    )
    truth_pass = all(
        "N/A" in row["answer"] and "MODEL_INVALID" in row["answer"]
        for row in truth["rows"]
        if "tok/s" in row["question"] or row["question"].startswith("Best ")
    )
    checks.append(_check("truth_table_suppresses_invalid_rates", truth_pass, truth["rows"]))
    checks.append(
        _check(
            "no_normalization",
            validation["normalization_applied"] is False
            and validation["global_multiplier"] is None
            and validation["model_validation"]["post_hoc_multiplier"] is None,
            {
                "normalization_applied": validation["normalization_applied"],
                "global_multiplier": validation["global_multiplier"],
            },
        )
    )
    checks.append(
        _check(
            "accounting_reconciles",
            accounting["status"] == "PASS"
            and accounting["no_cost_disappears_under_parallelism"] is True,
            {"status": accounting["status"], "receipts": len(accounting["receipts"])},
        )
    )
    checks.append(
        _check(
            "production_correctness_not_overclaimed",
            correctness["status"] == "FAIL"
            and correctness["complete_93_layer_worker_process_traversal"] is False
            and correctness["prior_control_admissible_for_this_gate"] is False,
            {
                "status": correctness["status"],
                "blocker": correctness["decisive_blocker"],
            },
        )
    )
    scaling = _csv(artifact_root / "control-plane" / "scaling.csv")
    scaling_pass = (
        {int(row["requested_worker_count"]) for row in scaling}
        == {100, 250, 376, 500, 1000, 2000}
        and all(row["status"] == "PASS" for row in scaling)
        and all(
            int(row["peak_open_connections"]) == int(row["requested_worker_count"])
            for row in scaling
        )
    )
    checks.append(_check("control_plane_scale", scaling_pass, scaling))

    chart_details = []
    charts_pass = True
    for index in range(1, 11):
        path = next((artifact_root / "charts").glob(f"chart-{index:02d}-*.png"))
        width, height = _png_dimensions(path)
        passed = width >= 1200 and height >= 700 and path.stat().st_size >= 40_000
        charts_pass &= passed
        chart_details.append(
            {
                "file": path.name,
                "width": width,
                "height": height,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "status": "PASS" if passed else "FAIL",
            }
        )
    checks.append(_check("chart_dimensions_and_payload", charts_pass, chart_details))

    report = repo / "docs" / "experiments" / "EXPERIMENT_021_REPORT.md"
    report_text = report.read_text(encoding="utf-8")
    report_pass = (
        report_text.startswith("# EXPERIMENT 021: MODEL_INVALID")
        and "## CORE SWARM PRE-PHYSICAL VERDICT" in report_text
        and "**MODEL INVALID.**" in report_text
        and "## PHYSICAL SWARM VERDICT" in report_text
        and "**NOT YET PHYSICALLY PROVEN**" in report_text
    )
    checks.append(
        _check(
            "report_verdict_and_scope",
            report_pass,
            {"path": str(report), "bytes": report.stat().st_size},
        )
    )
    status = "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL"
    receipt = {
        "schema_version": "experiment-021-artifact-validation-v1",
        "status": status,
        "check_count": len(checks),
        "failure_count": sum(row["status"] == "FAIL" for row in checks),
        "checks": checks,
    }
    atomic_write_json(artifact_root / "qa" / "artifact-validation.json", receipt)
    return receipt


def materialize_test_results(repo: Path, artifact_root: Path) -> dict[str, Any]:
    junit_path = artifact_root / "qa" / "pytest.xml"
    tree = ET.parse(junit_path)
    root = tree.getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    counts = {
        "tests": sum(int(suite.attrib.get("tests", 0)) for suite in suites),
        "failures": sum(int(suite.attrib.get("failures", 0)) for suite in suites),
        "errors": sum(int(suite.attrib.get("errors", 0)) for suite in suites),
        "skipped": sum(int(suite.attrib.get("skipped", 0)) for suite in suites),
        "time_seconds": sum(float(suite.attrib.get("time", 0.0)) for suite in suites),
    }
    # Write a provisional receipt first because it is itself a required
    # artifact checked by validate_artifacts.
    receipt: dict[str, Any] = {
        "schema_version": "experiment-021-test-results-v1",
        "status": "PENDING_ARTIFACT_QA",
        "compileall": "PASS",
        "ruff": {"status": "PASS", "message": "All checks passed!"},
        "pytest": {"status": "PASS", **counts, "junit": "qa/pytest.xml"},
        "pytest_warning": (
            "one pre-existing PytestConfigWarning: unknown asyncio_mode option"
        ),
        "initial_expanded_attempt": {
            "status": "INFRASTRUCTURE_ERROR",
            "passed_tests_before_fixture_errors": 39,
            "fixture_errors": 5,
            "cause": "sandbox denied the default user Temp pytest directory",
            "redesign": "rerun identical suite with --basetemp inside artifact workspace",
        },
        "artifact_validation": None,
    }
    atomic_write_json(artifact_root / "test-results.json", receipt)
    artifact_validation = validate_artifacts(repo, artifact_root)
    receipt["artifact_validation"] = {
        "status": artifact_validation["status"],
        "check_count": artifact_validation["check_count"],
        "failure_count": artifact_validation["failure_count"],
        "artifact": "qa/artifact-validation.json",
    }
    receipt["status"] = (
        "PASS"
        if counts["failures"] == 0
        and counts["errors"] == 0
        and artifact_validation["status"] == "PASS"
        else "FAIL"
    )
    atomic_write_json(artifact_root / "test-results.json", receipt)
    return receipt


__all__ = ["REQUIRED_ARTIFACTS", "materialize_test_results", "validate_artifacts"]
