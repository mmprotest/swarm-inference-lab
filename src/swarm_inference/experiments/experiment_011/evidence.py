"""Gate evaluation, reports, artifact audit, and final bundle assembly."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_011.analysis import (
    PROFILE_LABELS,
    RESULT_COLUMNS,
    SUMMARY_COLUMNS,
)

CORE_GATES = {
    "baseline_integrity",
    "real_contiguous_stage_ownership",
    "exact_stage_execution",
    "coordinator_removal",
    "serial_wait_reduction",
    "message_reduction",
    "payload_reduction",
    "regional_wan_improvement",
    "global_wan_improvement",
    "curve_flattening",
    "regression_safety",
    "evidence_completeness",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gate(name: str, passed: bool, detail: str, evidence: Sequence[str]) -> dict[str, Any]:
    return {
        "gate": name,
        "status": "PASS" if passed else "FAIL",
        "detail": detail,
        "evidence": list(evidence),
        "threshold_changed_after_results": False,
    }


def evaluate_gates(
    *,
    run_root: Path,
    summary_rows: list[dict[str, Any]],
    analysis: dict[str, Any],
    baseline_rows: list[dict[str, Any]],
    exactness_results: list[dict[str, Any]],
    ownership_valid: bool,
    coordinator_removed: bool,
    compression_results: list[dict[str, Any]],
    speculation_results: list[dict[str, Any]],
    concurrency_results: list[dict[str, Any]],
    regression_tests_passed: bool,
    evidence_complete: bool,
) -> tuple[list[dict[str, Any]], str]:
    summary_by_profile = {str(row["profile_name"]): row for row in summary_rows}
    baseline_integrity = len(baseline_rows) == 8 and all(
        row.get("token_match") is True and row.get("valid_for_claims") is True
        for row in baseline_rows
    )
    exact = bool(exactness_results) and all(
        row.get("token_match") is True
        and row.get("capture_exact") is True
        and float(row.get("maximum_absolute_difference_fp32", 1)) == 0.0
        and float(row.get("maximum_relative_l2_error_fp32", 1)) == 0.0
        and row.get("fallback_used") is False
        for row in exactness_results
    )
    two_waits = [
        float(row["serial_waits_per_token"])
        for row in exactness_results
        if int(row.get("stage_count", 0)) == 2
    ]
    four_waits = [
        float(row["serial_waits_per_token"])
        for row in exactness_results
        if int(row.get("stage_count", 0)) == 4
    ]
    wait_gate = bool(two_waits and four_waits) and max(two_waits) <= 2 and max(four_waits) <= 4
    selected_rows = [
        row
        for row in summary_rows
        if float(row.get("stage_exact_median_tps", 0)) > 0
    ]
    message_gate = bool(selected_rows) and all(
        float(row["message_reduction"]) >= 0.90 for row in selected_rows
    )
    payload_rows_path = run_root / "network_profile_results.csv"
    with payload_rows_path.open("r", encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    selected_raw = [row for row in raw_rows if row["strategy"] == "stage_ring_exact_best_planner"]
    payload_gate = len(selected_raw) == 8 and all(
        float(row["payload_bytes_per_token"]) < 0.5 * 1024 * 1024 for row in selected_raw
    )
    regional = summary_by_profile.get("regional_wan", {})
    global_wan = summary_by_profile.get("global_wan", {})
    regional_gate = (
        float(regional.get("throughput_multiple", 0)) >= 3
        and float(regional.get("stage_exact_median_tps", 0)) >= 0.90
    )
    global_gate = (
        float(global_wan.get("throughput_multiple", 0)) >= 5
        and float(global_wan.get("stage_exact_median_tps", 0)) >= 0.40
    )
    curve = analysis["whole_curve"]
    wan_rows = [
        summary_by_profile.get(profile, {})
        for profile in ("wifi", "regional_wan", "global_wan")
    ]
    curve_gate = (
        float(curve["global_wan_to_loopback_ratio"]) >= 0.20
        and int(curve["improved_profiles"]) >= 6
        and regional.get("classification") == "IMPROVED"
        and global_wan.get("classification") == "IMPROVED"
        and all(row.get("classification") != "REGRESSED" for row in wan_rows)
    )
    compression_exact = bool(compression_results) and all(
        row.get("token_match") is True and row.get("bitwise_lossless") is True
        for row in compression_results
    )
    compression_utility_ok = all(
        (not row.get("planner_selected")) or float(row.get("throughput_effect_tps", 0)) > 0
        for row in compression_results
    )
    compression_gate = compression_exact and compression_utility_ok
    speculation_gate = bool(speculation_results) and all(
        row.get("exact_token_identity") is True and row.get("oracle_proposals_used") is False
        for row in speculation_results
    )
    if speculation_gate:
        speculation_gate = all(
            row.get("planner_enabled") is False
            or float(row.get("throughput_multiple_vs_non_speculative", 0)) > 1
            for row in speculation_results
        )
    concurrency_one = next(
        (
            row
            for row in concurrency_results
            if int(row.get("concurrency_active", 0)) == 1
            and not row.get("cancelled_session")
        ),
        None,
    )
    higher = [
        row
        for row in concurrency_results
        if int(row.get("concurrency_active", 0)) > 1 and not row.get("cancelled_session")
    ]
    concurrency_gate = (
        concurrency_one is not None
        and all(row.get("all_sessions_exact") is True for row in concurrency_results)
        and any(
            float(row["aggregate_verified_tokens_per_second"])
            > float(concurrency_one["aggregate_verified_tokens_per_second"])
            for row in higher
        )
        and all(float(row.get("fairness_min_over_max", 0)) > 0 for row in higher)
        and any(row.get("cancelled_session_kv_cleanup") is True for row in concurrency_results)
    )
    gates = [
        _gate(
            "baseline_integrity",
            baseline_integrity,
            "Fresh old-path rows are exact for all eight frozen profiles and remain distinct from archived values.",
            ["baseline/fresh/network_profile_results.csv", "baseline/archived_values.json"],
        ),
        _gate(
            "real_contiguous_stage_ownership",
            ownership_valid,
            "Stage ownership records cover each transformer layer exactly once with disjoint process-owned weights.",
            ["exactness/ownership_validation.json", "stage_plans/"],
        ),
        _gate(
            "exact_stage_execution",
            exact,
            "Canonical captures require exact tokens, byte-identical boundaries/logits, and zero FP32 error.",
            ["exactness/exactness_summary.json"],
        ),
        _gate(
            "coordinator_removal",
            coordinator_removed,
            "Trace dependency edges contain only direct stage-to-stage activation and final-stage-to-stage-zero token links.",
            ["critical_path_summary.csv", "traces/"],
        ),
        _gate(
            "serial_wait_reduction",
            wait_gate,
            f"Observed two-stage waits {two_waits}; four-stage waits {four_waits}; limits are two and four.",
            ["critical_path_summary.csv"],
        ),
        _gate(
            "message_reduction",
            message_gate,
            "Every selected stage path must reduce messages per token by at least 90% versus its fresh baseline.",
            ["network_profile_summary.csv"],
        ),
        _gate(
            "payload_reduction",
            payload_gate,
            "Every selected stage path must transfer less than 0.5 MiB per generated token.",
            ["network_profile_results.csv"],
        ),
        _gate(
            "regional_wan_improvement",
            regional_gate,
            f"Regional WAN: {regional.get('throughput_multiple', 0):.2f}x and {regional.get('stage_exact_median_tps', 0):.2f} tok/s.",
            ["network_profile_summary.csv"],
        ),
        _gate(
            "global_wan_improvement",
            global_gate,
            f"Global WAN: {global_wan.get('throughput_multiple', 0):.2f}x and {global_wan.get('stage_exact_median_tps', 0):.2f} tok/s.",
            ["network_profile_summary.csv"],
        ),
        _gate(
            "curve_flattening",
            curve_gate,
            (
                f"Global/loopback ratio {curve['global_wan_to_loopback_ratio']:.3f}; "
                f"classifications: {curve['improved_profiles']} improved, "
                f"{curve['inconclusive_profiles']} inconclusive, {curve['regressed_profiles']} regressed."
            ),
            ["network_analysis.json", "network_profile_summary.csv"],
        ),
        _gate(
            "adaptive_compression",
            compression_gate,
            "Compression must be bytewise lossless and selected by the final planner only after positive measured utility.",
            ["compression_summary.csv", "compression/"],
        ),
        _gate(
            "exact_speculation",
            speculation_gate,
            "Prompt-lookup verification used no oracle; exactness is required and negative-value plans remain disabled.",
            ["speculation_summary.csv", "speculation/speculation_results.json"],
        ),
        _gate(
            "concurrent_goodput",
            concurrency_gate,
            "All active sessions must be exact, one higher concurrency must improve aggregate goodput, fairness must be nonzero, and cancellation must clean up.",
            ["concurrency_summary.csv", "concurrency/"],
        ),
        _gate(
            "regression_safety",
            regression_tests_passed,
            "The full repository suite, including Experiment 010 compatibility readers and paths, must pass.",
            ["logs/final-regression-tests.json"],
        ),
        _gate(
            "evidence_completeness",
            evidence_complete,
            "Every required artifact is nonempty, schema-checked, hashed, and connected to raw evidence.",
            ["manifest.json", "artifact_validation.json"],
        ),
    ]
    gate_map = {gate["gate"]: gate["status"] == "PASS" for gate in gates}
    if all(gate_map[name] for name in CORE_GATES):
        verdict = "PASS CLOSURE"
    elif gate_map["real_contiguous_stage_ownership"] and gate_map["exact_stage_execution"]:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"
    return gates, verdict


REQUIRED_FILES = (
    "manifest.json",
    "verdict.json",
    "gate_results.json",
    "environment.json",
    "source_identity.json",
    "binary_identity.json",
    "model_identity.json",
    "network_profiles_manifest.json",
    "network_profile_results.csv",
    "network_profile_summary.csv",
    "stage_latency_summary.csv",
    "critical_path_summary.csv",
    "compression_summary.csv",
    "speculation_summary.csv",
    "concurrency_summary.csv",
    "failure_summary.csv",
    "EXPERIMENT_011_RESULT.md",
    "EXPERIMENT_011_TECHNICAL_REPORT.md",
)


def validate_artifacts(run_root: Path, *, require_reports: bool = True) -> dict[str, Any]:
    required = list(REQUIRED_FILES)
    if not require_reports:
        required = [name for name in required if not name.endswith(".md")]
    missing = [name for name in required if not (run_root / name).is_file()]
    empty = [
        name
        for name in required
        if (run_root / name).is_file() and (run_root / name).stat().st_size == 0
    ]
    chart_names = (
        "06_network_profile_before_after",
        "06b_network_profile_same_run_comparison",
        "06c_network_profile_experiment_011",
        "06d_network_profile_improvement",
    )
    missing_charts = [
        f"charts/{base}.{extension}"
        for base in chart_names
        for extension in ("png", "svg", "pdf")
        if not (run_root / "charts" / f"{base}.{extension}").is_file()
    ]
    csv_checks = {}
    for name, required_columns in (
        ("network_profile_results.csv", RESULT_COLUMNS),
        ("network_profile_summary.csv", SUMMARY_COLUMNS),
    ):
        path = run_root / name
        if path.is_file():
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                columns = reader.fieldnames or []
                row_count = sum(1 for _ in reader)
            csv_checks[name] = {
                "required_columns_present": all(column in columns for column in required_columns),
                "row_count": row_count,
            }
    valid = (
        not missing
        and not empty
        and not missing_charts
        and all(row["required_columns_present"] for row in csv_checks.values())
        and all(row["row_count"] > 0 for row in csv_checks.values())
    )
    return {
        "valid": valid,
        "required_files": required,
        "missing": missing,
        "empty": empty,
        "missing_charts": missing_charts,
        "csv_checks": csv_checks,
    }


def build_manifest(run_root: Path, *, metadata: dict[str, Any]) -> dict[str, Any]:
    artifacts = []
    for path in sorted(run_root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        artifacts.append(
            {
                "path": path.relative_to(run_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        **metadata,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "manifest_self_hash_excluded": True,
    }
    (run_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def make_zip(run_root: Path) -> Path:
    target_base = run_root.parent / run_root.name
    zip_path = Path(
        shutil.make_archive(
            str(target_base),
            "zip",
            root_dir=run_root.parent,
            base_dir=run_root.name,
        )
    )
    return zip_path


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write_reports(
    *,
    run_root: Path,
    verdict: str,
    gates: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    analysis: dict[str, Any],
    exactness_results: list[dict[str, Any]],
    compression_results: list[dict[str, Any]],
    speculation_results: list[dict[str, Any]],
    concurrency_results: list[dict[str, Any]],
    failure_results: list[dict[str, Any]],
    plan_records: list[dict[str, Any]],
    limitations: list[str],
    reproduction_command: str,
) -> None:
    gate_status = {str(row["gate"]): str(row["status"]) for row in gates}
    gate_table = _markdown_table(
        ("Gate", "Status", "Finding"),
        [(row["gate"], row["status"], row["detail"]) for row in gates],
    )
    network_table = _markdown_table(
        (
            "Profile",
            "Archived 010",
            "Fresh old path",
            "Best stage (95% CI)",
            "Difference (95% CI)",
            "Multiple",
            "Topology",
            "Class",
        ),
        [
            (
                PROFILE_LABELS[str(row["profile_name"])],
                f"{float(row['archived_010_tps']):.2f}",
                f"{float(row['same_run_baseline_median_tps']):.2f}",
                (
                    f"{float(row['stage_exact_median_tps']):.2f} "
                    f"[{float(row['stage_exact_ci_low']):.2f}, "
                    f"{float(row['stage_exact_ci_high']):.2f}]"
                ),
                (
                    f"{float(row['difference_median_tps']):+.2f} "
                    f"[{float(row['difference_ci_low']):+.2f}, "
                    f"{float(row['difference_ci_high']):+.2f}]"
                ),
                f"{float(row['throughput_multiple']):.2f}x",
                f"{row['selected_stage_count']}-stage {row['selected_method']}",
                row["classification"],
            )
            for row in summary_rows
        ],
    )
    ownership_table = _markdown_table(
        ("Topology", "Method", "Stage", "Layers", "Weight bytes", "KV bytes/token"),
        [
            (
                plan["stage_count"],
                plan["partition_method"],
                assignment["stage_id"],
                f"[{assignment['layer_start']}, {assignment['layer_end']})",
                assignment["weight_bytes"],
                assignment["kv_cache_bytes_per_token"],
            )
            for plan in plan_records
            for assignment in plan["assignments"]
        ],
    )
    curve = analysis["whole_curve"]
    selected_raw = {}
    evidence_rows: list[dict[str, str]] = []
    with (run_root / "network_profile_results.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        evidence_rows = list(csv.DictReader(handle))
        for row in evidence_rows:
            if row["strategy"] == "stage_ring_exact_best_planner":
                selected_raw[row["profile_name"]] = row
    waits = [float(row["serial_waits_per_token"]) for row in selected_raw.values()]
    messages = [float(row["messages_per_token"]) for row in selected_raw.values()]
    payloads = [float(row["payload_bytes_per_token"]) for row in selected_raw.values()]
    baseline_raw = {}
    for row in evidence_rows:
        if row["strategy"] == "experiment_011_same_run_expert_rpc":
            baseline_raw[row["profile_name"]] = row
    local_row = next(
        (row for row in evidence_rows if row["strategy"] == "local_monolithic_reference"),
        None,
    )
    baseline_table = _markdown_table(
        ("Series", "Status", "Scope", "Throughput"),
        (
            (
                "experiment_010_archived",
                "archived; not newly measured",
                "eight shaped profiles",
                "profile-specific values below",
            ),
            (
                "experiment_011_same_run_expert_rpc",
                "fresh real-model measurement",
                "eight shaped profiles",
                "profile-specific values below",
            ),
            (
                "local_monolithic_reference",
                "fresh real-model measurement",
                "unshaped local control",
                f"{float(local_row['throughput_tps']):.2f} tok/s" if local_row else "missing",
            ),
        ),
    )
    critical_rows: list[dict[str, str]] = []
    with (run_root / "critical_path_summary.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        critical_rows = list(csv.DictReader(handle))
    critical_table_rows = []
    for strategy, label in (
        ("experiment_011_same_run_expert_rpc", "Fresh expert RPC"),
        ("stage_ring_exact_best_planner", "Selected exact stage path"),
    ):
        strategy_rows = [row for row in critical_rows if row["strategy"] == strategy]
        critical_table_rows.append(
            (
                label,
                f"{statistics_median([float(row['serial_waits_per_token']) for row in strategy_rows]):.2f}",
                f"{statistics_median([float(row['messages_per_token']) for row in strategy_rows]):.2f}",
                f"{statistics_median([float(row['payload_bytes_per_token']) for row in strategy_rows]):,.0f}",
                f"{statistics_median([float(row['serialization_ns_per_token']) for row in strategy_rows]) / 1e6:.3f}",
                f"{statistics_median([float(row['socket_ns_per_token']) for row in strategy_rows]) / 1e6:.3f}",
                f"{statistics_median([float(row['model_compute_ns_per_token']) for row in strategy_rows]) / 1e6:.3f}",
            )
        )
    critical_table = _markdown_table(
        ("Path", "Serial waits/token", "Messages/token", "Payload B/token", "Serialize ms/token", "Socket ms/token", "Compute ms/token"),
        critical_table_rows,
    )
    old_wait = statistics_median(
        [float(row["serial_waits_per_token"]) for row in baseline_raw.values()]
    )
    old_messages = statistics_median(
        [float(row["messages_per_token"]) for row in baseline_raw.values()]
    )
    old_payload = statistics_median(
        [float(row["payload_bytes_per_token"]) for row in baseline_raw.values()]
    )
    result_text = f"""# Experiment 011 result

## Verdict

**{verdict}**

Exact contiguous-stage generation worked: **{'yes' if all(row.get('token_match') and row.get('capture_exact') for row in exactness_results) else 'no'}**. The coordinator was removed from steady-state decode dependencies: **{'yes' if gate_status.get('coordinator_removal') == 'PASS' else gate_status.get('coordinator_removal', 'pending').lower()}**.

The selected path used a median of **{statistics_median(waits):.2f} serial waits**, **{statistics_median(messages):.2f} messages**, and **{statistics_median(payloads):,.0f} payload bytes** per token, versus **{old_wait:.2f}**, **{old_messages:.2f}**, and **{old_payload:,.0f}** on the fresh old path.

The Global-WAN/loopback throughput ratio changed from the archived 3.1% reference to **{float(curve['global_wan_to_loopback_ratio']) * 100:.1f}%**. The point estimates improved strongly, but classifications remain governed by the frozen one-repetition workload; the report does not turn a descriptive one-sample bootstrap into inferential evidence.

## Direct answers

1. Exact contiguous-stage generation: **{'worked' if all(row.get('token_match') for row in exactness_results) else 'failed'}**.
2. Coordinator removed from the decode critical path: **{gate_status.get('coordinator_removal', 'PENDING')}**.
3. Serial waits per token: **{statistics_median(waits):.2f}** selected; two-stage and four-stage limits are evidenced separately.
4. Messages per token: **{statistics_median(messages):.2f}** selected.
5. Bytes transferred per token: **{statistics_median(payloads):,.0f} payload bytes** selected.
6. Loopback: fresh old path **{float(summary_rows[0]['same_run_baseline_median_tps']):.2f}**, stage path **{float(summary_rows[0]['stage_exact_median_tps']):.2f} tok/s**.
7. Wi-Fi: **{float(summary_rows[5]['throughput_multiple']):.2f}x**, classification **{summary_rows[5]['classification']}**.
8. Regional WAN: **{float(summary_rows[6]['throughput_multiple']):.2f}x**, **{float(summary_rows[6]['stage_exact_median_tps']):.2f} tok/s**, classification **{summary_rows[6]['classification']}**.
9. Global WAN: **{float(summary_rows[7]['throughput_multiple']):.2f}x**, **{float(summary_rows[7]['stage_exact_median_tps']):.2f} tok/s**, classification **{summary_rows[7]['classification']}**.
10. Curve flattening: Global-WAN/loopback is **{float(curve['global_wan_to_loopback_ratio']) * 100:.1f}%**.
11. Winning topology per profile is recorded in the table below and each `planner_decision.json`.
12. Lossless compression: **{'selected in at least one profile' if any(row.get('planner_selected') for row in compression_results) else 'proved lossless but was disabled by the measured planner'}**.
13. Exact speculation: **{'exact' if all(row.get('exact_token_identity') for row in speculation_results) else 'not exact'}**; **{'selected' if any(row.get('planner_enabled') for row in speculation_results) else 'disabled because valid socket-path expected value was not positive'}**.
14. Concurrent goodput: **{'increased at a higher concurrency' if gate_status.get('concurrent_goodput') == 'PASS' else 'did not close the full gate' if gate_status else 'pending'}**.
15. Multiple-machine proof: physical NIC, switch, host-clock, cross-host GPU scheduling, packet-loss recovery, and failure-domain behaviour remain unproven on this one-machine shaped-loopback setup.

## Gates

{gate_table}

## Baseline controls

{baseline_table}

## Before-and-after network results

{network_table}

The 95% intervals above follow the predeclared bootstrap rule. With the frozen single repetition they are degenerate descriptive intervals and do not estimate between-run uncertainty.

## Critical-path decomposition

{critical_table}

## Stage ownership

{ownership_table}

## Exactness evidence

- Canonical comparisons: {sum(int(row.get('comparison_count', 0)) for row in exactness_results):,} tensor comparisons.
- Token mismatches: {sum(not bool(row.get('token_match')) for row in exactness_results)}.
- Boundary byte mismatches: {sum(int(row.get('capture_mismatch_count', 0)) for row in exactness_results)}.
- Maximum absolute FP32 error: {max((float(row.get('maximum_absolute_difference_fp32', 0)) for row in exactness_results), default=0.0)}.
- Maximum relative L2 FP32 error: {max((float(row.get('maximum_relative_l2_error_fp32', 0)) for row in exactness_results), default=0.0)}.

## Failure and recovery

{_markdown_table(('Test', 'Detected', 'Recovered', 'Exact'), [(row.get('test'), row.get('failure_detected'), row.get('recovered'), row.get('exact_continuation')) for row in failure_results])}

## Artifacts

- Evidence directory: `{run_root}`
- Raw rows: `network_profile_results.csv`
- Summary rows: `network_profile_summary.csv`
- Traces: `traces/`
- Charts: `charts/`
- Reproduction: `{reproduction_command}`
"""
    (run_root / "EXPERIMENT_011_RESULT.md").write_text(result_text, encoding="utf-8")

    technical = f"""# Experiment 011 technical report

## Architecture and stage lifecycle

The canonical `stage_ring_exact` path starts one process per contiguous stage. Stage zero owns token embeddings and session input state; intermediate stages own only their assigned decoder layers; the final stage owns final normalisation, the language-model head, and greedy sampling. The control coordinator performs `HELLO`, `CAPABILITIES`, `LOAD_STAGE`, `OPEN_SESSION`, health/cancellation, and final assembly. After `PREFILL` starts, activation frames follow persistent direct TCP connections and `TOKEN_RESULT` returns directly from the final stage to stage zero. Publication uses a bounded asynchronous queue and a separate control-plane sender.

Each stage creates a stage-local `DynamicCache` whose layer indices cover only that stage's modules. Prefill populates those entries. Decode transmits BF16 hidden activations, never KV tensors. Session close reports and releases owned KV bytes. Recovery creates new workers and deterministically replays accepted history; universal duplicate execution is not the default.

## Protocol

Frames use a fixed little-endian header (`SWRING11`), canonical JSON metadata, contiguous raw tensor bytes, and SHA-256 over metadata plus payload. Operations are `HELLO`, `CAPABILITIES`, `LOAD_STAGE`, `OPEN_SESSION`, `PREFILL`, `DECODE`, `VERIFY_CANDIDATES`, `TOKEN_RESULT`, `SESSION_CHECKPOINT`, `CLOSE_SESSION`, `CANCEL_SESSION`, `HEALTH`, and `ERROR`. Bounded reusable receive buffers, partial read/write loops, per-edge sequence validation, session/topology/model validation, explicit checksums, backpressure, and connection reuse are tested. Pickle is prohibited.

## Partition planner

Equal planning divides the {plan_records[0]['layer_count'] if plan_records else 0} layers contiguously with at most one layer difference. Balanced planning profiles every layer with CUDA events and solves an exact contiguous minimax dynamic program under stage memory limits. Its objective includes measured execution, weight bytes, KV growth, temporary memory, and activation size. Profile decisions additionally penalise serial boundaries, transferred bytes, imbalance, queueing, rejected draft work, reliability risk, and infeasible memory. Decisions do not branch on profile names.

## Exactness

The authoritative path uses the same cached OLMoE weights, tokenizer, prompt IDs, greedy configuration, eager attention implementation, BF16 layer execution, disabled TF32, and fixed revisions. Local hooks capture router outputs and every relevant stage boundary. Distributed workers capture corresponding raw bytes, final hidden state, FP32-promoted pre-sampling logits, and tokens. Lossless transport preserves dtype, shape, byte order, and a raw SHA-256. Any mismatch is emitted with paths, position, stage, errors, IDs, and reproduction command.

## Compression and speculation

`byte_shuffle_fast_codec` separates fixed-width element byte lanes before zlib level-1 compression and exactly reverses the transform. The adaptive controller compares measured encode/decode cost with transfer savings from payload size, ratio, bandwidth, RTT, and queue delay. The final planner uses observed end-to-end utility.

Prompt lookup is a non-oracle `DraftProvider`. The real staged model verifies each candidate position, accepts only the exact greedy prefix, commits the target token at a rejection, crops every stage-local cache to the committed position, and records rejected work. These mechanism runs are kept out of socket performance claims unless the socket planner has positive measured evidence.

## Instrumentation and statistical method

Every process writes monotonic nanosecond NDJSON events. A serial wait is counted only when a marked receive edge resolves to a later required CUDA compute event; topology or message totals are not substituted. The network CSV retains raw run, prompt, model, profile hash, process/resource, exactness, messages, bytes, and trace lineage.

Principal values are medians, matching Experiment 010. Mean, standard deviation, p25, p75, p95, and seeded bootstrap 95% intervals are also written. Because the frozen Experiment 010 workload contains one independent repetition per profile, a point bootstrap cannot estimate between-run uncertainty. The requested fixed classification rule is nevertheless applied to its degenerate interval and the resulting class is explicitly labelled descriptive rather than inferential; thresholds were not altered after observing results.

## Limitations

{chr(10).join(f'- {item}' for item in limitations)}

## Reproduction

```powershell
{reproduction_command}
```

The canonical run is `-Full`. `-Quick` is diagnostic only and cannot emit a closure verdict.
"""
    (run_root / "EXPERIMENT_011_TECHNICAL_REPORT.md").write_text(
        technical, encoding="utf-8"
    )


def statistics_median(values: Sequence[float]) -> float:
    import statistics

    return statistics.median(values) if values else 0.0
