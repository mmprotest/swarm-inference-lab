"""Trace reconstruction and metric inspection for Experiment 012 cycles."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_012.analysis import fit_scaling
from swarm_inference.experiments.experiment_012.baseline_harness import (
    _percentile,
    _write_json,
)

FIT_METRICS = (
    "root_rpc_count",
    "root_messages_total",
    "root_bytes_total",
    "root_serial_waits",
    "root_direct_degree",
    "root_leaf_rpc_count",
    "root_cpu_ns",
    "root_coordinator_waits",
    "critical_path_sync_points",
    "hierarchy_depth",
    "total_messages",
    "total_bytes",
    "end_to_end_latency_ms",
    "throughput_ops_s",
)

SUMMARY_METRICS = (
    *FIT_METRICS,
    "worker_to_worker_rpc_count",
    "leaf_rpc_count",
    "worker_serial_waits",
    "worker_barrier_waits",
    "parallel_dispatch_nodes",
    "leaf_latency_p50_ms",
    "leaf_latency_p95_ms",
    "leaf_latency_p99_ms",
    "edge_latency_p50_ms",
    "edge_latency_p95_ms",
    "edge_latency_p99_ms",
    "total_connection_count",
    "root_connection_count",
    "total_connection_ns",
    "total_worker_cpu_ns",
    "intermediate_reductions",
    "maximum_queue_depth",
    "retries",
    "failures",
    "stragglers",
    "duplicated_work",
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _eligible(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if not row.get("warmup") and row.get("status") == "ok" and row.get("correctness")
    ]


def metric_summary(rows: list[dict[str, Any]], branch_factor: int) -> dict[str, Any]:
    eligible = _eligible(rows)
    groups: list[dict[str, Any]] = []
    fits: dict[str, Any] = {}
    profiles = sorted({str(row["network_profile"]) for row in eligible})
    modes = sorted({str(row["mode"]) for row in eligible})
    for profile in profiles:
        fits[profile] = {}
        for mode in modes:
            selected_mode = [
                row for row in eligible if row["network_profile"] == profile and row["mode"] == mode
            ]
            if not selected_mode:
                continue
            fits[profile][mode] = {}
            worker_counts = sorted({int(row["worker_count"]) for row in selected_mode})
            for worker_count in worker_counts:
                selected = [
                    row for row in selected_mode if int(row["worker_count"]) == worker_count
                ]
                values: dict[str, Any] = {}
                for metric in SUMMARY_METRICS:
                    observed = [float(row[metric]) for row in selected if metric in row]
                    if observed:
                        values[metric] = {
                            "median": statistics.median(observed),
                            "p50": _percentile(observed, 50),
                            "p95": _percentile(observed, 95),
                            "p99": _percentile(observed, 99),
                            "min": min(observed),
                            "max": max(observed),
                            "raw": observed,
                        }
                groups.append(
                    {
                        "network_profile": profile,
                        "mode": mode,
                        "worker_count": worker_count,
                        "successful_trials": len(selected),
                        "metrics": values,
                    }
                )
            for metric in FIT_METRICS:
                observations: list[tuple[int, float]] = []
                for worker_count in worker_counts:
                    observed = [
                        float(row[metric])
                        for row in selected_mode
                        if int(row["worker_count"]) == worker_count and metric in row
                    ]
                    if observed:
                        observations.append((worker_count, statistics.median(observed)))
                if len(observations) >= 3:
                    fits[profile][mode][metric] = fit_scaling(
                        observations, branch_factor=branch_factor
                    )
    return {
        "eligible_trial_count": len(eligible),
        "excluded_trial_count": len(rows) - len(eligible),
        "groups": groups,
        "fits": fits,
    }


def trace_audit(cycle_directory: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    root_events = _jsonl(cycle_directory / "traces" / "root.jsonl")
    worker_events: list[dict[str, Any]] = []
    paths = sorted((cycle_directory / "raw").glob("workers-*/processes/*/trace.jsonl"))
    for path in paths:
        worker_events.extend(_jsonl(path))
    root_by_operation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    child_by_operation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reductions_by_operation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in root_events:
        if event.get("event") == "root_delegated_round_trip":
            root_by_operation[str(event.get("operation_id"))].append(event)
    for event in worker_events:
        operation_id = str(event.get("operation_id"))
        if event.get("event") == "worker_child_round_trip":
            child_by_operation[operation_id].append(event)
        elif event.get("event") == "delegated_subtree_reduced":
            reductions_by_operation[operation_id].append(event)

    operations: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    delegated_rows = [row for row in rows if str(row.get("mode", "")).startswith("delegated")]
    for row in delegated_rows:
        operation_id = str(row["operation_id"])
        root_edges = root_by_operation[operation_id]
        child_edges = child_by_operation[operation_id]
        reductions = reductions_by_operation[operation_id]
        checks = {
            "root_edge_count_matches": len(root_edges) == int(row["root_rpc_count"]),
            "worker_edge_count_matches": len(child_edges) == int(row["worker_to_worker_rpc_count"]),
            "reduction_count_matches_workers": len(reductions) == int(row["worker_count"]),
            "no_root_to_leaf_edge": all(
                not bool(event.get("receiver_is_leaf")) for event in root_edges
            ),
            "all_root_edges_owned_by_stage_owner": all(
                event.get("sender_worker_id") == "stage-owner" for event in root_edges
            ),
            "all_child_edges_owned_by_workers": all(
                event.get("sender_worker_id") != "stage-owner" for event in child_edges
            ),
            "all_child_edges_cross_processes": all(
                event.get("sender_process_id") != event.get("receiver_process_id")
                for event in child_edges
            ),
            "exact_digest": row.get("expected_digest") == row.get("actual_digest"),
            "exact_contribution_count": (
                int(row["actual"]["contribution_count"]) == int(row["worker_count"])
                if row.get("actual")
                else False
            ),
        }
        failed = sorted(name for name, passed in checks.items() if not passed)
        if failed:
            issues.append({"operation_id": operation_id, "failed_checks": failed})
        operations.append(
            {
                "operation_id": operation_id,
                "warmup": bool(row.get("warmup")),
                "status": row.get("status"),
                "worker_count": row["worker_count"],
                "network_profile": row["network_profile"],
                "root_edges": len(root_edges),
                "worker_edges": len(child_edges),
                "reductions": len(reductions),
                "checks": checks,
            }
        )
    return {
        "delegated_operation_count": len(delegated_rows),
        "root_delegated_edge_count": sum(len(value) for value in root_by_operation.values()),
        "worker_child_edge_count": sum(len(value) for value in child_by_operation.values()),
        "hierarchical_reduction_event_count": sum(
            len(value) for value in reductions_by_operation.values()
        ),
        "process_identity_count": len(
            {
                int(event["process_id"])
                for event in worker_events
                if event.get("event") == "worker_ready"
            }
        ),
        "all_checks_pass": not issues,
        "issues": issues,
        "operations": operations,
    }


def inspect_cycle(
    cycle_directory: Path,
    branch_factor: int,
    saturated_min_worker_count: int | None = None,
) -> dict[str, Any]:
    rows = _jsonl(cycle_directory / "raw" / "trials.jsonl")
    result = {
        "schema_version": "1.0",
        "evidence_classification": (
            "measured synthetic workload over independent loopback processes and shaped links"
        ),
        "raw_trial_count": len(rows),
        "failed_trial_count": sum(row.get("status") != "ok" for row in rows),
        "metric_analysis": metric_summary(rows, branch_factor),
        "trace_audit": trace_audit(cycle_directory, rows),
    }
    if saturated_min_worker_count is not None:
        saturated = [row for row in rows if int(row["worker_count"]) >= saturated_min_worker_count]
        result["saturated_min_worker_count"] = saturated_min_worker_count
        result["saturated_metric_analysis"] = metric_summary(saturated, branch_factor)
        _write_json(
            cycle_directory / "saturated-scaling-fits.json",
            result["saturated_metric_analysis"]["fits"],
        )
    _write_json(cycle_directory / "inspection.json", result)
    _write_json(cycle_directory / "scaling-fits.json", result["metric_analysis"]["fits"])
    _write_json(cycle_directory / "trace-audit.json", result["trace_audit"])
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect an Experiment 012 cycle")
    parser.add_argument("--cycle", type=Path, required=True)
    parser.add_argument("--branch-factor", type=int, required=True)
    parser.add_argument("--saturated-min-worker-count", type=int)
    args = parser.parse_args(argv)
    result = inspect_cycle(
        args.cycle.resolve(), args.branch_factor, args.saturated_min_worker_count
    )
    print(
        json.dumps(
            {
                "raw_trial_count": result["raw_trial_count"],
                "failed_trial_count": result["failed_trial_count"],
                "eligible_trial_count": result["metric_analysis"]["eligible_trial_count"],
                "trace_checks_pass": result["trace_audit"]["all_checks_pass"],
                "trace_issues": len(result["trace_audit"]["issues"]),
            },
            indent=2,
        )
    )
    return 0 if result["trace_audit"]["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["inspect_cycle", "metric_summary", "trace_audit"]
