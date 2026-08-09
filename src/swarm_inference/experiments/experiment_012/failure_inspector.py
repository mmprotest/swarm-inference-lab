"""Locked-criteria and trace audit for Experiment 012 hierarchical recovery."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_012.baseline_harness import (
    _percentile,
    _write_json,
)

EXPECTED_WORKER_COUNTS = (32, 128, 512, 1000)
TRANSIENT_SCENARIOS = (
    "single_leaf_failure_once",
    "multiple_leaf_failures_once",
    "intermediate_failure_once",
    "dropped_result_once",
    "stale_generation_once",
)
CONTROL_SCENARIOS = ("clean", "slow_child")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _topology_maps(cycle_directory: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for path in sorted((cycle_directory / "topologies").glob("workers-*.json")):
        topology = json.loads(path.read_text(encoding="utf-8"))
        worker_count = sum(1 for _ in _walk_nodes(topology["root_children"]))
        parents: dict[str, str] = {}
        subtree_sizes: dict[str, int] = {}
        for node in _walk_nodes(topology["root_children"]):
            worker_id = str(node["worker_id"])
            parents[worker_id] = str(node["parent_worker"])
            subtree_sizes[worker_id] = int(node["subtree_worker_count"])
        result[worker_count] = {
            "parents": parents,
            "subtree_sizes": subtree_sizes,
            "root_children": {str(node["worker_id"]) for node in topology["root_children"]},
            "hierarchy_depth": int(topology["hierarchy_depth"]),
        }
    return result


def _walk_nodes(nodes: list[dict[str, Any]]):
    for node in nodes:
        yield node
        yield from _walk_nodes(list(node["children"]))


def _exact_success(row: dict[str, Any]) -> bool:
    return bool(
        row.get("status") == "ok"
        and row.get("correctness")
        and row.get("result_published")
        and row.get("actual") == row.get("expected")
        and row.get("actual_digest") == row.get("expected_digest")
    )


def inspect_failure_cycle(cycle_directory: Path) -> dict[str, Any]:
    rows = _jsonl(cycle_directory / "raw" / "trials.jsonl")
    duplicate_rows = _jsonl(cycle_directory / "raw" / "duplicate-replay.jsonl")
    summary = json.loads((cycle_directory / "benchmark-summary.json").read_text())
    topologies = _topology_maps(cycle_directory)

    root_retries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    root_attempts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in _jsonl(cycle_directory / "traces" / "root.jsonl"):
        operation_id = str(event.get("operation_id", ""))
        if event.get("event") == "root_child_retry":
            root_retries[operation_id].append(event)
        if event.get("event") in {
            "root_delegated_round_trip",
            "root_delegated_round_trip_failed",
        }:
            root_attempts[operation_id].append(event)

    worker_retries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cache_replays: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reductions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    application_errors_retained = 0
    worker_processes: dict[int, set[int]] = defaultdict(set)
    for count_directory in sorted((cycle_directory / "raw").glob("workers-*")):
        try:
            worker_count = int(count_directory.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        for path in sorted((count_directory / "processes").glob("*/trace.jsonl")):
            for event in _jsonl(path):
                operation_id = str(event.get("operation_id", ""))
                kind = event.get("event")
                if kind == "worker_child_retry":
                    worker_retries[operation_id].append(event)
                elif kind == "duplicate_delegated_request_replayed":
                    cache_replays[operation_id].append(event)
                elif kind == "delegated_subtree_reduced":
                    reductions[operation_id].append(event)
                elif kind == "server_request_failed" and event.get("persistent_session_retained"):
                    application_errors_retained += 1
                elif kind == "worker_ready":
                    worker_processes[worker_count].add(int(event["process_id"]))

    non_warm_rows = [row for row in rows if not row.get("warmup")]
    operation_worker_count = {str(row["operation_id"]): int(row["worker_count"]) for row in rows}
    operation_worker_count.update(
        {str(row["operation_id"]): int(row["worker_count"]) for row in duplicate_rows}
    )

    criteria: list[dict[str, Any]] = []

    def criterion(identifier: str, claim: str, passed: bool, evidence: Any) -> None:
        criteria.append(
            {
                "criterion_id": identifier,
                "claim": claim,
                "passed": bool(passed),
                "evidence": evidence,
            }
        )

    attempts = list(summary["attempts"])
    completed_counts = sorted(
        int(item["worker_count"]) for item in attempts if item.get("status") == "completed"
    )
    criterion(
        "H012-011-C01",
        "all declared process scales completed",
        completed_counts == list(EXPECTED_WORKER_COUNTS),
        {"completed_worker_counts": completed_counts},
    )

    process_counts = {
        str(worker_count): len(worker_processes[worker_count])
        for worker_count in EXPECTED_WORKER_COUNTS
    }
    criterion(
        "H012-011-C02",
        "each scale used the declared number of independent process identities",
        all(process_counts[str(count)] == count for count in EXPECTED_WORKER_COUNTS),
        process_counts,
    )

    transient_rows = [
        row for row in non_warm_rows if row.get("failure_scenario") in TRANSIENT_SCENARIOS
    ]
    criterion(
        "H012-011-C03",
        "all 40 one-shot transient trials recover the exact deterministic result",
        len(transient_rows) == 40 and all(_exact_success(row) for row in transient_rows),
        {
            "trials": len(transient_rows),
            "exact_successes": sum(_exact_success(row) for row in transient_rows),
            "incorrect_published_results": sum(
                bool(row.get("result_published")) and not bool(row.get("correctness"))
                for row in transient_rows
            ),
        },
    )

    ownership_records: list[dict[str, Any]] = []
    ownership_pass = True
    for row in transient_rows:
        operation_id = str(row["operation_id"])
        worker_count = int(row["worker_count"])
        topology = topologies[worker_count]
        observed = [*worker_retries[operation_id], *root_retries[operation_id]]
        targets = {str(value) for value in row.get("target_worker_ids", [])}
        for target in sorted(targets):
            expected_parent = str(topology["parents"][target])
            matches = [
                event
                for event in observed
                if str(event.get("receiver_worker_id")) == target
                and str(event.get("sender_worker_id")) == expected_parent
            ]
            passed = len(matches) == 1
            ownership_pass = ownership_pass and passed
            ownership_records.append(
                {
                    "operation_id": operation_id,
                    "target_worker_id": target,
                    "expected_parent": expected_parent,
                    "matching_retry_events": len(matches),
                    "passed": passed,
                }
            )
        ownership_pass = ownership_pass and len(observed) == len(targets)
    criterion(
        "H012-011-C04",
        "each transient retry is owned once by the target's immediate parent",
        ownership_pass,
        {
            "target_checks": len(ownership_records),
            "failed_target_checks": [
                record for record in ownership_records if not record["passed"]
            ],
        },
    )

    root_retry_violations = []
    for operation_id, events in root_retries.items():
        worker_count = operation_worker_count.get(operation_id)
        if worker_count is None:
            root_retry_violations.append(
                {"operation_id": operation_id, "reason": "unknown operation"}
            )
            continue
        direct = topologies[worker_count]["root_children"]
        for event in events:
            receiver = str(event.get("receiver_worker_id"))
            if receiver not in direct:
                root_retry_violations.append(
                    {
                        "operation_id": operation_id,
                        "receiver_worker_id": receiver,
                        "worker_count": worker_count,
                    }
                )
    criterion(
        "H012-011-C05",
        "the root retries only directly connected children",
        not root_retry_violations,
        {
            "root_retry_events": sum(len(events) for events in root_retries.values()),
            "violations": root_retry_violations,
        },
    )

    transient_amplifications = [float(row["latency_amplification"]) for row in transient_rows]
    per_scale_transient_latency = {
        str(worker_count): {
            "p95": _percentile(
                [
                    float(row["latency_amplification"])
                    for row in transient_rows
                    if int(row["worker_count"]) == worker_count
                ],
                95,
            ),
            "maximum": max(
                float(row["latency_amplification"])
                for row in transient_rows
                if int(row["worker_count"]) == worker_count
            ),
        }
        for worker_count in EXPECTED_WORKER_COUNTS
    }
    transient_p95 = _percentile(transient_amplifications, 95)
    criterion(
        "H012-011-C06",
        "recovered-transient latency amplification p95 is at most 2.0",
        transient_p95 <= 2.0,
        {
            "trials": len(transient_amplifications),
            "p95": transient_p95,
            "maximum": max(transient_amplifications),
            "per_scale": per_scale_transient_latency,
        },
    )

    replay_scenarios = {"dropped_result_once", "stale_generation_once"}
    replay_rows = [row for row in transient_rows if row.get("failure_scenario") in replay_scenarios]
    replay_missing = [
        str(row["operation_id"])
        for row in replay_rows
        if not cache_replays[str(row["operation_id"])]
    ]
    criterion(
        "H012-011-C07",
        "retries never duplicate contributions and dropped/stale retries replay cache",
        all(int(row.get("duplicated_work", -1)) == 0 for row in transient_rows)
        and not replay_missing,
        {
            "zero_duplicated_work_trials": sum(
                int(row.get("duplicated_work", -1)) == 0 for row in transient_rows
            ),
            "cache_replay_operations": len(replay_rows) - len(replay_missing),
            "missing_cache_replay_operations": replay_missing,
        },
    )

    cancellation_rows = [
        row for row in non_warm_rows if row.get("failure_scenario") == "cancellation_during_fanout"
    ]
    cancellation_pass = len(cancellation_rows) == 8 and all(
        row.get("status") == "failed"
        and not row.get("result_published")
        and int(row["cancellation"]["propagated_workers"]) == int(row["worker_count"])
        and int(row["cancellation"]["propagation_failures"]) == 0
        and float(row["cancellation"]["elapsed_ms"]) <= 1_000.0
        for row in cancellation_rows
    )
    criterion(
        "H012-011-C08",
        "cancellation reaches every worker within one second and publishes no result",
        cancellation_pass,
        {
            "trials": len(cancellation_rows),
            "maximum_propagation_ms": max(
                float(row["cancellation"]["elapsed_ms"]) for row in cancellation_rows
            ),
            "minimum_reach_ratio": min(
                int(row["cancellation"]["propagated_workers"]) / int(row["worker_count"])
                for row in cancellation_rows
            ),
            "total_propagation_failures": sum(
                int(row["cancellation"]["propagation_failures"]) for row in cancellation_rows
            ),
        },
    )
    criterion(
        "H012-011-C09",
        "root cancellation degree/messages remain bounded by 8/16",
        all(
            int(row["cancellation"]["root_cancel_degree"]) <= 8
            and int(row["cancellation"]["root_cancel_messages"]) <= 16
            for row in cancellation_rows
        ),
        {
            "maximum_root_cancel_degree": max(
                int(row["cancellation"]["root_cancel_degree"]) for row in cancellation_rows
            ),
            "maximum_root_cancel_messages": max(
                int(row["cancellation"]["root_cancel_messages"]) for row in cancellation_rows
            ),
            "tree_messages_by_scale": {
                str(count): sorted(
                    int(row["cancellation"]["tree_messages"])
                    for row in cancellation_rows
                    if int(row["worker_count"]) == count
                )
                for count in EXPECTED_WORKER_COUNTS
            },
        },
    )

    root_bound_violations = [
        {
            "operation_id": row["operation_id"],
            "worker_count": row["worker_count"],
            "root_direct_degree": row["root_direct_degree"],
            "root_rpc_count": row["root_rpc_count"],
            "root_messages_total": row["root_messages_total"],
            "root_leaf_rpc_count": row["root_leaf_rpc_count"],
        }
        for row in non_warm_rows
        if int(row["root_direct_degree"]) > 8
        or int(row["root_rpc_count"]) > 9
        or int(row["root_messages_total"]) > 18
        or int(row["root_leaf_rpc_count"]) != 0
    ]
    criterion(
        "H012-011-C10",
        "operation root degree/RPCs/messages stay within 8/9/18 and leaf RPCs stay zero",
        not root_bound_violations,
        {
            "operations": len(non_warm_rows),
            "violations": root_bound_violations,
            "maxima": {
                "root_direct_degree": max(int(row["root_direct_degree"]) for row in rows),
                "root_rpc_count": max(int(row["root_rpc_count"]) for row in rows),
                "root_messages_total": max(int(row["root_messages_total"]) for row in rows),
                "root_leaf_rpc_count": max(int(row["root_leaf_rpc_count"]) for row in rows),
            },
        },
    )

    duplicate_failures = []
    for row in duplicate_rows:
        operation_id = str(row["operation_id"])
        expected_reductions = int(row["subtree_worker_count"])
        observed_reductions = len(reductions[operation_id])
        if not (
            row.get("correctness")
            and row.get("identical_response")
            and observed_reductions == expected_reductions
        ):
            duplicate_failures.append(
                {
                    "operation_id": operation_id,
                    "expected_reductions": expected_reductions,
                    "observed_reductions": observed_reductions,
                    "correctness": row.get("correctness"),
                    "identical_response": row.get("identical_response"),
                }
            )
    criterion(
        "H012-011-C11",
        "duplicate replay is exact and performs zero extra subtree reductions",
        len(duplicate_rows) == 8 and not duplicate_failures,
        {"trials": len(duplicate_rows), "failures": duplicate_failures},
    )

    permanent_rows = [
        row
        for row in non_warm_rows
        if row.get("failure_scenario") == "permanent_parent_process_loss"
    ]
    criterion(
        "H012-011-C12",
        "permanent parent loss is explicit before three seconds with no partial publish",
        len(permanent_rows) == 4
        and all(
            row.get("status") == "failed"
            and not row.get("result_published")
            and float(row["end_to_end_latency_ms"]) < 3_000.0
            for row in permanent_rows
        ),
        {
            "trials": len(permanent_rows),
            "detection_ms_by_scale": {
                str(row["worker_count"]): row["end_to_end_latency_ms"] for row in permanent_rows
            },
        },
    )

    control_rows = [
        row for row in non_warm_rows if row.get("failure_scenario") in CONTROL_SCENARIOS
    ]
    criterion(
        "H012-011-C13",
        "clean and slow-child controls remain exact",
        len(control_rows) == 16 and all(_exact_success(row) for row in control_rows),
        {"trials": len(control_rows), "exact_successes": sum(map(_exact_success, control_rows))},
    )

    repetition_failures = []
    for worker_count in EXPECTED_WORKER_COUNTS:
        for scenario in (*CONTROL_SCENARIOS, *TRANSIENT_SCENARIOS):
            selected = [
                row
                for row in non_warm_rows
                if int(row["worker_count"]) == worker_count
                and row.get("failure_scenario") == scenario
            ]
            if len(selected) != 2 or not all(_exact_success(row) for row in selected):
                repetition_failures.append(
                    {
                        "worker_count": worker_count,
                        "scenario": scenario,
                        "trials": len(selected),
                        "exact_successes": sum(map(_exact_success, selected)),
                    }
                )
    criterion(
        "H012-011-C14",
        "every non-destructive scenario retains two successful repetitions per scale",
        not repetition_failures,
        {"groups": 28, "failures": repetition_failures},
    )

    successful_rows = [row for row in rows if _exact_success(row)]
    reduction_failures = [
        {
            "operation_id": row["operation_id"],
            "worker_count": row["worker_count"],
            "observed_reductions": len(reductions[str(row["operation_id"])]),
        }
        for row in successful_rows
        if len(reductions[str(row["operation_id"])]) != int(row["worker_count"])
    ]
    criterion(
        "H012-011-C15",
        "every successful operation performs one deterministic reduction per worker",
        not reduction_failures,
        {"operations": len(successful_rows), "failures": reduction_failures},
    )

    failed_criteria = [item["criterion_id"] for item in criteria if not item["passed"]]
    result = {
        "schema_version": "1.0",
        "hypothesis_id": "H012-011",
        "evidence_classification": (
            "measured synthetic fault injection over independent loopback worker processes"
        ),
        "result": "PASS" if not failed_criteria else "FAIL",
        "criteria": criteria,
        "failed_criteria": failed_criteria,
        "raw_counts": {
            "trial_rows": len(rows),
            "non_warm_trial_rows": len(non_warm_rows),
            "duplicate_rows": len(duplicate_rows),
            "transient_rows": len(transient_rows),
            "cancellation_rows": len(cancellation_rows),
            "permanent_failure_rows": len(permanent_rows),
            "application_error_sessions_retained": application_errors_retained,
            "root_retry_events": sum(len(events) for events in root_retries.values()),
            "worker_retry_events": sum(len(events) for events in worker_retries.values()),
            "cache_replay_events": sum(len(events) for events in cache_replays.values()),
            "reduction_events": sum(len(events) for events in reductions.values()),
            "root_attempt_events": sum(len(events) for events in root_attempts.values()),
        },
    }
    _write_json(cycle_directory / "failure-inspection.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect H012 hierarchical failure evidence")
    parser.add_argument("--cycle", required=True, type=Path)
    args = parser.parse_args(argv)
    result = inspect_failure_cycle(args.cycle.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["inspect_failure_cycle"]
