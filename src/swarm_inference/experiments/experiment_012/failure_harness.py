"""H012-010 audit of the pre-recovery hierarchical failure behavior."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from uuid import uuid4

from swarm_inference.experiments.experiment_012.baseline_harness import (
    WorkerPool,
    _append_jsonl,
    _percentile,
    _sha256_file,
    _source_identity,
    _write_json,
)
from swarm_inference.experiments.experiment_012.delegation_harness import (
    DelegatedNode,
    DelegatedRunner,
    _control_round_trip,
    build_delegated_topology,
    install_topology,
    topology_record,
)
from swarm_inference.microworker_protocol import MAGIC, NETWORK_PROFILES, PROTOCOL_VERSION

HYPOTHESIS_ID = "H012-010"
WORKER_COUNTS = (32, 128, 512, 1000)
BRANCH_FACTOR = 8
REPEATED_SCENARIOS = (
    "clean",
    "slow_child",
    "single_leaf_failure_once",
    "multiple_leaf_failures_once",
    "intermediate_failure_once",
    "dropped_result_once",
    "stale_generation_once",
    "cancellation_during_fanout",
)


def _descendants(node: DelegatedNode) -> list[DelegatedNode]:
    values = [node]
    for child in node.children:
        values.extend(_descendants(child))
    return values


def _scenario_control(topology: Any, scenario: str) -> tuple[dict[str, Any], int, list[str]]:
    leaves_by_root = [
        [node for node in _descendants(root) if not node.children]
        for root in topology.root_children
    ]
    leaf = leaves_by_root[0][-1]
    intermediate = next(
        (
            node
            for root in topology.root_children
            for node in _descendants(root)
            if node.children and node.parent_worker != "stage-owner"
        ),
        topology.root_children[0],
    )
    if scenario == "clean":
        return {}, 0, []
    if scenario == "slow_child":
        return (
            {
                "kind": "slow_child",
                "target_worker_ids": [leaf.worker.worker_id],
                "one_shot": False,
                "delay_ms": 250.0,
            },
            1,
            [leaf.worker.worker_id],
        )
    if scenario == "single_leaf_failure_once":
        return (
            {
                "kind": "worker_failure_once",
                "target_worker_ids": [leaf.worker.worker_id],
                "one_shot": True,
            },
            1,
            [leaf.worker.worker_id],
        )
    if scenario == "multiple_leaf_failures_once":
        targets = [group[-1].worker.worker_id for group in leaves_by_root[:3]]
        return (
            {
                "kind": "worker_failure_once",
                "target_worker_ids": targets,
                "one_shot": True,
            },
            len(targets),
            targets,
        )
    if scenario == "intermediate_failure_once":
        return (
            {
                "kind": "worker_failure_once",
                "target_worker_ids": [intermediate.worker.worker_id],
                "one_shot": True,
            },
            intermediate.subtree_worker_count,
            [intermediate.worker.worker_id],
        )
    if scenario == "dropped_result_once":
        return (
            {
                "kind": "drop_result_once",
                "target_worker_ids": [leaf.worker.worker_id],
                "one_shot": True,
            },
            1,
            [leaf.worker.worker_id],
        )
    if scenario == "stale_generation_once":
        return (
            {
                "kind": "stale_generation_once",
                "target_worker_ids": [leaf.worker.worker_id],
                "one_shot": True,
            },
            1,
            [leaf.worker.worker_id],
        )
    if scenario == "cancellation_during_fanout":
        return (
            {
                "kind": "slow_child",
                "target_worker_ids": [leaf.worker.worker_id],
                "one_shot": False,
                "delay_ms": 500.0,
            },
            leaf.subtree_worker_count,
            [leaf.worker.worker_id],
        )
    raise ValueError(f"unknown H012-010 scenario {scenario!r}")


def _send_bounded_cancellation(topology: Any, operation_id: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    lock = threading.Lock()

    def cancel(node: DelegatedNode) -> None:
        started_ns = time.perf_counter_ns()
        try:
            response, metrics = _control_round_trip(
                node.worker,
                {
                    "magic": MAGIC,
                    "protocol_version": PROTOCOL_VERSION,
                    "kind": "cancel_operation",
                    "operation_id": operation_id,
                },
                timeout_s=2.0,
            )
            value = {
                "worker_id": node.worker.worker_id,
                "status": response.get("status"),
                "messages": 2,
                "bytes": metrics["request_bytes"] + metrics["response_bytes"],
                "elapsed_ns": metrics["elapsed_ns"],
                "propagated_workers": int(response.get("propagated_workers", 1)),
                "tree_messages": int(response.get("tree_messages", 0)),
                "cancellation_depth": int(response.get("cancellation_depth", 0)),
                "propagation_failures": int(response.get("propagation_failures", 0)),
            }
        except BaseException as error:
            value = {
                "worker_id": node.worker.worker_id,
                "status": "failed",
                "messages": 0,
                "bytes": 0,
                "elapsed_ns": time.perf_counter_ns() - started_ns,
                "propagated_workers": 0,
                "tree_messages": 0,
                "cancellation_depth": 0,
                "propagation_failures": 1,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        with lock:
            records.append(value)

    with ThreadPoolExecutor(max_workers=len(topology.root_children)) as executor:
        futures = [executor.submit(cancel, node) for node in topology.root_children]
        for future in futures:
            future.result()
    return {
        "operation_id": operation_id,
        "root_cancel_degree": len(topology.root_children),
        "root_cancel_messages": sum(record["messages"] for record in records),
        "root_cancel_bytes": sum(record["bytes"] for record in records),
        "elapsed_ms": max((record["elapsed_ns"] for record in records), default=0) / 1_000_000,
        "propagated_workers": sum(record["propagated_workers"] for record in records),
        "tree_messages": sum(record["tree_messages"] for record in records),
        "cancellation_depth": 1
        + max((record["cancellation_depth"] for record in records), default=0),
        "propagation_failures": sum(record["propagation_failures"] for record in records),
        "records": sorted(records, key=lambda record: record["worker_id"]),
    }


def _duplicate_replay(
    runner: DelegatedRunner,
    *,
    execution_generation: int,
    repetition: int,
) -> dict[str, Any]:
    node = runner.topology.root_children[0]
    operation_id = (
        f"{runner.cycle_id.lower()}-n{len(runner.topology.nodes)}-duplicate-replay-"
        f"r{repetition}-{uuid4().hex[:8]}"
    )
    deadline_unix_ns = time.time_ns() + 3_000_000_000
    arguments = {
        "node": node,
        "operation_id": operation_id,
        "execution_generation": execution_generation,
        "profile": NETWORK_PROFILES["same_host_shaped"],
        "deadline_unix_ns": deadline_unix_ns,
        "dispatch_policy": "parallel",
        "connection_policy": "persistent",
        "retry_policy": {"max_attempts": 1, "backoff_ms": 0.0},
        "fault_control": {},
    }
    first = runner._root_child_request(**arguments)
    second = runner._root_child_request(**arguments)
    first_response = first[1]
    second_response = second[1]
    return {
        "scenario": "duplicate_request_replay",
        "repetition": repetition,
        "operation_id": operation_id,
        "worker_count": len(runner.topology.nodes),
        "root_direct_degree": 1,
        "root_leaf_rpc_count": int(not node.children),
        "root_rpc_attempts": 2,
        "root_messages": 4,
        "identical_response": first_response == second_response,
        "correctness": (
            first_response["aggregate"] == second_response["aggregate"]
            and first_response["aggregate_digest"] == second_response["aggregate_digest"]
        ),
        "subtree_worker_count": node.subtree_worker_count,
        "first_connection_count": first[2]["new_connection_count"],
        "second_connection_count": second[2]["new_connection_count"],
        "measured_unix_ns": time.time_ns(),
    }


def _summarize(
    rows: list[dict[str, Any]],
    duplicate_rows: list[dict[str, Any]],
    *,
    hypothesis_id: str,
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for worker_count in WORKER_COUNTS:
        for scenario in (*REPEATED_SCENARIOS, "permanent_parent_process_loss"):
            selected = [
                row
                for row in rows
                if int(row["worker_count"]) == worker_count
                and row["failure_scenario"] == scenario
                and not row["warmup"]
            ]
            if not selected:
                continue
            summaries.append(
                {
                    "worker_count": worker_count,
                    "scenario": scenario,
                    "trials": len(selected),
                    "successful": sum(row["status"] == "ok" for row in selected),
                    "failed": sum(row["status"] != "ok" for row in selected),
                    "published_results": sum(bool(row["result_published"]) for row in selected),
                    "incorrect_published_results": sum(
                        bool(row["result_published"]) and not bool(row["correctness"])
                        for row in selected
                    ),
                    "detection_time_p50_ms": _percentile(
                        [row["end_to_end_latency_ms"] for row in selected], 50
                    ),
                    "detection_time_p95_ms": _percentile(
                        [row["end_to_end_latency_ms"] for row in selected], 95
                    ),
                    "latency_amplification_median": statistics.median(
                        row["latency_amplification"] for row in selected
                    ),
                    "root_rpc_attempts_median": statistics.median(
                        row["root_rpc_count"] for row in selected
                    ),
                    "root_messages_median": statistics.median(
                        row["root_messages_total"] for row in selected
                    ),
                    "root_direct_degree_median": statistics.median(
                        row["root_direct_degree"] for row in selected
                    ),
                    "root_leaf_rpc_count_median": statistics.median(
                        row["root_leaf_rpc_count"] for row in selected
                    ),
                    "retries_median": statistics.median(row["retries"] for row in selected),
                    "affected_subtree_size_median": statistics.median(
                        row["affected_subtree_size"] for row in selected
                    ),
                }
            )
    return {
        "schema_version": "1.0",
        "hypothesis_id": hypothesis_id,
        "evidence_classification": (
            "measured synthetic fault injection over independent process hierarchy"
        ),
        "summaries": summaries,
        "duplicate_replay": duplicate_rows,
        "failed_operation_rows": [row for row in rows if row["status"] != "ok"],
        "generated_unix_ns": time.time_ns(),
    }


def _write_csv(path: Path, summary: dict[str, Any]) -> None:
    rows = list(summary["summaries"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_h012_010(
    *,
    output_directory: Path,
    worker_counts: tuple[int, ...] = WORKER_COUNTS,
    repetitions: int = 2,
    payload_bytes: int = 256,
    operation_deadline_s: float = 3.0,
    startup_deadline_s: float = 180.0,
    hypothesis_id: str = HYPOTHESIS_ID,
    hypothesis_path: Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[4]
    protocol_script = repo_root / "src" / "swarm_inference" / "microworker_protocol.py"
    source_files = [
        protocol_script,
        Path(__file__).with_name("baseline_harness.py"),
        Path(__file__).with_name("delegation_harness.py"),
        Path(__file__).resolve(),
    ]
    raw_directory = output_directory / "raw"
    raw_directory.mkdir(parents=True, exist_ok=True)
    trials_path = raw_directory / "trials.jsonl"
    errors_path = raw_directory / "errors.jsonl"
    duplicate_path = raw_directory / "duplicate-replay.jsonl"
    _write_json(
        output_directory / "source-identity.json", _source_identity(repo_root, source_files)
    )
    snapshot_directory = raw_directory / "source-snapshot"
    snapshot_directory.mkdir(parents=True, exist_ok=True)
    for source_file in source_files:
        shutil.copy2(source_file, snapshot_directory / source_file.name)
    hypothesis_path = hypothesis_path or output_directory / "hypothesis.json"
    _write_json(
        output_directory / "hypothesis-identity.json",
        {
            "path": str(hypothesis_path),
            "sha256": _sha256_file(hypothesis_path),
            "bytes": hypothesis_path.stat().st_size,
        },
    )

    rows: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    execution_generation = 1
    for worker_count in worker_counts:
        pool = WorkerPool(
            count=worker_count,
            directory=raw_directory / f"workers-{worker_count:04d}" / "processes",
            startup_deadline_s=startup_deadline_s,
            protocol_script=protocol_script,
        )
        attempt: dict[str, Any] = {
            "worker_count": worker_count,
            "status": "starting",
            "attempted_unix_ns": time.time_ns(),
        }
        started_ns = time.perf_counter_ns()
        runner: DelegatedRunner | None = None
        try:
            workers = pool.start()
            topology = build_delegated_topology(
                workers,
                branch_factor=BRANCH_FACTOR,
                topology_id=f"{hypothesis_id.lower()}-topology-n{worker_count}",
                route_lease_id=f"{hypothesis_id.lower()}-lease-n{worker_count}",
                route_generation=1,
            )
            _write_json(
                output_directory / "topologies" / f"workers-{worker_count:04d}.json",
                topology_record(topology),
            )
            _write_json(
                output_directory / "topology-installation" / f"workers-{worker_count:04d}.json",
                install_topology(topology),
            )
            runner = DelegatedRunner(
                topology=topology,
                output_directory=output_directory,
                payload_bytes=payload_bytes,
                operation_deadline_s=operation_deadline_s,
                maximum_root_concurrency=BRANCH_FACTOR,
                cycle_id=hypothesis_id,
            )
            profile = NETWORK_PROFILES["same_host_shaped"]
            warmup = runner.run_trial(
                trial_index=0,
                warmup=True,
                execution_generation=execution_generation,
                profile=profile,
                mode="delegated_parallel_persistent",
            )
            execution_generation += 1
            warmup.update(
                {
                    "failure_scenario": "clean",
                    "repetition": -1,
                    "affected_subtree_size": 0,
                    "target_worker_ids": [],
                    "latency_amplification": 1.0,
                    "cancellation": None,
                }
            )
            rows.append(warmup)
            _append_jsonl(trials_path, warmup)

            clean_rows: list[dict[str, Any]] = []
            for repetition in range(repetitions):
                row = runner.run_trial(
                    trial_index=repetition,
                    warmup=False,
                    execution_generation=execution_generation,
                    profile=profile,
                    mode="delegated_parallel_persistent",
                )
                execution_generation += 1
                row.update(
                    {
                        "failure_scenario": "clean",
                        "repetition": repetition,
                        "affected_subtree_size": 0,
                        "target_worker_ids": [],
                        "latency_amplification": 1.0,
                        "cancellation": None,
                    }
                )
                clean_rows.append(row)
                rows.append(row)
                _append_jsonl(trials_path, row)
            clean_latency_ms = statistics.median(row["end_to_end_latency_ms"] for row in clean_rows)

            for repetition in range(repetitions):
                duplicate = _duplicate_replay(
                    runner,
                    execution_generation=execution_generation,
                    repetition=repetition,
                )
                execution_generation += 1
                duplicates.append(duplicate)
                _append_jsonl(duplicate_path, duplicate)

            for scenario in REPEATED_SCENARIOS[1:]:
                for repetition in range(repetitions):
                    control, affected_size, target_ids = _scenario_control(topology, scenario)
                    operation_id = (
                        f"{hypothesis_id.lower()}-n{worker_count}-{scenario}-"
                        f"r{repetition}-{uuid4().hex[:8]}"
                    )
                    cancellation: dict[str, Any] | None = None
                    if scenario == "cancellation_during_fanout":
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(
                                runner.run_trial,
                                trial_index=repetition,
                                warmup=False,
                                execution_generation=execution_generation,
                                profile=profile,
                                mode="delegated_parallel_persistent",
                                operation_id_override=operation_id,
                                retry_policy={"max_attempts": 1, "backoff_ms": 0.0},
                                fault_control=control,
                            )
                            time.sleep(0.05)
                            cancellation = _send_bounded_cancellation(topology, operation_id)
                            row = future.result(timeout=operation_deadline_s + 2.0)
                    else:
                        row = runner.run_trial(
                            trial_index=repetition,
                            warmup=False,
                            execution_generation=execution_generation,
                            profile=profile,
                            mode="delegated_parallel_persistent",
                            operation_id_override=operation_id,
                            retry_policy={"max_attempts": 1, "backoff_ms": 0.0},
                            fault_control=control,
                        )
                    execution_generation += 1
                    row.update(
                        {
                            "failure_scenario": scenario,
                            "repetition": repetition,
                            "affected_subtree_size": affected_size,
                            "target_worker_ids": target_ids,
                            "latency_amplification": (
                                row["end_to_end_latency_ms"] / clean_latency_ms
                                if clean_latency_ms
                                else 0.0
                            ),
                            "cancellation": cancellation,
                        }
                    )
                    rows.append(row)
                    _append_jsonl(trials_path, row)
                    if row["status"] != "ok":
                        _append_jsonl(errors_path, row)

            parent = next(
                (
                    node
                    for root in topology.root_children
                    for node in _descendants(root)
                    if node.children and node.parent_worker != "stage-owner"
                ),
                topology.root_children[0],
            )
            parent.worker.process.terminate()
            parent.worker.process.wait(timeout=2.0)
            termination_record = {
                "worker_id": parent.worker.worker_id,
                "worker_index": parent.worker.worker_index,
                "process_id": parent.worker.process_id,
                "subtree_worker_count": parent.subtree_worker_count,
                "returncode": parent.worker.process.returncode,
                "terminated_unix_ns": time.time_ns(),
            }
            _write_json(
                output_directory
                / "fault-injections"
                / f"workers-{worker_count:04d}-parent-process-loss.json",
                termination_record,
            )
            row = runner.run_trial(
                trial_index=0,
                warmup=False,
                execution_generation=execution_generation,
                profile=profile,
                mode="delegated_parallel_persistent",
                operation_id_override=(
                    f"{hypothesis_id.lower()}-n{worker_count}-"
                    f"permanent-parent-process-loss-{uuid4().hex[:8]}"
                ),
                retry_policy={"max_attempts": 1, "backoff_ms": 0.0},
            )
            execution_generation += 1
            row.update(
                {
                    "failure_scenario": "permanent_parent_process_loss",
                    "repetition": 0,
                    "affected_subtree_size": parent.subtree_worker_count,
                    "target_worker_ids": [parent.worker.worker_id],
                    "latency_amplification": (
                        row["end_to_end_latency_ms"] / clean_latency_ms if clean_latency_ms else 0.0
                    ),
                    "cancellation": None,
                }
            )
            rows.append(row)
            _append_jsonl(trials_path, row)
            if row["status"] != "ok":
                _append_jsonl(errors_path, row)
            attempt.update(
                {
                    "status": "completed",
                    "started_processes": len(workers),
                    "memory": pool.sample_memory_snapshot(),
                    "injected_parent_process_loss": termination_record,
                }
            )
        except BaseException as error:
            attempt.update(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "started_processes": len(pool.workers),
                }
            )
            _append_jsonl(errors_path, {"event": "scale_failed", **attempt})
        finally:
            if runner is not None:
                runner.close()
            attempt["shutdown"] = pool.stop()
            attempt["elapsed_ms"] = (time.perf_counter_ns() - started_ns) / 1_000_000
            attempts.append(attempt)
            _write_json(raw_directory / "scale-attempts.json", attempts)

    summary = _summarize(rows, duplicates, hypothesis_id=hypothesis_id)
    summary["attempts"] = attempts
    _write_json(output_directory / "benchmark-summary.json", summary)
    _write_csv(output_directory / "benchmark-summary.csv", summary)
    return summary


def _parse_counts(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("at least one worker count is required")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Experiment 012 H012-010")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counts", type=_parse_counts, default=WORKER_COUNTS)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--payload-bytes", type=int, default=256)
    parser.add_argument("--operation-deadline-s", type=float, default=3.0)
    parser.add_argument("--startup-deadline-s", type=float, default=180.0)
    parser.add_argument("--hypothesis-id", default=HYPOTHESIS_ID)
    parser.add_argument("--hypothesis-path", type=Path)
    args = parser.parse_args(argv)
    summary = run_h012_010(
        output_directory=args.output.resolve(),
        worker_counts=args.counts,
        repetitions=args.repetitions,
        payload_bytes=args.payload_bytes,
        operation_deadline_s=args.operation_deadline_s,
        startup_deadline_s=args.startup_deadline_s,
        hypothesis_id=args.hypothesis_id,
        hypothesis_path=args.hypothesis_path.resolve() if args.hypothesis_path else None,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if all(attempt["status"] == "completed" for attempt in summary["attempts"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["HYPOTHESIS_ID", "REPEATED_SCENARIOS", "WORKER_COUNTS", "run_h012_010"]
