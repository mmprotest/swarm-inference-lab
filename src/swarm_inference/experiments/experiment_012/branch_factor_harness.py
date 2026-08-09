"""H012-005 fixed branch-factor comparison harness."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_012.baseline_harness import (
    WorkerPool,
    _append_jsonl,
    _percentile,
    _sha256_file,
    _source_identity,
    _write_json,
)
from swarm_inference.experiments.experiment_012.delegation_harness import (
    DelegatedRunner,
    build_delegated_topology,
    install_topology,
    topology_record,
)
from swarm_inference.microworker_protocol import NETWORK_PROFILES

HYPOTHESIS_ID = "H012-005"
BRANCH_FACTORS = (8, 4, 16, 32)
MODE = "delegated_parallel"


def _summarize(rows: list[dict[str, Any]], *, hypothesis_id: str = HYPOTHESIS_ID) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for profile in sorted({str(row["network_profile"]) for row in rows}):
        for worker_count in sorted({int(row["worker_count"]) for row in rows}):
            for branch_factor in sorted({int(row["branch_factor"]) for row in rows}):
                selected = [
                    row
                    for row in rows
                    if row["network_profile"] == profile
                    and int(row["worker_count"]) == worker_count
                    and int(row["branch_factor"]) == branch_factor
                    and not row["warmup"]
                    and row["status"] == "ok"
                    and row["correctness"]
                ]
                if not selected:
                    continue
                summaries.append(
                    {
                        "network_profile": profile,
                        "worker_count": worker_count,
                        "branch_factor": branch_factor,
                        "successful_trials": len(selected),
                        "root_messages_median": statistics.median(
                            row["root_messages_total"] for row in selected
                        ),
                        "root_bytes_median": statistics.median(
                            row["root_bytes_total"] for row in selected
                        ),
                        "root_serial_waits_median": statistics.median(
                            row["root_serial_waits"] for row in selected
                        ),
                        "root_direct_degree_median": statistics.median(
                            row["root_direct_degree"] for row in selected
                        ),
                        "root_leaf_rpc_count_median": statistics.median(
                            row["root_leaf_rpc_count"] for row in selected
                        ),
                        "root_cpu_ms_median": statistics.median(
                            row["root_cpu_ns"] / 1_000_000 for row in selected
                        ),
                        "end_to_end_latency_p50_ms": _percentile(
                            [row["end_to_end_latency_ms"] for row in selected], 50
                        ),
                        "end_to_end_latency_p95_ms": _percentile(
                            [row["end_to_end_latency_ms"] for row in selected], 95
                        ),
                        "end_to_end_latency_p99_ms": _percentile(
                            [row["end_to_end_latency_ms"] for row in selected], 99
                        ),
                        "throughput_ops_s_median": statistics.median(
                            row["throughput_ops_s"] for row in selected
                        ),
                        "hierarchy_depth_median": statistics.median(
                            row["hierarchy_depth"] for row in selected
                        ),
                        "critical_path_sync_points_median": statistics.median(
                            row["critical_path_sync_points"] for row in selected
                        ),
                        "parallel_dispatch_nodes_median": statistics.median(
                            row["parallel_dispatch_nodes"] for row in selected
                        ),
                        "worker_barrier_waits_median": statistics.median(
                            row["worker_barrier_waits"] for row in selected
                        ),
                        "maximum_queue_depth_median": statistics.median(
                            row["maximum_queue_depth"] for row in selected
                        ),
                        "total_worker_cpu_ms_median": statistics.median(
                            row["total_worker_cpu_ns"] / 1_000_000 for row in selected
                        ),
                        "total_messages_median": statistics.median(
                            row["total_messages"] for row in selected
                        ),
                        "total_bytes_median": statistics.median(
                            row["total_bytes"] for row in selected
                        ),
                        "total_connection_count_median": statistics.median(
                            row["total_connection_count"] for row in selected
                        ),
                    }
                )
    return {
        "schema_version": "1.0",
        "hypothesis_id": hypothesis_id,
        "evidence_classification": (
            "measured synthetic workload over independent loopback processes and shaped links"
        ),
        "summaries": summaries,
        "failed_trials": [row for row in rows if row["status"] != "ok"],
        "generated_unix_ns": time.time_ns(),
    }


def _write_csv(path: Path, summary: dict[str, Any]) -> None:
    rows = list(summary["summaries"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_h012_005(
    *,
    output_directory: Path,
    worker_counts: tuple[int, ...],
    branch_factors: tuple[int, ...],
    profiles: tuple[str, ...],
    payload_bytes: int,
    warmup_trials: int,
    measured_trials: int,
    operation_deadline_s: float,
    startup_deadline_s: float,
    hypothesis_id: str = HYPOTHESIS_ID,
    additional_source_files: tuple[Path, ...] = (),
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[4]
    protocol_script = repo_root / "src" / "swarm_inference" / "microworker_protocol.py"
    delegation_harness = Path(__file__).with_name("delegation_harness.py")
    source_files = [
        protocol_script,
        delegation_harness,
        Path(__file__).resolve(),
        *additional_source_files,
    ]
    raw_directory = output_directory / "raw"
    raw_directory.mkdir(parents=True, exist_ok=True)
    trials_path = raw_directory / "trials.jsonl"
    errors_path = raw_directory / "errors.jsonl"
    _write_json(
        output_directory / "source-identity.json",
        _source_identity(repo_root, source_files),
    )
    snapshot_directory = raw_directory / "source-snapshot"
    snapshot_directory.mkdir(parents=True, exist_ok=True)
    for source_file in source_files:
        shutil.copy2(source_file, snapshot_directory / source_file.name)
    hypothesis_path = output_directory / "hypothesis.json"
    _write_json(
        output_directory / "hypothesis-identity.json",
        {
            "path": str(hypothesis_path),
            "sha256": _sha256_file(hypothesis_path),
            "bytes": hypothesis_path.stat().st_size,
        },
    )

    rows: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    execution_generation = 1
    for worker_count in worker_counts:
        scale_started_ns = time.perf_counter_ns()
        pool = WorkerPool(
            count=worker_count,
            directory=raw_directory / f"workers-{worker_count:04d}" / "processes",
            startup_deadline_s=startup_deadline_s,
            protocol_script=protocol_script,
        )
        scale_attempt: dict[str, Any] = {
            "worker_count": worker_count,
            "status": "starting",
            "branch_attempts": [],
            "attempted_unix_ns": time.time_ns(),
        }
        try:
            workers = pool.start()
            for topology_generation, branch_factor in enumerate(branch_factors, start=1):
                branch_started_ns = time.perf_counter_ns()
                branch_attempt: dict[str, Any] = {
                    "branch_factor": branch_factor,
                    "status": "starting",
                }
                runner: DelegatedRunner | None = None
                try:
                    topology = build_delegated_topology(
                        workers,
                        branch_factor=branch_factor,
                        topology_id=(
                            f"{hypothesis_id.lower()}-topology-n{worker_count}-b{branch_factor}"
                        ),
                        route_lease_id=(
                            f"{hypothesis_id.lower()}-lease-n{worker_count}-b{branch_factor}"
                        ),
                        route_generation=topology_generation,
                    )
                    _write_json(
                        output_directory
                        / "topologies"
                        / f"workers-{worker_count:04d}-b{branch_factor:02d}.json",
                        topology_record(topology),
                    )
                    installation = install_topology(topology)
                    _write_json(
                        output_directory
                        / "topology-installation"
                        / f"workers-{worker_count:04d}-b{branch_factor:02d}.json",
                        installation,
                    )
                    runner = DelegatedRunner(
                        topology=topology,
                        output_directory=output_directory,
                        payload_bytes=payload_bytes,
                        operation_deadline_s=operation_deadline_s,
                        maximum_root_concurrency=branch_factor,
                        cycle_id=hypothesis_id,
                    )
                    for profile_name in profiles:
                        profile = NETWORK_PROFILES[profile_name]
                        for warmup_index in range(warmup_trials):
                            row = runner.run_trial(
                                trial_index=warmup_index,
                                warmup=True,
                                execution_generation=execution_generation,
                                profile=profile,
                                mode=MODE,
                            )
                            execution_generation += 1
                            rows.append(row)
                            _append_jsonl(trials_path, row)
                            if row["status"] != "ok":
                                _append_jsonl(errors_path, row)
                        for trial_index in range(measured_trials):
                            row = runner.run_trial(
                                trial_index=trial_index,
                                warmup=False,
                                execution_generation=execution_generation,
                                profile=profile,
                                mode=MODE,
                            )
                            execution_generation += 1
                            rows.append(row)
                            _append_jsonl(trials_path, row)
                            if row["status"] != "ok":
                                _append_jsonl(errors_path, row)
                    branch_attempt.update(
                        {
                            "status": "completed",
                            "root_direct_degree": len(topology.root_children),
                            "root_leaf_rpc_count": sum(
                                not node.children for node in topology.root_children
                            ),
                            "topology_depth": topology.depth,
                        }
                    )
                except BaseException as error:
                    branch_attempt.update(
                        {
                            "status": "failed",
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
                    _append_jsonl(
                        errors_path,
                        {"event": "branch_factor_failed", **branch_attempt},
                    )
                finally:
                    if runner is not None:
                        runner.close()
                    branch_attempt["elapsed_ms"] = (
                        time.perf_counter_ns() - branch_started_ns
                    ) / 1_000_000
                    scale_attempt["branch_attempts"].append(branch_attempt)
            scale_attempt.update(
                {
                    "status": "completed"
                    if all(
                        item["status"] == "completed" for item in scale_attempt["branch_attempts"]
                    )
                    else "failed",
                    "started_processes": len(workers),
                    "memory": pool.sample_memory_snapshot(),
                }
            )
        except BaseException as error:
            scale_attempt.update(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "started_processes": len(pool.workers),
                }
            )
            _append_jsonl(errors_path, {"event": "scale_failed", **scale_attempt})
        finally:
            scale_attempt["shutdown"] = pool.stop()
            scale_attempt["elapsed_ms"] = (time.perf_counter_ns() - scale_started_ns) / 1_000_000
            attempts.append(scale_attempt)
            _write_json(raw_directory / "scale-attempts.json", attempts)
    summary = _summarize(rows, hypothesis_id=hypothesis_id)
    summary["attempts"] = attempts
    _write_json(output_directory / "benchmark-summary.json", summary)
    _write_csv(output_directory / "benchmark-summary.csv", summary)
    return summary


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return parsed


def _parse_profiles(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(parsed) - NETWORK_PROFILES.keys()
    if not parsed or unknown:
        raise argparse.ArgumentTypeError(f"unknown network profiles: {sorted(unknown)}")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Experiment 012 H012-005")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counts", type=_parse_csv_ints, default=(512,))
    parser.add_argument("--branch-factors", type=_parse_csv_ints, default=BRANCH_FACTORS)
    parser.add_argument("--profiles", type=_parse_profiles, default=("same_host_shaped",))
    parser.add_argument("--payload-bytes", type=int, default=256)
    parser.add_argument("--hypothesis-id", default=HYPOTHESIS_ID)
    parser.add_argument("--warmup-trials", type=int, default=1)
    parser.add_argument("--measured-trials", type=int, default=3)
    parser.add_argument("--operation-deadline-s", type=float, default=30.0)
    parser.add_argument("--startup-deadline-s", type=float, default=180.0)
    args = parser.parse_args(argv)
    summary = run_h012_005(
        output_directory=args.output.resolve(),
        worker_counts=args.counts,
        branch_factors=args.branch_factors,
        profiles=args.profiles,
        payload_bytes=args.payload_bytes,
        warmup_trials=args.warmup_trials,
        measured_trials=args.measured_trials,
        operation_deadline_s=args.operation_deadline_s,
        startup_deadline_s=args.startup_deadline_s,
        hypothesis_id=args.hypothesis_id,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return (
        1
        if summary["failed_trials"]
        or any(attempt["status"] != "completed" for attempt in summary["attempts"])
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["BRANCH_FACTORS", "HYPOTHESIS_ID", "run_h012_005"]
