"""H012-008 process experiment for heterogeneous worker role placement."""

from __future__ import annotations

import argparse
import csv
import json
import random
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
    build_capacity_parent_topology,
    build_delegated_topology,
    install_topology,
    topology_record,
)
from swarm_inference.microworker_protocol import NETWORK_PROFILES

HYPOTHESIS_ID = "H012-008"
WORKER_COUNT = 128
BRANCH_FACTOR = 8
PROFILE_SHUFFLE_SEED = 1208
POLICIES = ("identity_balanced", "capacity_parents")
CONDITIONS = ("homogeneous", "heterogeneous_five_class")

PROFILE_SPECS: tuple[tuple[str, int, float, float, int], ...] = (
    ("fast_gpu_like", 16, 0.25, 8.0, 8 * 1024 * 1024),
    ("slower_gpu_like", 24, 1.5, 4.0, 8 * 1024 * 1024),
    ("cpu_like", 40, 5.0, 1.0, 8 * 1024 * 1024),
    ("memory_constrained", 32, 8.0, 0.75, 512),
    ("straggler", 16, 40.0, 0.1, 8 * 1024 * 1024),
)


def worker_profiles(condition: str, count: int = WORKER_COUNT) -> tuple[dict[str, Any], ...]:
    if condition == "homogeneous":
        return tuple(
            {
                "name": "homogeneous",
                "compute_delay_ms": 0.25,
                "capacity_score": 8.0,
                "maximum_payload_bytes": 8 * 1024 * 1024,
            }
            for _ in range(count)
        )
    if condition != "heterogeneous_five_class" or count != WORKER_COUNT:
        raise ValueError("H012-008 heterogeneous profile is locked to 128 workers")
    profiles = [
        {
            "name": name,
            "compute_delay_ms": delay_ms,
            "capacity_score": capacity,
            "maximum_payload_bytes": maximum_payload,
        }
        for name, class_count, delay_ms, capacity, maximum_payload in PROFILE_SPECS
        for _ in range(class_count)
    ]
    random.Random(PROFILE_SHUFFLE_SEED).shuffle(profiles)
    return tuple(profiles)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for policy in POLICIES:
            selected = [
                row
                for row in rows
                if row["worker_condition"] == condition
                and row["topology_policy"] == policy
                and not row["warmup"]
                and row["status"] == "ok"
                and row["correctness"]
            ]
            if not selected:
                continue
            summaries.append(
                {
                    "worker_condition": condition,
                    "topology_policy": policy,
                    "successful_trials": len(selected),
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
                    "critical_path_compute_delay_ms_median": statistics.median(
                        row["critical_path_compute_delay_ms"] for row in selected
                    ),
                    "simulated_compute_delay_ms_sum_median": statistics.median(
                        row["simulated_compute_delay_ms_sum"] for row in selected
                    ),
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
                    "hierarchy_depth_median": statistics.median(
                        row["hierarchy_depth"] for row in selected
                    ),
                    "total_messages_median": statistics.median(
                        row["total_messages"] for row in selected
                    ),
                    "total_bytes_median": statistics.median(row["total_bytes"] for row in selected),
                    "stragglers_median": statistics.median(row["stragglers"] for row in selected),
                }
            )
    return {
        "schema_version": "1.0",
        "hypothesis_id": HYPOTHESIS_ID,
        "evidence_classification": (
            "measured synthetic workload over independent processes with declared compute shaping"
        ),
        "summaries": summaries,
        "failed_trials": [row for row in rows if row["status"] != "ok"],
        "generated_unix_ns": time.time_ns(),
    }


def _write_csv(path: Path, summary: dict[str, Any]) -> None:
    rows = list(summary["summaries"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_h012_008(
    *,
    output_directory: Path,
    payload_bytes: int = 256,
    warmup_trials: int = 1,
    measured_trials: int = 5,
    operation_deadline_s: float = 30.0,
    startup_deadline_s: float = 180.0,
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
    _write_json(
        output_directory / "source-identity.json", _source_identity(repo_root, source_files)
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

    all_rows: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    execution_generation = 1
    profile = NETWORK_PROFILES["same_host_shaped"]
    for condition in CONDITIONS:
        profiles = worker_profiles(condition)
        manifest = {
            "condition": condition,
            "shuffle_seed": PROFILE_SHUFFLE_SEED if condition != "homogeneous" else None,
            "workers": [
                {"worker_index": index, "worker_id": f"worker-{index:06d}", **worker_profile}
                for index, worker_profile in enumerate(profiles)
            ],
        }
        manifest_path = output_directory / "worker-profiles" / f"{condition}.json"
        _write_json(manifest_path, manifest)
        manifest_sha256 = _sha256_file(manifest_path)
        pool = WorkerPool(
            count=WORKER_COUNT,
            directory=raw_directory / f"workers-{condition}-0128" / "processes",
            startup_deadline_s=startup_deadline_s,
            protocol_script=protocol_script,
            runtime_profiles=profiles,
        )
        attempt: dict[str, Any] = {
            "worker_condition": condition,
            "worker_count": WORKER_COUNT,
            "profile_manifest_sha256": manifest_sha256,
            "status": "starting",
            "trial_attempts": [],
            "attempted_unix_ns": time.time_ns(),
        }
        started_ns = time.perf_counter_ns()
        try:
            workers = pool.start()
            _write_json(
                output_directory / "worker-identities" / f"{condition}.json",
                {
                    "condition": condition,
                    "workers": [
                        {
                            "worker_id": worker.worker_id,
                            "worker_index": worker.worker_index,
                            "process_id": worker.process_id,
                            "endpoint": worker.endpoint,
                            "runtime_profile": worker.runtime_profile,
                        }
                        for worker in workers
                    ],
                },
            )
            schedule: list[tuple[str, bool, int]] = [
                (policy, True, warmup_index)
                for warmup_index in range(warmup_trials)
                for policy in POLICIES
            ]
            for trial_index in range(measured_trials):
                ordered_policies = POLICIES if trial_index % 2 == 0 else tuple(reversed(POLICIES))
                schedule.extend((policy, False, trial_index) for policy in ordered_policies)

            for topology_generation, (policy, warmup, trial_index) in enumerate(schedule, start=1):
                topology_id = f"h012-008-{condition}-{policy}-g{topology_generation}"
                builder = (
                    build_delegated_topology
                    if policy == "identity_balanced"
                    else build_capacity_parent_topology
                )
                topology = builder(
                    workers,
                    branch_factor=BRANCH_FACTOR,
                    topology_id=topology_id,
                    route_lease_id=f"h012-008-lease-{condition}-{topology_generation}",
                    route_generation=topology_generation,
                )
                topology_path = (
                    output_directory
                    / "topologies"
                    / f"{condition}-{policy}-g{topology_generation:02d}.json"
                )
                _write_json(topology_path, topology_record(topology))
                installation = install_topology(topology)
                _write_json(
                    output_directory
                    / "topology-installation"
                    / f"{condition}-{policy}-g{topology_generation:02d}.json",
                    installation,
                )
                runner = DelegatedRunner(
                    topology=topology,
                    output_directory=output_directory,
                    payload_bytes=payload_bytes,
                    operation_deadline_s=operation_deadline_s,
                    maximum_root_concurrency=BRANCH_FACTOR,
                    cycle_id=HYPOTHESIS_ID,
                )
                try:
                    row = runner.run_trial(
                        trial_index=trial_index,
                        warmup=warmup,
                        execution_generation=execution_generation,
                        profile=profile,
                        mode="delegated_parallel",
                    )
                finally:
                    runner.close()
                execution_generation += 1
                row.update(
                    {
                        "worker_condition": condition,
                        "topology_policy": policy,
                        "profile_manifest_sha256": manifest_sha256,
                        "topology_record_sha256": _sha256_file(topology_path),
                    }
                )
                all_rows.append(row)
                _append_jsonl(trials_path, row)
                if row["status"] != "ok":
                    _append_jsonl(errors_path, row)
                attempt["trial_attempts"].append(
                    {
                        "policy": policy,
                        "warmup": warmup,
                        "trial_index": trial_index,
                        "status": row["status"],
                        "execution_generation": row["execution_generation"],
                    }
                )
            attempt.update(
                {
                    "status": "completed"
                    if all(item["status"] == "ok" for item in attempt["trial_attempts"])
                    else "failed",
                    "started_processes": len(workers),
                    "memory": pool.sample_memory_snapshot(),
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
            _append_jsonl(errors_path, {"event": "condition_failed", **attempt})
        finally:
            attempt["shutdown"] = pool.stop()
            attempt["elapsed_ms"] = (time.perf_counter_ns() - started_ns) / 1_000_000
            attempts.append(attempt)
            _write_json(raw_directory / "condition-attempts.json", attempts)

    summary = _summary(all_rows)
    summary["attempts"] = attempts
    _write_json(output_directory / "benchmark-summary.json", summary)
    _write_csv(output_directory / "benchmark-summary.csv", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Experiment 012 H012-008")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--payload-bytes", type=int, default=256)
    parser.add_argument("--warmup-trials", type=int, default=1)
    parser.add_argument("--measured-trials", type=int, default=5)
    parser.add_argument("--operation-deadline-s", type=float, default=30.0)
    parser.add_argument("--startup-deadline-s", type=float, default=180.0)
    args = parser.parse_args(argv)
    summary = run_h012_008(
        output_directory=args.output.resolve(),
        payload_bytes=args.payload_bytes,
        warmup_trials=args.warmup_trials,
        measured_trials=args.measured_trials,
        operation_deadline_s=args.operation_deadline_s,
        startup_deadline_s=args.startup_deadline_s,
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


__all__ = [
    "BRANCH_FACTOR",
    "CONDITIONS",
    "HYPOTHESIS_ID",
    "POLICIES",
    "PROFILE_SPECS",
    "WORKER_COUNT",
    "run_h012_008",
    "worker_profiles",
]
