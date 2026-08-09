"""H012-009 paired tail-latency experiment on the capacity-parent topology."""

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
    build_capacity_parent_topology,
    install_topology,
    topology_record,
)
from swarm_inference.microworker_protocol import NETWORK_PROFILES

HYPOTHESIS_ID = "H012-009"
WORKER_COUNT = 128
BRANCH_FACTOR = 8
MODES = ("delegated_parallel", "delegated_parallel_persistent")


def _load_profiles(path: Path) -> tuple[dict[str, Any], ...]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    workers = sorted(manifest["workers"], key=lambda row: int(row["worker_index"]))
    if len(workers) != WORKER_COUNT or [int(row["worker_index"]) for row in workers] != list(
        range(WORKER_COUNT)
    ):
        raise ValueError("H012-009 requires the exact complete H012-008 worker manifest")
    return tuple(
        {
            "name": str(row["name"]),
            "compute_delay_ms": float(row["compute_delay_ms"]),
            "capacity_score": float(row["capacity_score"]),
            "maximum_payload_bytes": int(row["maximum_payload_bytes"]),
        }
        for row in workers
    )


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for mode in MODES:
        selected = [
            row
            for row in rows
            if row["mode"] == mode
            and not row["warmup"]
            and row["status"] == "ok"
            and row["correctness"]
        ]
        if not selected:
            continue
        summaries.append(
            {
                "mode": mode,
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
                "latency_mean_ms": statistics.fmean(
                    row["end_to_end_latency_ms"] for row in selected
                ),
                "latency_stdev_ms": statistics.stdev(
                    row["end_to_end_latency_ms"] for row in selected
                ),
                "throughput_ops_s_median": statistics.median(
                    row["throughput_ops_s"] for row in selected
                ),
                "root_connection_count_median": statistics.median(
                    row["root_connection_count"] for row in selected
                ),
                "total_connection_count_median": statistics.median(
                    row["total_connection_count"] for row in selected
                ),
                "total_connection_ns_median": statistics.median(
                    row["total_connection_ns"] for row in selected
                ),
                "root_messages_median": statistics.median(
                    row["root_messages_total"] for row in selected
                ),
                "root_bytes_median": statistics.median(row["root_bytes_total"] for row in selected),
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
                "critical_path_compute_delay_ms_median": statistics.median(
                    row["critical_path_compute_delay_ms"] for row in selected
                ),
            }
        )
    return {
        "schema_version": "1.0",
        "hypothesis_id": HYPOTHESIS_ID,
        "evidence_classification": (
            "measured synthetic workload over independent heterogeneous processes"
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


def run_h012_009(
    *,
    output_directory: Path,
    profile_manifest_path: Path,
    payload_bytes: int = 256,
    warmup_trials: int = 1,
    measured_trials: int = 20,
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
    profile_source = {
        "path": str(profile_manifest_path),
        "sha256": _sha256_file(profile_manifest_path),
        "bytes": profile_manifest_path.stat().st_size,
    }
    _write_json(output_directory / "worker-profile-source.json", profile_source)
    shutil.copy2(profile_manifest_path, output_directory / "worker-profiles.json")
    profiles = _load_profiles(profile_manifest_path)

    rows: list[dict[str, Any]] = []
    pool = WorkerPool(
        count=WORKER_COUNT,
        directory=raw_directory / "workers-heterogeneous-0128" / "processes",
        startup_deadline_s=startup_deadline_s,
        protocol_script=protocol_script,
        runtime_profiles=profiles,
    )
    attempt: dict[str, Any] = {
        "status": "starting",
        "worker_count": WORKER_COUNT,
        "profile_manifest_sha256": profile_source["sha256"],
        "attempted_unix_ns": time.time_ns(),
    }
    started_ns = time.perf_counter_ns()
    runner: DelegatedRunner | None = None
    try:
        workers = pool.start()
        _write_json(
            output_directory / "worker-identities.json",
            {
                "workers": [
                    {
                        "worker_id": worker.worker_id,
                        "worker_index": worker.worker_index,
                        "process_id": worker.process_id,
                        "endpoint": worker.endpoint,
                        "runtime_profile": worker.runtime_profile,
                    }
                    for worker in workers
                ]
            },
        )
        topology = build_capacity_parent_topology(
            workers,
            branch_factor=BRANCH_FACTOR,
            topology_id="h012-009-capacity-parent-b8",
            route_lease_id="h012-009-lease-1",
            route_generation=1,
        )
        _write_json(output_directory / "topology.json", topology_record(topology))
        _write_json(output_directory / "topology-installation.json", install_topology(topology))
        runner = DelegatedRunner(
            topology=topology,
            output_directory=output_directory,
            payload_bytes=payload_bytes,
            operation_deadline_s=operation_deadline_s,
            maximum_root_concurrency=BRANCH_FACTOR,
            cycle_id=HYPOTHESIS_ID,
        )
        execution_generation = 1
        profile = NETWORK_PROFILES["same_host_shaped"]
        schedule: list[tuple[str, bool, int]] = [
            (mode, True, warmup_index) for warmup_index in range(warmup_trials) for mode in MODES
        ]
        for trial_index in range(measured_trials):
            ordered_modes = MODES if trial_index % 2 == 0 else tuple(reversed(MODES))
            schedule.extend((mode, False, trial_index) for mode in ordered_modes)
        for mode, warmup, trial_index in schedule:
            row = runner.run_trial(
                trial_index=trial_index,
                warmup=warmup,
                execution_generation=execution_generation,
                profile=profile,
                mode=mode,
            )
            execution_generation += 1
            row.update(
                {
                    "worker_condition": "heterogeneous_five_class",
                    "topology_policy": "capacity_parents",
                    "profile_manifest_sha256": profile_source["sha256"],
                }
            )
            rows.append(row)
            _append_jsonl(trials_path, row)
            if row["status"] != "ok":
                _append_jsonl(errors_path, row)
        attempt.update(
            {
                "status": "completed" if all(row["status"] == "ok" for row in rows) else "failed",
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
        _append_jsonl(errors_path, {"event": "run_failed", **attempt})
    finally:
        if runner is not None:
            runner.close()
        attempt["shutdown"] = pool.stop()
        attempt["elapsed_ms"] = (time.perf_counter_ns() - started_ns) / 1_000_000
        _write_json(raw_directory / "attempt.json", attempt)

    summary = _summarize(rows)
    summary["attempt"] = attempt
    _write_json(output_directory / "benchmark-summary.json", summary)
    _write_csv(output_directory / "benchmark-summary.csv", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Experiment 012 H012-009")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-manifest", type=Path, required=True)
    parser.add_argument("--payload-bytes", type=int, default=256)
    parser.add_argument("--warmup-trials", type=int, default=1)
    parser.add_argument("--measured-trials", type=int, default=20)
    parser.add_argument("--operation-deadline-s", type=float, default=30.0)
    parser.add_argument("--startup-deadline-s", type=float, default=180.0)
    args = parser.parse_args(argv)
    summary = run_h012_009(
        output_directory=args.output.resolve(),
        profile_manifest_path=args.profile_manifest.resolve(),
        payload_bytes=args.payload_bytes,
        warmup_trials=args.warmup_trials,
        measured_trials=args.measured_trials,
        operation_deadline_s=args.operation_deadline_s,
        startup_deadline_s=args.startup_deadline_s,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not summary["failed_trials"] and summary["attempt"]["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["HYPOTHESIS_ID", "MODES", "run_h012_009"]
