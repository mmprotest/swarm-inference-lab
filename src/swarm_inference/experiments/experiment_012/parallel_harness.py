"""H012-002 bounded parallel-child dispatch experiment harness."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_012.baseline_harness import (
    WorkerPool,
    _append_jsonl,
    _sha256_file,
    _source_identity,
    _write_json,
)
from swarm_inference.experiments.experiment_012.delegation_harness import (
    DELEGATED_MODES,
    DelegatedRunner,
    _summarize,
    _write_summary_csv,
    build_delegated_topology,
    install_topology,
    topology_record,
)
from swarm_inference.microworker_protocol import NETWORK_PROFILES

HYPOTHESIS_ID = "H012-002"
MODES = ("delegated_serial", "delegated_parallel")
DISCRIMINATING_COUNTS = (32, 128)
DISCRIMINATING_PROFILES = ("same_host_shaped", "moderate_wan_shaped")


def run_h012_002(
    *,
    output_directory: Path,
    worker_counts: tuple[int, ...],
    profiles: tuple[str, ...],
    branch_factor: int,
    maximum_root_concurrency: int,
    payload_bytes: int,
    warmup_trials: int,
    measured_trials: int,
    operation_deadline_s: float,
    startup_deadline_s: float,
    modes: tuple[str, ...] = MODES,
    hypothesis_id: str = HYPOTHESIS_ID,
) -> dict[str, Any]:
    if not modes or set(modes) - set(DELEGATED_MODES):
        raise ValueError(f"unsupported delegated modes: {modes!r}")
    repo_root = Path(__file__).resolve().parents[4]
    protocol_script = repo_root / "src" / "swarm_inference" / "microworker_protocol.py"
    delegation_harness = Path(__file__).with_name("delegation_harness.py")
    source_files = [protocol_script, delegation_harness, Path(__file__).resolve()]
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
        scale_directory = raw_directory / f"workers-{worker_count:04d}"
        pool = WorkerPool(
            count=worker_count,
            directory=scale_directory / "processes",
            startup_deadline_s=startup_deadline_s,
            protocol_script=protocol_script,
        )
        attempt: dict[str, Any] = {
            "worker_count": worker_count,
            "status": "starting",
            "attempted_unix_ns": time.time_ns(),
        }
        runner: DelegatedRunner | None = None
        try:
            workers = pool.start()
            topology = build_delegated_topology(
                workers,
                branch_factor=branch_factor,
                topology_id=f"{hypothesis_id.lower()}-topology-n{worker_count}",
                route_lease_id=f"{hypothesis_id.lower()}-lease-n{worker_count}",
            )
            _write_json(
                output_directory / "topologies" / f"workers-{worker_count:04d}.json",
                topology_record(topology),
            )
            installation = install_topology(topology)
            _write_json(
                output_directory / "topology-installation" / f"workers-{worker_count:04d}.json",
                installation,
            )
            runner = DelegatedRunner(
                topology=topology,
                output_directory=output_directory,
                payload_bytes=payload_bytes,
                operation_deadline_s=operation_deadline_s,
                maximum_root_concurrency=maximum_root_concurrency,
                cycle_id=hypothesis_id,
            )
            for profile_name in profiles:
                profile = NETWORK_PROFILES[profile_name]
                for warmup_index in range(warmup_trials):
                    for mode in modes:
                        row = runner.run_trial(
                            trial_index=warmup_index,
                            warmup=True,
                            execution_generation=execution_generation,
                            profile=profile,
                            mode=mode,
                        )
                        execution_generation += 1
                        rows.append(row)
                        _append_jsonl(trials_path, row)
                        if row["status"] != "ok":
                            _append_jsonl(errors_path, row)
                for trial_index in range(measured_trials):
                    for mode in modes:
                        row = runner.run_trial(
                            trial_index=trial_index,
                            warmup=False,
                            execution_generation=execution_generation,
                            profile=profile,
                            mode=mode,
                        )
                        execution_generation += 1
                        rows.append(row)
                        _append_jsonl(trials_path, row)
                        if row["status"] != "ok":
                            _append_jsonl(errors_path, row)
            attempt.update(
                {
                    "status": "completed",
                    "started_processes": len(workers),
                    "peak_worker_rss_bytes": pool.sample_memory(),
                    "topology_depth": topology.depth,
                    "root_direct_degree": len(topology.root_children),
                    "root_leaf_rpc_count": sum(
                        not node.children for node in topology.root_children
                    ),
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
            attempt["elapsed_ms"] = (time.perf_counter_ns() - scale_started_ns) / 1_000_000
            attempts.append(attempt)
            _write_json(raw_directory / "scale-attempts.json", attempts)
    summary = _summarize(rows, attempts, hypothesis_id=hypothesis_id)
    _write_json(output_directory / "benchmark-summary.json", summary)
    _write_summary_csv(output_directory / "benchmark-summary.csv", summary)
    return summary


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(item < 2 for item in parsed):
        raise argparse.ArgumentTypeError("counts must contain integers of at least two")
    return parsed


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(parsed) - NETWORK_PROFILES.keys()
    if not parsed or unknown:
        raise argparse.ArgumentTypeError(f"unknown network profiles: {sorted(unknown)}")
    return parsed


def _parse_modes(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(parsed) - set(DELEGATED_MODES)
    if not parsed or unknown:
        raise argparse.ArgumentTypeError(f"unknown delegated modes: {sorted(unknown)}")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Experiment 012 H012-002")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counts", type=_parse_csv_ints, default=DISCRIMINATING_COUNTS)
    parser.add_argument("--profiles", type=_parse_csv_strings, default=DISCRIMINATING_PROFILES)
    parser.add_argument("--branch-factor", type=int, default=8)
    parser.add_argument("--maximum-root-concurrency", type=int, default=8)
    parser.add_argument("--payload-bytes", type=int, default=256)
    parser.add_argument("--modes", type=_parse_modes, default=MODES)
    parser.add_argument("--hypothesis-id", default=HYPOTHESIS_ID)
    parser.add_argument("--warmup-trials", type=int, default=1)
    parser.add_argument("--measured-trials", type=int, default=5)
    parser.add_argument("--operation-deadline-s", type=float, default=30.0)
    parser.add_argument("--startup-deadline-s", type=float, default=120.0)
    args = parser.parse_args(argv)
    summary = run_h012_002(
        output_directory=args.output.resolve(),
        worker_counts=args.counts,
        profiles=args.profiles,
        branch_factor=args.branch_factor,
        maximum_root_concurrency=args.maximum_root_concurrency,
        payload_bytes=args.payload_bytes,
        warmup_trials=args.warmup_trials,
        measured_trials=args.measured_trials,
        operation_deadline_s=args.operation_deadline_s,
        startup_deadline_s=args.startup_deadline_s,
        modes=args.modes,
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


__all__ = ["DISCRIMINATING_COUNTS", "HYPOTHESIS_ID", "MODES", "run_h012_002"]
