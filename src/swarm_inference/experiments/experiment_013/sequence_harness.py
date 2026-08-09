"""Measure long sequences through one live persistent subtree collective."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_012.baseline_harness import (
    WorkerPool,
    _percentile,
    _sha256_file,
    _source_identity,
    _write_json,
)
from swarm_inference.experiments.experiment_012.delegation_harness import (
    build_delegated_topology,
    topology_record,
)
from swarm_inference.experiments.experiment_013.persistent_harness import (
    PersistentCollectiveRunner,
    _install_collective,
    _worker_statuses,
)
from swarm_inference.experiments.experiment_013.persistent_protocol import (
    ARCHITECTURES,
    BUFFERED_ARCHITECTURE,
)
from swarm_inference.microworker_protocol import NETWORK_PROFILES, LinkProfile

HYPOTHESIS_ID = "H013-011"
SEQUENCE_LENGTHS = (1, 2, 8, 32, 128, 512)


def _current_process_rss_bytes(process_id: int) -> int | None:
    """Return current resident memory; the older harness intentionally sampled peak RSS."""
    if os.name != "nt":
        status = Path(f"/proc/{process_id}/status")
        if not status.exists():
            return None
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    handle = kernel32.OpenProcess(0x0400 | 0x0010, False, process_id)
    if not handle:
        return None
    try:
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), ctypes.sizeof(counters)):
            return None
        return int(counters.WorkingSetSize)
    finally:
        kernel32.CloseHandle(handle)


def _current_memory_snapshot(pool: WorkerPool) -> dict[str, int | float]:
    observations = [
        value
        for worker in pool.workers
        if (value := _current_process_rss_bytes(worker.launcher_process_id)) is not None
    ]
    return {
        "observed_workers": len(observations),
        "total_worker_rss_bytes": sum(observations),
        "maximum_worker_rss_bytes": max(observations, default=0),
        "median_worker_rss_bytes": statistics.median(observations) if observations else 0,
    }


def _baseline_by_count(path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["worker_count"]): row for row in payload["summaries"]}


def _prefix_summary(
    *,
    rows: list[dict[str, Any]],
    completion_ns: list[int],
    length: int,
    setup_ns: int,
    process_start_ns: int,
    baseline: dict[str, Any],
    memory: dict[str, int | float],
) -> dict[str, Any]:
    prefix = rows[:length]
    warm = prefix[1:]
    warm_latencies = [float(row["end_to_end_latency_ms"]) for row in warm]
    all_latencies = [float(row["end_to_end_latency_ms"]) for row in prefix]
    sequence_ns = completion_ns[length - 1]
    baseline_p50_ms = float(baseline["end_to_end_latency_p50_ms"])
    return {
        "sequence_length": length,
        "all_operations_correct": all(bool(row["correctness"]) for row in prefix),
        "collective_setup_ms": setup_ns / 1_000_000,
        "process_start_ms": process_start_ns / 1_000_000,
        "first_operation_latency_ms": all_latencies[0],
        "warm_operation_count": len(warm_latencies),
        "warm_p50_ms": statistics.median(warm_latencies) if warm_latencies else None,
        "warm_p95_ms": _percentile(warm_latencies, 95) if warm_latencies else None,
        "warm_p99_ms": _percentile(warm_latencies, 99) if warm_latencies else None,
        "sequence_duration_ms": sequence_ns / 1_000_000,
        "steady_sequence_throughput_ops_s": length * 1_000_000_000 / sequence_ns,
        "amortised_collective_ms_per_operation": (setup_ns + sequence_ns) / 1_000_000 / length,
        "amortised_full_cold_ms_per_operation": (process_start_ns + setup_ns + sequence_ns)
        / 1_000_000
        / length,
        "delegated_baseline_p50_ms": baseline_p50_ms,
        "persistent_faster_including_collective_setup": (
            setup_ns + sequence_ns < baseline_p50_ms * 1_000_000 * length
        ),
        "memory": memory,
        "tail_statistical_note": (
            "warm tail unavailable for length 1"
            if not warm_latencies
            else "empirical nearest-rank percentile"
        ),
    }


def run_sequence_benchmark(
    *,
    output_directory: Path,
    baseline_summary: Path,
    worker_counts: tuple[int, ...],
    sequence_lengths: tuple[int, ...],
    branch_factor: int,
    payload_bytes: int,
    profile: LinkProfile,
    operation_deadline_s: float,
    startup_deadline_s: float,
    mailbox_depth: int,
    hypothesis_id: str,
    architecture: str,
) -> dict[str, Any]:
    if sorted(set(sequence_lengths)) != list(sequence_lengths) or sequence_lengths[0] < 1:
        raise ValueError("sequence lengths must be positive, unique, and increasing")
    output_directory.mkdir(parents=True, exist_ok=True)
    hypothesis_path = output_directory / "hypothesis.json"
    if not hypothesis_path.exists():
        raise FileNotFoundError("sequence hypothesis must be preregistered")

    repo_root = Path(__file__).resolve().parents[4]
    protocol_script = Path(__file__).with_name("persistent_protocol.py").resolve()
    source_files = [
        Path(__file__).resolve(),
        protocol_script,
        Path(__file__).with_name("persistent_harness.py").resolve(),
        repo_root / "src" / "swarm_inference" / "microworker_protocol.py",
    ]
    raw_directory = output_directory / "raw"
    snapshot_directory = raw_directory / "source-snapshot"
    snapshot_directory.mkdir(parents=True, exist_ok=True)
    for source in source_files:
        shutil.copy2(source, snapshot_directory / source.name)
    _write_json(
        output_directory / "source-identity.json", _source_identity(repo_root, source_files)
    )
    _write_json(
        output_directory / "hypothesis-identity.json",
        {
            "path": str(hypothesis_path),
            "sha256": _sha256_file(hypothesis_path),
            "bytes": hypothesis_path.stat().st_size,
        },
    )

    baselines = _baseline_by_count(baseline_summary)
    scales: list[dict[str, Any]] = []
    max_length = sequence_lengths[-1]
    for worker_count in worker_counts:
        pool = WorkerPool(
            count=worker_count,
            directory=output_directory / "workers" / f"workers-{worker_count:04d}",
            startup_deadline_s=startup_deadline_s,
            protocol_script=protocol_script,
        )
        runner: PersistentCollectiveRunner | None = None
        scale: dict[str, Any] = {"worker_count": worker_count, "status": "failed"}
        try:
            process_started_ns = time.perf_counter_ns()
            workers = pool.start()
            process_start_ns = time.perf_counter_ns() - process_started_ns
            topology = build_delegated_topology(
                workers,
                branch_factor=branch_factor,
                topology_id=f"{hypothesis_id.lower()}-sequence-n{worker_count}",
                route_lease_id=f"{hypothesis_id.lower()}-sequence-lease-n{worker_count}",
            )
            _write_json(
                output_directory / "topologies" / f"workers-{worker_count:04d}.json",
                topology_record(topology),
            )
            collective_id = f"{hypothesis_id.lower()}-sequence-collective-n{worker_count}"
            setup_started_ns = time.perf_counter_ns()
            installation = _install_collective(
                topology,
                collective_id=collective_id,
                architecture=architecture,
                profile=profile,
                mailbox_depth=mailbox_depth,
            )
            runner = PersistentCollectiveRunner(
                topology=topology,
                collective_id=collective_id,
                output_directory=output_directory,
                payload_bytes=payload_bytes,
                operation_deadline_s=operation_deadline_s,
                cycle_id=hypothesis_id,
                architecture=architecture,
            )
            root_prepare = runner.prepare(profile)
            setup_ns = time.perf_counter_ns() - setup_started_ns
            memory_before = _current_memory_snapshot(pool)
            rows: list[dict[str, Any]] = []
            completion_ns: list[int] = []
            memory_by_length: dict[int, dict[str, int | float]] = {}
            sequence_started_ns = time.perf_counter_ns()
            for operation_index in range(max_length):
                row = runner.run_trial(
                    trial_index=operation_index,
                    warmup=operation_index == 0,
                    execution_generation=operation_index + 1,
                    profile=profile,
                )
                rows.append(row)
                completion_ns.append(time.perf_counter_ns() - sequence_started_ns)
                completed = operation_index + 1
                if completed in sequence_lengths:
                    memory_by_length[completed] = _current_memory_snapshot(pool)
            statuses = _worker_statuses(topology)
            prefixes = [
                _prefix_summary(
                    rows=rows,
                    completion_ns=completion_ns,
                    length=length,
                    setup_ns=setup_ns,
                    process_start_ns=process_start_ns,
                    baseline=baselines[worker_count],
                    memory=memory_by_length[length],
                )
                for length in sequence_lengths
            ]
            observed_break_even = next(
                (
                    row["sequence_length"]
                    for row in prefixes
                    if row["persistent_faster_including_collective_setup"]
                ),
                None,
            )
            warm_latencies = [float(row["end_to_end_latency_ms"]) for row in rows[1:]]
            baseline_p50_ms = float(baselines[worker_count]["end_to_end_latency_p50_ms"])
            denominator = baseline_p50_ms - statistics.median(warm_latencies)
            analytic_break_even = (
                math.ceil(setup_ns / 1_000_000 / denominator) if denominator > 0 else None
            )
            scale.update(
                {
                    "status": "completed",
                    "architecture": architecture,
                    "branch_factor": branch_factor,
                    "hierarchy_depth": topology.depth,
                    "root_degree": len(topology.root_children),
                    "root_leaf_rpc_count": 0,
                    "process_start_ms": process_start_ns / 1_000_000,
                    "collective_setup_ms": setup_ns / 1_000_000,
                    "installation": installation,
                    "root_prepare": root_prepare,
                    "operation_count": len(rows),
                    "all_operations_correct": all(bool(row["correctness"]) for row in rows),
                    "first_operation_latency_ms": rows[0]["end_to_end_latency_ms"],
                    "warm_p50_ms": statistics.median(warm_latencies),
                    "warm_p95_ms": _percentile(warm_latencies, 95),
                    "warm_p99_ms": _percentile(warm_latencies, 99),
                    "warm_dispersion_ms": {
                        "minimum": min(warm_latencies),
                        "maximum": max(warm_latencies),
                        "mean": statistics.fmean(warm_latencies),
                        "stddev": statistics.stdev(warm_latencies),
                    },
                    "observed_break_even_sequence_length": observed_break_even,
                    "analytic_break_even_sequence_length": analytic_break_even,
                    "memory_before": memory_before,
                    "memory_after": memory_by_length[max_length],
                    "memory_growth_bytes": int(
                        memory_by_length[max_length]["total_worker_rss_bytes"]
                    )
                    - int(memory_before["total_worker_rss_bytes"]),
                    "prefixes": prefixes,
                    "worker_statuses": statuses,
                    "operations": rows,
                }
            )
        except BaseException as error:
            scale.update({"error_type": type(error).__name__, "error": str(error)})
        finally:
            if runner is not None:
                scale["root_teardown"] = runner.close()
            scale["worker_teardown"] = pool.stop()
        _write_json(raw_directory / f"sequence-n{worker_count:04d}.json", scale)
        scales.append(scale)

    summary = {
        "experiment_id": "013",
        "hypothesis_id": hypothesis_id,
        "run_kind": "persistent_sequence_prefix",
        "architecture": architecture,
        "sequence_lengths": list(sequence_lengths),
        "same_collective_for_entire_sequence": True,
        "all_scales_completed": all(scale["status"] == "completed" for scale in scales),
        "all_operations_correct": all(
            bool(scale.get("all_operations_correct")) for scale in scales
        ),
        "scales": [
            {
                key: value
                for key, value in scale.items()
                if key not in {"installation", "operations", "worker_statuses"}
            }
            for scale in scales
        ],
        "baseline_summary": str(baseline_summary),
        "environment": _source_identity(repo_root, source_files),
    }
    _write_json(output_directory / "sequence-summary.json", summary)
    return summary


def _parse_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--counts", type=_parse_ints, default=(73, 1000))
    parser.add_argument("--sequence-lengths", type=_parse_ints, default=SEQUENCE_LENGTHS)
    parser.add_argument("--branch-factor", type=int, default=8)
    parser.add_argument("--payload-bytes", type=int, default=256)
    parser.add_argument("--profile", choices=tuple(NETWORK_PROFILES), default="same_host_shaped")
    parser.add_argument("--operation-deadline-s", type=float, default=30.0)
    parser.add_argument("--startup-deadline-s", type=float, default=180.0)
    parser.add_argument("--mailbox-depth", type=int, default=2)
    parser.add_argument("--hypothesis-id", default=HYPOTHESIS_ID)
    parser.add_argument("--architecture", choices=ARCHITECTURES, default=BUFFERED_ARCHITECTURE)
    args = parser.parse_args(argv)
    result = run_sequence_benchmark(
        output_directory=args.output.resolve(),
        baseline_summary=args.baseline_summary.resolve(),
        worker_counts=args.counts,
        sequence_lengths=args.sequence_lengths,
        branch_factor=args.branch_factor,
        payload_bytes=args.payload_bytes,
        profile=NETWORK_PROFILES[args.profile],
        operation_deadline_s=args.operation_deadline_s,
        startup_deadline_s=args.startup_deadline_s,
        mailbox_depth=args.mailbox_depth,
        hypothesis_id=args.hypothesis_id,
        architecture=args.architecture,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_scales_completed"] and result["all_operations_correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
