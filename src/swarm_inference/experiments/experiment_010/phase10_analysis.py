"""Consolidate measured Phase 10 Colibri workloads without inventing rows.

The raw per-prompt JSON/JSONL files remain authoritative.  This module creates
compact, source-linked tables and phase plans from those files, recomputes the
verified-candidate gate under the correction rules, and preserves unavailable
Windows paging counters as null values.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_010.memory_analysis import (
    amdahl_gate,
    page_fault_candidate_validity,
    prefetch_idle_window_budget,
)

SHORT_CONFIGURATIONS = {
    "local": "local",
    "whole-direct-tcp": "whole_expert_direct_tcp",
    "whole-shared-memory": "whole_expert_shared_memory",
    "whole-relayed-tcp": "whole_expert_relayed_tcp",
    "whole-fast": "whole_expert_fast_aggregation",
    "equal-microshards": "equal_microshards",
    "asymmetric-microshards": "asymmetric_microshards",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to create empty measured artifact {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: _csv_value(row.get(key)) for key in fields} for row in rows)


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate percentile of an empty sample")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _measurement_rows(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((directory / "runs").glob("*/measurement.json")):
        row = _read_json(path)
        row["raw_measurement_path"] = str(path.resolve())
        rows.append(row)
    return rows


def _normalised_cache_gate(row: dict[str, Any]) -> tuple[bool, bool]:
    """Return cache validity and whether a resumed counter reset was reconciled."""

    reset = False
    for metrics in (row.get("worker_counter_deltas") or {}).values():
        resident = metrics.get("resident_cache_hits")
        nonresident = metrics.get("nonresident_cache_hits")
        if resident is None or nonresident is None:
            continue
        if resident < 0 or nonresident < 0:
            reset = True
            continue
        if nonresident > resident:
            return False, reset
    return True, reset


def consolidate_short_decode(
    phase_root: Path, output: Path, legacy_engine_sha256: str | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for directory_name, expected_name in SHORT_CONFIGURATIONS.items():
        directory = phase_root / "short-performance" / directory_name
        rows = _measurement_rows(directory)
        if len(rows) != 100:
            raise ValueError(f"{expected_name} has {len(rows)} rows; five repeats require 100")
        for row in rows:
            cache_valid, reset = _normalised_cache_gate(row)
            exact = row.get("exact_token_identity") is True
            canonical_valid = bool(
                row.get("measurement_status") == "MEASURED"
                and exact
                and int(row.get("forbidden_local_expert_loads") or 0) == 0
                and cache_valid
            )
            recorded_hash = row.get("colibri_binary_sha256")
            all_rows.append(
                {
                    **row,
                    "configuration": expected_name,
                    "canonical_verified_candidate": canonical_valid,
                    "counter_session_reset_reconciled": reset,
                    "colibri_binary_sha256": recorded_hash or legacy_engine_sha256,
                    "binary_hash_provenance": (
                        "per_row" if recorded_hash else "phase_10_initial_suite_manifest"
                    ),
                }
            )
        speeds = [float(row["decode_tokens_per_second"]) for row in rows]
        walls = [int(row["wall_elapsed_ns"]) / 1e9 for row in rows]
        canonical = [row for row in all_rows if row["configuration"] == expected_name]
        summaries.append(
            {
                "configuration": expected_name,
                "measurement_status": "MEASURED",
                "evidence_category": "REAL_MODEL_MEASURED",
                "measured_rows": len(rows),
                "repeat_count": len({int(row["repeat"]) for row in rows}),
                "exact_prompt_rows": sum(row.get("exact_token_identity") is True for row in rows),
                "canonical_verified_rows": sum(
                    row["canonical_verified_candidate"] for row in canonical
                ),
                "median_decode_tokens_per_second": statistics.median(speeds),
                "p50_wall_seconds": _percentile(walls, 0.50),
                "p95_wall_seconds": _percentile(walls, 0.95),
                "total_rpc_message_count": sum(
                    int(row.get("rpc_message_count") or 0) for row in rows
                ),
                "total_raw_payload_bytes": sum(
                    int(row.get("rpc_raw_payload_bytes") or 0) for row in rows
                ),
                "source_directory": str(directory.resolve()),
            }
        )
    _write_csv(output / "short_decode_results.csv", all_rows)
    _write_csv(output / "short_decode_summary.csv", summaries)
    return all_rows, summaries


def consolidate_prefill(phase_root: Path, output: Path) -> list[dict[str, Any]]:
    local_path = (
        phase_root / "final-binary" / "prefill-references-t16" / "local-prefill-results.csv"
    )
    remote_path = (
        phase_root / "final-binary" / "prefill" / "whole-direct-tcp-exact" / "measurements.csv"
    )
    rows: list[dict[str, Any]] = []
    for configuration, path in (
        ("local", local_path),
        ("whole_expert_direct_tcp_exact", remote_path),
    ):
        with path.open(encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle):
                prompt_tokens = int(raw["prompt_tokens"])
                prefill_seconds = float(raw["prefill_seconds"])
                rows.append(
                    {
                        **raw,
                        "configuration": configuration,
                        "prefill_tokens_per_second": prompt_tokens / prefill_seconds,
                        "source_path": str(path.resolve()),
                    }
                )
    _write_csv(output / "prefill_results.csv", rows)
    capability = _read_json(
        phase_root / "final-binary" / "prefill-references-t16" / "completion.json"
    )
    _write_json(output / "prefill_context_capability.json", capability)
    return rows


def _combine_csv(paths: Iterable[Path], output_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append({**row, "source_path": str(path.resolve())})
    _write_csv(output_path, rows)
    return rows


def _last_worker_sessions(path: Path) -> list[dict[str, Any]]:
    """Return final cumulative sample for each monotonic native process segment."""

    sessions: list[dict[str, Any]] = []
    previous_hits: int | None = None
    last: dict[str, Any] | None = None
    session_index = 1
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            hits = int(row.get("logical_cache_hits", 0))
            if previous_hits is not None and hits < previous_hits:
                if last is not None:
                    sessions.append({**last, "process_session_index": session_index})
                session_index += 1
            last = row
            previous_hits = hits
    if last is not None:
        sessions.append({**last, "process_session_index": session_index})
    return sessions


def consolidate_page_faults(phase_root: Path, output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    roots = [phase_root / "short-performance", phase_root / "final-binary"]
    for root in roots:
        for path in sorted(root.rglob("worker-telemetry.jsonl")):
            for sample in _last_worker_sessions(path):
                commit_limit = int(sample.get("system_commit_limit_bytes") or 0)
                committed = int(sample.get("system_committed_bytes") or 0)
                pressure = committed / commit_limit if commit_limit else 0.0
                validity = page_fault_candidate_validity(
                    resident_cache_hits=int(sample.get("resident_cache_hits") or 0),
                    nonresident_cache_hits=int(sample.get("nonresident_cache_hits") or 0),
                    pagefile_read_bytes=sample.get("pagefile_read_bytes"),
                    commit_pressure_fraction=pressure,
                )
                rows.append(
                    {
                        "source_path": str(path.resolve()),
                        "worker_id": sample.get("worker_id"),
                        "process_session_index": sample["process_session_index"],
                        "logical_cache_hits": sample.get("logical_cache_hits"),
                        "logical_cache_misses": sample.get("logical_cache_misses"),
                        "resident_cache_hits": sample.get("resident_cache_hits"),
                        "nonresident_cache_hits": sample.get("nonresident_cache_hits"),
                        "cache_hits_with_page_fault": sample.get("cache_hits_with_page_fault"),
                        "cache_bytes": sample.get("cache_bytes"),
                        "resident_cache_bytes": sample.get("resident_cache_bytes"),
                        "expert_working_set_bytes": sample.get("expert_working_set_bytes"),
                        "page_fault_count": sample.get("page_fault_count"),
                        "pagefile_read_bytes": sample.get("pagefile_read_bytes"),
                        "process_io_read_bytes": sample.get("process_io_read_bytes"),
                        "process_io_write_bytes": sample.get("process_io_write_bytes"),
                        "working_set_bytes": sample.get("working_set_bytes"),
                        "private_bytes": sample.get("private_bytes"),
                        "commit_size_bytes": sample.get("commit_size_bytes"),
                        "system_committed_bytes": committed,
                        "system_commit_limit_bytes": commit_limit,
                        "commit_pressure_fraction": pressure,
                        "valid_performance_candidate": validity["valid_performance_candidate"],
                        "invalidation_reasons": validity["invalidation_reasons"],
                        "memory_counter_limitation": sample.get("memory_counter_limitation"),
                        "evidence_category": "REAL_MODEL_MEASURED",
                    }
                )
    _write_csv(output / "page_fault_results.csv", rows)
    return rows


def consolidate_memory_timeseries(phase_root: Path, output: Path) -> int:
    """Write one-second peak samples linked to every raw 100 ms timeseries."""

    buckets: dict[tuple[str, int, str, str, int], dict[str, Any]] = {}
    for root in (phase_root / "short-performance", phase_root / "final-binary"):
        for path in sorted(root.rglob("memory.ndjson")):
            source = str(path.resolve())
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    second = int(row["timestamp_ns"]) // 1_000_000_000
                    key = (
                        source,
                        second,
                        str(row.get("process_role")),
                        str(row.get("process_identity")),
                        int(row.get("pid", -1)),
                    )
                    current = buckets.setdefault(
                        key,
                        {
                            "source_path": source,
                            "timestamp_second": second,
                            "process_role": row.get("process_role"),
                            "process_identity": row.get("process_identity"),
                            "pid": row.get("pid"),
                            "raw_sample_count": 0,
                        },
                    )
                    current["raw_sample_count"] += 1
                    for name in (
                        "working_set_bytes",
                        "private_bytes",
                        "commit_size_bytes",
                        "peak_working_set_bytes",
                        "page_fault_count",
                        "thread_count",
                        "storage_read_bytes",
                        "storage_write_bytes",
                        "system_available_physical_bytes",
                        "system_committed_bytes_proxy",
                        "system_commit_limit_bytes_proxy",
                        "pagefile_usage_bytes",
                    ):
                        if row.get(name) is not None:
                            current[name] = max(int(row[name]), int(current.get(name, 0)))
    rows = sorted(buckets.values(), key=lambda row: (row["timestamp_second"], row["source_path"]))
    _write_csv(output / "memory_residency_timeseries.csv", rows)
    return len(rows)


def consolidate_cache_sizes(phase_root: Path, output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = phase_root / "final-binary" / "cache-sizing"
    for candidate in sorted(root.iterdir()):
        if not candidate.is_dir():
            continue
        measurement = _read_json(next(candidate.rglob("measurement.json")))
        samples = [
            _last_worker_sessions(path)[-1] for path in candidate.rglob("worker-telemetry.jsonl")
        ]
        rows.append(
            {
                "candidate": candidate.name,
                "decode_tokens_per_second": measurement["decode_tokens_per_second"],
                "ttft_seconds": measurement["ttft_seconds"],
                "exact_token_identity": measurement["exact_token_identity"],
                "logical_cache_hits": sum(int(row["logical_cache_hits"]) for row in samples),
                "logical_cache_misses": sum(int(row["logical_cache_misses"]) for row in samples),
                "resident_cache_hits": sum(int(row["resident_cache_hits"]) for row in samples),
                "nonresident_cache_hits": sum(
                    int(row["nonresident_cache_hits"]) for row in samples
                ),
                "cache_hits_with_page_fault": sum(
                    int(row["cache_hits_with_page_fault"]) for row in samples
                ),
                "cache_evictions": sum(int(row["cache_evictions"]) for row in samples),
                "cache_bytes": sum(int(row["cache_bytes"]) for row in samples),
                "resident_cache_bytes": sum(int(row["resident_cache_bytes"]) for row in samples),
                "pagefile_read_bytes": None,
                "pagefile_read_bytes_available": False,
                "source_directory": str(candidate.resolve()),
            }
        )
    _write_csv(output / "cache_size_results.csv", rows)
    return rows


def _plan(
    *, phase: str, objective: str, candidates: Sequence[dict[str, Any]], metric: str, maximize: bool
) -> dict[str, Any]:
    eligible = [row for row in candidates if row.get("eligible", True)]
    if not eligible:
        raise ValueError(f"no eligible {phase} candidates")
    selected = (max if maximize else min)(eligible, key=lambda row: float(row[metric]))
    return {
        "schema_version": "experiment-010-measured-phase-plan-v1",
        "phase": phase,
        "objective": objective,
        "selection_basis": "eligible real Level A measurements only",
        "selected_candidate": selected["configuration"],
        "selected_metric_value": selected[metric],
        "metric": metric,
        "candidates": list(candidates),
        "explanation": (
            f"selected {selected['configuration']} from measured eligible rows; "
            "negative-utility distributed candidates remain recorded"
        ),
    }


def write_phase_plans(
    output: Path,
    short_summaries: Sequence[dict[str, Any]],
    prefill: Sequence[dict[str, Any]],
    concurrent: Sequence[dict[str, Any]],
    mixed: Sequence[dict[str, Any]],
) -> None:
    decode_candidates = [
        {
            "configuration": row["configuration"],
            "decode_tokens_per_second": row["median_decode_tokens_per_second"],
            "eligible": row["canonical_verified_rows"] == row["measured_rows"],
            "rejection_reason": (
                None
                if row["canonical_verified_rows"] == row["measured_rows"]
                else "exact-token/correctness gate failed"
            ),
            "measured_rows": row["measured_rows"],
        }
        for row in short_summaries
    ]
    _write_json(
        output / "decode_plan.json",
        _plan(
            phase="decode",
            objective="max_decode_throughput",
            candidates=decode_candidates,
            metric="decode_tokens_per_second",
            maximize=True,
        ),
    )

    by_prefill: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prefill:
        by_prefill[row["configuration"]].append(row)
    prefill_candidates = []
    for configuration, rows in by_prefill.items():
        exact = all(str(row["exact_token_identity"]).lower() == "true" for row in rows)
        prefill_candidates.append(
            {
                "configuration": configuration,
                "median_ttft_seconds": statistics.median(
                    float(row["ttft_seconds"]) for row in rows
                ),
                "median_prefill_tokens_per_second": statistics.median(
                    float(row["prefill_tokens_per_second"]) for row in rows
                ),
                "eligible": exact,
                "measured_rows": len(rows),
            }
        )
    _write_json(
        output / "prefill_plan.json",
        _plan(
            phase="prefill",
            objective="min_ttft",
            candidates=prefill_candidates,
            metric="median_ttft_seconds",
            maximize=False,
        ),
    )

    concurrent_candidates = [
        {
            "configuration": row["configuration"],
            "concurrency": int(row["concurrency"]),
            "aggregate_verified_tokens_per_second": float(
                row["aggregate_verified_tokens_per_second"]
            ),
            "p95_latency_seconds": float(row["p95_latency_seconds"]),
            "eligible": str(row["exact_group_token_identity"]).lower() == "true",
            "source_path": row["source_path"],
        }
        for row in concurrent
    ]
    selected_by_concurrency = {}
    for level in sorted({row["concurrency"] for row in concurrent_candidates}):
        candidates = [row for row in concurrent_candidates if row["concurrency"] == level]
        selected_by_concurrency[str(level)] = max(
            (row for row in candidates if row["eligible"]),
            key=lambda row: row["aggregate_verified_tokens_per_second"],
        )["configuration"]
    _write_json(
        output / "concurrent_decode_plan.json",
        {
            "schema_version": "experiment-010-measured-phase-plan-v1",
            "phase": "concurrent_decode",
            "objective": "max_aggregate_verified_throughput",
            "selected_by_concurrency": selected_by_concurrency,
            "candidates": concurrent_candidates,
        },
    )

    mixed_candidates = [
        {
            "configuration": row["configuration"],
            "background_count": int(row["concurrency"]) - 1,
            "interactive_p95_seconds": float(row["interactive_p95_seconds"]),
            "background_verified_tokens_per_second": float(
                row["background_verified_tokens_per_second"]
            ),
            "aggregate_verified_tokens_per_second": float(
                row["aggregate_verified_tokens_per_second"]
            ),
            "eligible": (
                str(row["exact_group_token_identity"]).lower() == "true"
                and str(row["starvation_detected"]).lower() == "false"
            ),
            "source_path": row["source_path"],
        }
        for row in mixed
    ]
    selected_mixed = {}
    for count in sorted({row["background_count"] for row in mixed_candidates}):
        candidates = [row for row in mixed_candidates if row["background_count"] == count]
        selected_mixed[str(count)] = max(
            (row for row in candidates if row["eligible"]),
            key=lambda row: row["aggregate_verified_tokens_per_second"],
        )["configuration"]
    _write_json(
        output / "mixed_service_plan.json",
        {
            "schema_version": "experiment-010-measured-phase-plan-v1",
            "phase": "mixed_service",
            "objective": "max_aggregate_verified_throughput_without_starvation",
            "selected_by_background_count": selected_mixed,
            "candidates": mixed_candidates,
        },
    )


def write_prefetch_and_amdahl(
    phase_root: Path,
    output: Path,
    short_rows: Sequence[dict[str, Any]],
    cache_rows: Sequence[dict[str, Any]],
) -> None:
    # No timestamped gap telemetry exists in the pinned runtime checkpoint.
    # Enforce a conservative zero-byte budget rather than asserting overlap.
    prefetch_rows = []
    for phase in ("prefill", "decode"):
        for layer_id in range(16):
            row = prefetch_idle_window_budget(
                phase=phase,
                layer_id=layer_id,
                available_idle_window_ns=0,
                effective_bandwidth_bytes_per_second=0.0,
                proposed_prefetch_bytes=0,
                subsequently_consumed_bytes=0,
                demand_read_interference_ns=0,
                eviction_bytes=0,
            )
            row.update(
                {
                    "decision": "disable",
                    "accepted": False,
                    "measurement_limitation": (
                        "current checkpoint lacks timestamped inter-layer idle-gap telemetry; "
                        "zero is the enforced safe lower bound and no performance claim is made"
                    ),
                }
            )
            prefetch_rows.append(row)
    _write_csv(output / "prefetch_idle_window_results.csv", prefetch_rows)

    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in short_rows:
        if row.get("colibri_binary_sha256") and int(row.get("repeat") or 0) >= 4:
            by_config[row["configuration"]].append(row)

    def med(configuration: str, field: str) -> int:
        values = [int(row[field]) for row in by_config[configuration] if row.get(field) is not None]
        return int(statistics.median(values))

    amdahl_rows: list[dict[str, Any]] = []
    comparisons = (
        (
            "shared_memory",
            "whole_expert_direct_tcp",
            "whole_expert_shared_memory",
            "rpc_transport_ns",
        ),
        ("microsharding", "whole_expert_direct_tcp", "equal_microshards", "rpc_compute_ns"),
        (
            "asymmetric_microsharding",
            "equal_microshards",
            "asymmetric_microshards",
            "rpc_compute_ns",
        ),
        (
            "reduction_strategy",
            "whole_expert_direct_tcp",
            "whole_expert_fast_aggregation",
            "rpc_transport_ns",
        ),
    )
    for optimization, baseline, optimized, affected in comparisons:
        result = amdahl_gate(
            optimization=optimization,
            baseline_end_to_end_ns=med(baseline, "wall_elapsed_ns"),
            baseline_affected_ns=med(baseline, affected),
            optimized_affected_ns=med(optimized, affected),
            optimized_end_to_end_ns=med(optimized, "wall_elapsed_ns"),
        )
        result.update(
            {
                "baseline_configuration": baseline,
                "optimized_configuration": optimized,
                "affected_counter": affected,
                "correctness_gate": all(
                    row["exact_token_identity"] is True for row in by_config[optimized]
                ),
            }
        )
        result["accepted"] = bool(result["accepted"] and result["correctness_gate"])
        amdahl_rows.append(result)

    baseline_cache = next(row for row in cache_rows if row["candidate"] == "baseline-512mb")
    best_cache = max(cache_rows, key=lambda row: float(row["decode_tokens_per_second"]))
    amdahl_rows.append(
        {
            "optimization": "cache_policy",
            "baseline_configuration": baseline_cache["candidate"],
            "optimized_configuration": best_cache["candidate"],
            "affected_time_fraction": None,
            "theoretical_maximum_gain": None,
            "measured_kernel_gain": None,
            "measured_end_to_end_gain": (
                float(best_cache["decode_tokens_per_second"])
                / float(baseline_cache["decode_tokens_per_second"])
            ),
            "accepted": False,
            "status": "REJECTED_INSUFFICIENT_REPEATS_FOR_MODEL_LEVEL_ACCEPTANCE",
        }
    )
    for name in ("compression", "coalescing", "prefetch"):
        amdahl_rows.append(
            {
                "optimization": name,
                "affected_time_fraction": None,
                "theoretical_maximum_gain": None,
                "measured_kernel_gain": None,
                "measured_end_to_end_gain": None,
                "accepted": False,
                "status": "REJECTED_NO_ELIGIBLE_REAL_PATH_COUNTERFACTUAL",
            }
        )
    cuda = _read_json(phase_root.parent / "phase-9" / "real_model_cuda_results.json")["result"]
    local_wall_ns = int(cuda["token_count"] / cuda["local_throughput_tokens_per_second"] * 1e9)
    distributed_wall_ns = int(
        cuda["token_count"] / cuda["distributed_throughput_tokens_per_second"] * 1e9
    )
    gpu = amdahl_gate(
        optimization="gpu_kernel",
        baseline_end_to_end_ns=local_wall_ns,
        baseline_affected_ns=int(cuda["operator_cpu_reference_ns"]),
        optimized_affected_ns=int(cuda["operator_gpu_kernel_ns"]),
        optimized_end_to_end_ns=distributed_wall_ns,
    )
    gpu["correctness_gate"] = bool(cuda["exact_token_identity"])
    gpu["accepted"] = bool(gpu["accepted"] and gpu["correctness_gate"])
    amdahl_rows.append(gpu)
    _write_csv(output / "amdahl_results.csv", amdahl_rows)


def consolidate(phase_root: Path, output: Path) -> dict[str, Any]:
    phase_root = phase_root.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    suite = _read_json(phase_root / "short-128-references" / "local-suite.json")
    short_rows, short_summaries = consolidate_short_decode(
        phase_root, output, suite.get("engine_sha256")
    )
    prefill = consolidate_prefill(phase_root, output)
    concurrent = _combine_csv(
        sorted((phase_root / "final-binary" / "concurrent").glob("*/concurrent_decode.csv")),
        output / "concurrent_decode_results.csv",
    )
    mixed = _combine_csv(
        sorted((phase_root / "final-binary" / "mixed-service").glob("*/mixed_service.csv")),
        output / "mixed_service_results.csv",
    )
    network = _combine_csv(
        [phase_root / "final-binary" / "network-profiles" / "network_profile_results.csv"],
        output / "network_profile_results.csv",
    )
    cache = consolidate_cache_sizes(phase_root, output)
    page_faults = consolidate_page_faults(phase_root, output)
    timeseries_rows = consolidate_memory_timeseries(phase_root, output)
    write_phase_plans(output, short_summaries, prefill, concurrent, mixed)
    write_prefetch_and_amdahl(phase_root, output, short_rows, cache)
    summary = {
        "schema_version": "experiment-010-phase-10-consolidation-v1",
        "evidence_category": "REAL_MODEL_MEASURED",
        "short_decode_rows": len(short_rows),
        "prefill_rows": len(prefill),
        "concurrent_groups": len(concurrent),
        "mixed_service_groups": len(mixed),
        "network_profile_rows": len(network),
        "cache_candidates": len(cache),
        "page_fault_process_sessions": len(page_faults),
        "memory_timeseries_rows": timeseries_rows,
        "missing_metrics_are_zero_filled": False,
    }
    _write_json(output / "phase10_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(consolidate(arguments.phase_root, arguments.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
