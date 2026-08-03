"""Trace-derived cache, paging, prefetch, and Amdahl analysis for Experiment 010."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


def route_accesses(paths: Sequence[Path]) -> list[tuple[int, int]]:
    """Read Colibri route traces in emitted token/layer/rank order."""

    accesses: list[tuple[int, int]] = []
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            fields = line.split()
            if len(fields) < 4:
                raise ValueError(f"malformed route trace {path}:{number}")
            layer_id = int(fields[1])
            for selected in fields[3:]:
                expert, separator, _weight = selected.partition(":")
                if not separator:
                    raise ValueError(f"malformed selected expert {path}:{number}")
                accesses.append((layer_id, int(expert)))
    return accesses


def reuse_distances(accesses: Iterable[tuple[int, int]]) -> list[int | None]:
    """Return per-layer LRU stack distance; ``None`` denotes a cold access."""

    stacks: dict[int, list[int]] = {}
    result: list[int | None] = []
    for layer_id, expert_id in accesses:
        stack = stacks.setdefault(layer_id, [])
        try:
            distance = stack.index(expert_id)
        except ValueError:
            distance = None
        result.append(distance)
        if distance is not None:
            stack.pop(distance)
        stack.insert(0, expert_id)
    return result


def _nearest_rank(values: Sequence[int], fraction: float) -> int:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def reuse_distance_curve(
    trace_paths: Sequence[Path], *, expert_bytes: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Choose cache candidates around measured reuse-distance thresholds."""

    if expert_bytes <= 0:
        raise ValueError("expert byte size must be positive")
    accesses = route_accesses(trace_paths)
    distances = reuse_distances(accesses)
    finite = [value for value in distances if value is not None]
    if not finite:
        raise ValueError("routing trace has no repeated expert access")
    thresholds = {
        label: _nearest_rank(finite, fraction)
        for label, fraction in (("p50", 0.50), ("p75", 0.75), ("p90", 0.90), ("p95", 0.95))
    }
    candidate_slots = sorted(
        {
            max(1, min(64, threshold + offset))
            for threshold in thresholds.values()
            for offset in (-1, 0, 1)
        }
        | {64}
    )
    rows: list[dict[str, Any]] = []
    for slots in candidate_slots:
        hits = sum(distance is not None and distance < slots for distance in distances)
        rows.append(
            {
                "cache_slots_per_layer": slots,
                "cache_bytes_per_layer": slots * expert_bytes,
                "cache_bytes_all_layers": slots * expert_bytes * 16,
                "access_count": len(distances),
                "cold_access_count": sum(distance is None for distance in distances),
                "logical_cache_hit_count": hits,
                "logical_cache_hit_fraction": hits / len(distances),
                "candidate_basis": "measured_reuse_distance_threshold_neighbourhood",
                "evidence_category": "REAL_MODEL_MEASURED",
            }
        )
    summary = {
        "schema_version": "experiment-010-reuse-distance-v1",
        "trace_paths": [str(path.resolve()) for path in trace_paths],
        "access_count": len(distances),
        "cold_access_count": sum(distance is None for distance in distances),
        "threshold_slots": thresholds,
        "candidate_slots": candidate_slots,
        "distance_measure": "per-layer exact LRU stack distance in selected-expert rank order",
    }
    return rows, summary


def page_fault_candidate_validity(
    *,
    resident_cache_hits: int,
    nonresident_cache_hits: int,
    pagefile_read_bytes: int | None,
    commit_pressure_fraction: float,
    commit_safety_threshold: float = 0.90,
) -> dict[str, Any]:
    reasons: list[str] = []
    if pagefile_read_bytes is not None and pagefile_read_bytes > 0:
        reasons.append("sustained pagefile reads observed")
    if commit_pressure_fraction > commit_safety_threshold:
        reasons.append("commit pressure exceeds configured safety threshold")
    if nonresident_cache_hits > resident_cache_hits:
        reasons.append("cache hits are predominantly nonresident")
    return {
        "valid_performance_candidate": not reasons,
        "invalidation_reasons": reasons,
        "resident_cache_hits": resident_cache_hits,
        "nonresident_cache_hits": nonresident_cache_hits,
        "pagefile_read_bytes": pagefile_read_bytes,
        "commit_pressure_fraction": commit_pressure_fraction,
        "commit_safety_threshold": commit_safety_threshold,
        "hard_soft_fault_limitation": (
            "selected Windows APIs expose total faults and process I/O but not reliable "
            "hard-versus-soft attribution; unavailable pagefile bytes remain null"
        ),
    }


def prefetch_idle_window_budget(
    *,
    phase: str,
    layer_id: int,
    available_idle_window_ns: int,
    effective_bandwidth_bytes_per_second: float,
    proposed_prefetch_bytes: int,
    subsequently_consumed_bytes: int,
    demand_read_interference_ns: int,
    eviction_bytes: int,
) -> dict[str, Any]:
    if phase not in {"prefill", "decode"}:
        raise ValueError("prefetch phase must be prefill or decode")
    if min(available_idle_window_ns, proposed_prefetch_bytes, subsequently_consumed_bytes) < 0:
        raise ValueError("prefetch measurements must be non-negative")
    maximum = math.floor(
        available_idle_window_ns / 1e9 * effective_bandwidth_bytes_per_second
    )
    extra = max(0, proposed_prefetch_bytes - subsequently_consumed_bytes)
    useful = (
        proposed_prefetch_bytes <= maximum
        and extra == 0
        and demand_read_interference_ns == 0
        and eviction_bytes == 0
    )
    return {
        "phase": phase,
        "layer_id": layer_id,
        "available_idle_window_ns": available_idle_window_ns,
        "effective_bandwidth_bytes_per_second": effective_bandwidth_bytes_per_second,
        "maximum_prefetch_bytes": maximum,
        "proposed_prefetch_bytes": proposed_prefetch_bytes,
        "subsequently_consumed_bytes": subsequently_consumed_bytes,
        "extra_bytes": extra,
        "demand_read_interference_ns": demand_read_interference_ns,
        "eviction_bytes": eviction_bytes,
        "accepted": useful,
        "decision": "enable" if useful else "disable",
    }


def amdahl_gate(
    *,
    optimization: str,
    baseline_end_to_end_ns: int,
    baseline_affected_ns: int,
    optimized_affected_ns: int,
    optimized_end_to_end_ns: int,
) -> dict[str, Any]:
    if min(
        baseline_end_to_end_ns,
        baseline_affected_ns,
        optimized_affected_ns,
        optimized_end_to_end_ns,
    ) <= 0:
        raise ValueError("Amdahl measurements must be positive")
    if baseline_affected_ns > baseline_end_to_end_ns:
        raise ValueError("affected time cannot exceed end-to-end time")
    fraction = baseline_affected_ns / baseline_end_to_end_ns
    theoretical_speedup = 1.0 / (1.0 - fraction) if fraction < 1.0 else math.inf
    kernel_gain = baseline_affected_ns / optimized_affected_ns
    end_to_end_gain = baseline_end_to_end_ns / optimized_end_to_end_ns
    maximum_with_kernel_gain = 1.0 / ((1.0 - fraction) + fraction / kernel_gain)
    return {
        "optimization": optimization,
        "affected_time_fraction": fraction,
        "theoretical_maximum_gain": theoretical_speedup,
        "measured_kernel_gain": kernel_gain,
        "kernel_limited_maximum_end_to_end_gain": maximum_with_kernel_gain,
        "measured_end_to_end_gain": end_to_end_gain,
        "within_amdahl_upper_bound": end_to_end_gain <= maximum_with_kernel_gain + 1e-9,
        "accepted": end_to_end_gain > 1.0 and end_to_end_gain <= maximum_with_kernel_gain + 1e-9,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--expert-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    rows, summary = reuse_distance_curve(
        arguments.trace, expert_bytes=arguments.expert_bytes
    )
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "reuse_distance_curves.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "reuse_distance_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"rows": len(rows), **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
