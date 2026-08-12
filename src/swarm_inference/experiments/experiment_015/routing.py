"""Analyze real Kimi routes without overstating the three-token fixture."""

from __future__ import annotations

import csv
import math
import statistics
from collections import Counter, OrderedDict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_015.evidence import atomic_json, sha256_file


@dataclass(frozen=True, slots=True)
class RouteRow:
    layer: int
    position: int
    experts: tuple[int, ...]
    weights: tuple[float, ...]


def parse_route_trace(path: Path) -> list[RouteRow]:
    """Parse the immutable K3 route trace and recover per-layer positions."""
    source = path.expanduser().resolve()
    per_layer: Counter[int] = Counter()
    rows: list[RouteRow] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        layer = int(fields[2])
        position = per_layer[layer]
        per_layer[layer] += 1
        experts: list[int] = []
        weights: list[float] = []
        for field in fields[3:]:
            expert, weight = field.split(":", 1)
            experts.append(int(expert))
            weights.append(float(weight))
        if len(experts) != 16 or len(set(experts)) != 16:
            raise ValueError(f"route layer {layer} position {position} is not exact top-16")
        if any(expert < 0 or expert >= 896 for expert in experts):
            raise ValueError("route contains an expert outside Kimi's 896-expert domain")
        rows.append(RouteRow(layer, position, tuple(experts), tuple(weights)))
    if not rows:
        raise ValueError("route trace is empty")
    positions = {row.position for row in rows}
    layers = {row.layer for row in rows}
    if positions != {0, 1, 2} or layers != set(range(1, 93)):
        raise ValueError("route trace is not the retained 92-layer, three-token fixture")
    return rows


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _static_owner(expert: int, workers: int) -> int:
    return expert % workers


def _balanced_placement(hit_counts: Counter[int], workers: int) -> dict[int, int]:
    """Greedy load balance with exact expert-count balance as a secondary cost."""
    load = [0] * workers
    count = [0] * workers
    placement: dict[int, int] = {}
    for expert in sorted(range(896), key=lambda item: (-hit_counts[item], item)):
        worker = min(range(workers), key=lambda item: (load[item], count[item], item))
        placement[expert] = worker
        load[worker] += hit_counts[expert]
        count[worker] += 1
    return placement


def _loads(experts: tuple[int, ...], placement: dict[int, int], workers: int) -> list[int]:
    result = [0] * workers
    for expert in experts:
        result[placement[expert]] += 1
    return result


def _lru_hits(route_rows: list[RouteRow], capacity: int) -> tuple[int, int]:
    caches = {layer: OrderedDict[int, None]() for layer in range(1, 93)}
    hits = 0
    requests = 0
    for row in sorted(route_rows, key=lambda item: (item.position, item.layer)):
        cache = caches[row.layer]
        for expert in row.experts:
            requests += 1
            if expert in cache:
                hits += 1
                cache.move_to_end(expert)
            else:
                cache[expert] = None
                if len(cache) > capacity:
                    cache.popitem(last=False)
    return hits, requests


def analyze_routes(
    route_path: Path,
    output_path: Path,
    placement_path: Path,
    residency_path: Path,
    *,
    workers: int = 4,
    pcie_bandwidth_gbps: float = 32.0,
) -> dict[str, Any]:
    """Build routing, placement, prediction, and residency evidence."""
    if workers not in (2, 4, 8, 16):
        raise ValueError("expert placement workers must be 2/4/8/16")
    if pcie_bandwidth_gbps <= 0:
        raise ValueError("PCIe bandwidth must be positive")
    rows = parse_route_trace(route_path)
    by_layer: dict[int, list[RouteRow]] = {layer: [] for layer in range(1, 93)}
    for row in rows:
        by_layer[row.layer].append(row)
    for layer_rows in by_layer.values():
        layer_rows.sort(key=lambda item: item.position)

    static_max: list[float] = []
    static_ratio: list[float] = []
    balanced_max: list[float] = []
    balanced_ratio: list[float] = []
    prediction_intersections: list[int] = []
    jaccards: list[float] = []
    placement_layers: list[dict[str, Any]] = []
    global_hits: Counter[tuple[int, int]] = Counter()

    for layer, layer_rows in by_layer.items():
        hits = Counter(expert for row in layer_rows for expert in row.experts)
        static = {expert: _static_owner(expert, workers) for expert in range(896)}
        balanced = _balanced_placement(hits, workers)
        layer_static_max: list[int] = []
        layer_balanced_max: list[int] = []
        for row in layer_rows:
            for expert in row.experts:
                global_hits[(layer, expert)] += 1
            static_load = _loads(row.experts, static, workers)
            balanced_load = _loads(row.experts, balanced, workers)
            static_max.append(float(max(static_load)))
            static_ratio.append(max(static_load) / max(1, min(static_load)))
            balanced_max.append(float(max(balanced_load)))
            balanced_ratio.append(max(balanced_load) / max(1, min(balanced_load)))
            layer_static_max.append(max(static_load))
            layer_balanced_max.append(max(balanced_load))
        for previous, current in pairwise(layer_rows):
            intersection = len(set(previous.experts) & set(current.experts))
            prediction_intersections.append(intersection)
            jaccards.append(intersection / (32 - intersection))
        moved_selected = [
            {
                "expert": expert,
                "hits": hits[expert],
                "static_worker": static[expert],
                "balanced_worker": balanced[expert],
            }
            for expert in sorted(hits)
            if static[expert] != balanced[expert]
        ]
        placement_layers.append(
            {
                "layer": layer,
                "selected_unique_experts": len(hits),
                "hottest_selected_hits": max(hits.values()),
                "static_max_worker_selections": layer_static_max,
                "balanced_max_worker_selections": layer_balanced_max,
                "moved_selected_experts": moved_selected,
                "full_balanced_owner_by_expert": [balanced[expert] for expert in range(896)],
            }
        )

    transitions = len(prediction_intersections)
    predicted = transitions * 16
    correct = sum(prediction_intersections)
    precision = correct / predicted
    recall = correct / predicted
    activation_bytes_per_expert = 2048 * 4
    predicted_bytes = predicted * activation_bytes_per_expert
    useful_bytes = correct * activation_bytes_per_expert
    wasted_bytes = predicted_bytes - useful_bytes

    # One expert partition is derived from the measured exact four-worker
    # resident bytes, not from a generic MoE estimate.
    expert_bytes = (4 * 3_930_830_848) / 896.0
    residency_rows: list[dict[str, Any]] = []
    for fraction in (1.0, 0.75, 0.50, 0.25):
        capacity = max(1, math.floor(896 * fraction))
        hits, requests = _lru_hits(rows, capacity)
        misses = requests - hits
        bytes_moved = misses * expert_bytes
        transfer_ms = bytes_moved * 8.0 / (pcie_bandwidth_gbps * 1_000_000.0)
        residency_rows.append(
            {
                "gpu_residency_fraction": fraction,
                "experts_per_layer": capacity,
                "requests": requests,
                "cold_start_lru_hits": hits,
                "cold_start_lru_misses": misses,
                "cold_start_hit_rate": hits / requests,
                "expert_bytes": expert_bytes,
                "h2d_bytes_over_fixture": bytes_moved,
                "h2d_transfer_ms_over_fixture_at_configured_bandwidth": transfer_ms,
                "configured_pcie_bandwidth_gbps": pcie_bandwidth_gbps,
                "representative_workload": False,
            }
        )

    source_sha = sha256_file(route_path)
    placement_receipt: dict[str, Any] = {
        "schema_version": "experiment-015-expert-placement-v1",
        "cycle_id": "H015-008A",
        "status": "PASS",
        "evidence_class": "PROJECTED",
        "source_trace_class": "MEASURED",
        "scope": "three target tokens across 92 real Kimi layers; not representative",
        "workers": workers,
        "source_sha256": source_sha,
        "layers": placement_layers,
    }
    atomic_json(placement_path, placement_receipt)
    residency_receipt: dict[str, Any] = {
        "schema_version": "experiment-015-tiered-residency-v1",
        "cycle_id": "H015-010A",
        "status": "PASS",
        "evidence_class": "PROJECTED",
        "source_trace_class": "MEASURED",
        "source_sha256": source_sha,
        "rows": residency_rows,
        "decision": "STOP_INSUFFICIENT_REPRESENTATIVE_ROUTE_HISTORY",
        "limitation": (
            "Cold-start LRU over three tokens cannot establish a production hit rate; "
            "no paid-GPU residency reduction is admitted into architecture search."
        ),
    }
    atomic_json(residency_path, residency_receipt)

    summary: dict[str, Any] = {
        "schema_version": "experiment-015-route-analysis-v1",
        "cycle_id": "H015-008A-H015-009A",
        "status": "PASS",
        "evidence_class": "MEASURED",
        "representative_workload": False,
        "fixture": {
            "layers": len(by_layer),
            "positions_per_layer": 3,
            "route_rows": len(rows),
            "selected_experts_per_row": 16,
            "source": str(route_path.resolve()),
            "source_sha256": source_sha,
        },
        "static_modulo_placement": {
            "workers": workers,
            "mean_critical_worker_selections": statistics.fmean(static_max),
            "p95_critical_worker_selections": _percentile(static_max, 0.95),
            "maximum_critical_worker_selections": max(static_max),
            "mean_hottest_to_coldest_ratio": statistics.fmean(static_ratio),
            "p95_hottest_to_coldest_ratio": _percentile(static_ratio, 0.95),
        },
        "trace_fitted_balanced_placement": {
            "mean_critical_worker_selections": statistics.fmean(balanced_max),
            "p95_critical_worker_selections": _percentile(balanced_max, 0.95),
            "maximum_critical_worker_selections": max(balanced_max),
            "mean_hottest_to_coldest_ratio": statistics.fmean(balanced_ratio),
            "p95_hottest_to_coldest_ratio": _percentile(balanced_ratio, 0.95),
            "critical_load_gain_fraction": 1.0
            - statistics.fmean(balanced_max) / statistics.fmean(static_max),
            "admitted": False,
            "reason": "placement is fitted and evaluated on only the same three tokens",
        },
        "previous_token_same_layer_predictor": {
            "transitions": transitions,
            "predicted_experts": predicted,
            "correct_predictions": correct,
            "precision": precision,
            "recall": recall,
            "mean_jaccard": statistics.fmean(jaccards),
            "p95_jaccard": _percentile(jaccards, 0.95),
            "predicted_activation_bytes": predicted_bytes,
            "useful_activation_bytes": useful_bytes,
            "wasted_activation_bytes": wasted_bytes,
            "admitted": False,
            "reason": "insufficient recall and non-representative three-token fixture",
        },
        "global_trace_hits": {
            "selected_layer_expert_pairs": len(global_hits),
            "hottest_pair_hits": max(global_hits.values()),
            "coldest_pair_hits": min(global_hits.values()),
        },
        "decisions": {
            "H015-EPLB": "STOP_REPRESENTATIVE_TRACE_GATE_NOT_MET",
            "H015-PREDICT": "REJECT_SIMPLE_PREVIOUS_TOKEN_PREDICTOR",
            "dynamic_placement": "NOT_JUSTIFIED",
            "hot_expert_replication": "NOT_ADMITTED",
        },
    }
    atomic_json(output_path, summary)
    return summary


def export_route_rows(route_path: Path, output_path: Path) -> None:
    """Export exact selected experts in an inspectable CSV trace."""
    rows = parse_route_trace(route_path)
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["layer", "position", "slot", "expert", "route_weight"])
        for row in rows:
            for slot, (expert, weight) in enumerate(
                zip(row.experts, row.weights, strict=True)
            ):
                writer.writerow([row.layer, row.position, slot, expert, weight])
    temporary.replace(destination)


__all__ = ["RouteRow", "analyze_routes", "export_route_rows", "parse_route_trace"]
