"""Source-backed routing-imbalance analysis for real Kimi expert microwork."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "experiment-014-k3-real-route-ownership-v1"
EXPERTS = 896
TOPK = 16
MOE_LAYERS = tuple(range(1, 93))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "minimum": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "maximum": 0.0,
        }
    return {
        "count": len(values),
        "minimum": min(values),
        "mean": sum(values) / len(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "maximum": max(values),
    }


def _parse_routes(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    positions: Counter[int] = Counter()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = line.split()
        if len(fields) != 3 + TOPK:
            raise ValueError(
                f"route line {line_number} has {len(fields) - 3} selections"
            )
        call_id, row, layer = (int(value) for value in fields[:3])
        expert_ids: list[int] = []
        weights: list[float] = []
        for field in fields[3:]:
            expert_text, weight_text = field.split(":", 1)
            expert_ids.append(int(expert_text))
            weights.append(float(weight_text))
        if call_id != len(records):
            raise ValueError("route call IDs are not contiguous")
        if row != 0:
            raise ValueError("canonical K3_CHUNK=1 route row must be zero")
        if layer not in MOE_LAYERS:
            raise ValueError(f"unexpected routed layer {layer}")
        if len(set(expert_ids)) != TOPK:
            raise ValueError(f"layer {layer} call {call_id} repeats an expert")
        if any(not 0 <= expert < EXPERTS for expert in expert_ids):
            raise ValueError("route contains an out-of-range expert")
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("route contains an invalid expert weight")
        position = positions[layer]
        positions[layer] += 1
        records.append(
            {
                "call_id": call_id,
                "row": row,
                "layer": layer,
                "position": position,
                "expert_ids": expert_ids,
                "weights": weights,
            }
        )
    if set(positions) != set(MOE_LAYERS) or any(
        positions[layer] != 3 for layer in MOE_LAYERS
    ):
        raise ValueError("canonical route trace must contain three calls for 92 layers")
    for record in records:
        expected_call = record["position"] * len(MOE_LAYERS) + record["layer"] - 1
        if record["call_id"] != expected_call:
            raise ValueError("route trace does not preserve step-major layer ordering")
    return records


def _capacity_assignment(
    records: list[dict[str, Any]], workers: int, train_positions: frozenset[int]
) -> tuple[dict[int, dict[int, int]], dict[str, Any]]:
    capacity = EXPERTS // workers
    assignment: dict[int, dict[int, int]] = {}
    layer_training_loads: dict[str, list[int]] = {}
    changed = 0
    for layer in MOE_LAYERS:
        hits: Counter[int] = Counter()
        for record in records:
            if record["layer"] == layer and record["position"] in train_positions:
                hits.update(record["expert_ids"])
        counts = [0] * workers
        loads = [0] * workers
        layer_assignment: dict[int, int] = {}
        order = sorted(range(EXPERTS), key=lambda expert: (-hits[expert], expert))
        for expert in order:
            eligible = [worker for worker in range(workers) if counts[worker] < capacity]
            owner = min(
                eligible,
                key=lambda worker: (loads[worker], counts[worker], worker),
            )
            layer_assignment[expert] = owner
            counts[owner] += 1
            loads[owner] += hits[expert]
            changed += int(owner != expert % workers)
        if counts != [capacity] * workers:
            raise RuntimeError("capacity-aware ownership violated equal expert capacity")
        assignment[layer] = layer_assignment
        layer_training_loads[str(layer)] = loads
    return assignment, {
        "experts_per_worker_per_layer": capacity,
        "all_layer_worker_capacities": [capacity] * workers,
        "mapping_changes": changed,
        "mapping_entries": len(MOE_LAYERS) * EXPERTS,
        "mapping_change_percent": 100.0 * changed / (len(MOE_LAYERS) * EXPERTS),
        "layer_training_selection_loads": layer_training_loads,
    }


def _evaluate(
    records: list[dict[str, Any]],
    workers: int,
    strategy: str,
    owner: Callable[[int, int], int],
    positions: frozenset[int] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = [
        record
        for record in records
        if positions is None or record["position"] in positions
    ]
    ideal = TOPK / workers
    call_rows: list[dict[str, Any]] = []
    totals = [0] * workers
    weighted_totals = [0.0] * workers
    for record in selected:
        loads = [0] * workers
        weighted = [0.0] * workers
        for expert, weight in zip(
            record["expert_ids"], record["weights"], strict=True
        ):
            worker = owner(record["layer"], expert)
            if not 0 <= worker < workers:
                raise RuntimeError("ownership resolved outside the worker domain")
            loads[worker] += 1
            weighted[worker] += weight
        for worker in range(workers):
            totals[worker] += loads[worker]
            weighted_totals[worker] += weighted[worker]
        critical = max(loads)
        coldest = min(loads)
        call_rows.append(
            {
                "strategy": strategy,
                "call_id": record["call_id"],
                "position": record["position"],
                "layer": record["layer"],
                **{f"worker_{worker}_selections": loads[worker] for worker in range(workers)},
                **{
                    f"worker_{worker}_route_weight": weighted[worker]
                    for worker in range(workers)
                },
                "critical_worker_selections": critical,
                "coldest_worker_selections": coldest,
                "critical_minus_ideal": critical - ideal,
                "ideal_parallel_efficiency": ideal / critical,
                "hottest_to_coldest_ratio": (
                    critical / coldest if coldest else None
                ),
                "critical_route_weight": max(weighted),
                "route_weight_sum": sum(weighted),
            }
        )
    aggregate_mean = sum(totals) / workers
    per_step: list[dict[str, Any]] = []
    for position in sorted({row["position"] for row in call_rows}):
        rows = [row for row in call_rows if row["position"] == position]
        step_totals = [
            sum(int(row[f"worker_{worker}_selections"]) for row in rows)
            for worker in range(workers)
        ]
        per_step.append(
            {
                "position": position,
                "call_count": len(rows),
                "worker_selection_totals": step_totals,
                "aggregate_hottest_worker": step_totals.index(max(step_totals)),
                "aggregate_coldest_worker": step_totals.index(min(step_totals)),
                "aggregate_hottest_to_coldest_ratio": max(step_totals)
                / min(step_totals),
                "critical_worker_selections": _summary(
                    [float(row["critical_worker_selections"]) for row in rows]
                ),
                "ideal_parallel_efficiency": _summary(
                    [float(row["ideal_parallel_efficiency"]) for row in rows]
                ),
            }
        )
    per_layer: list[dict[str, Any]] = []
    for layer in MOE_LAYERS:
        rows = [row for row in call_rows if row["layer"] == layer]
        if not rows:
            continue
        layer_totals = [
            sum(int(row[f"worker_{worker}_selections"]) for row in rows)
            for worker in range(workers)
        ]
        per_layer.append(
            {
                "layer": layer,
                "call_count": len(rows),
                "worker_selection_totals": layer_totals,
                "hottest_to_coldest_ratio": (
                    max(layer_totals) / min(layer_totals)
                    if min(layer_totals)
                    else None
                ),
                "critical_worker_selections": _summary(
                    [float(row["critical_worker_selections"]) for row in rows]
                ),
                "ideal_parallel_efficiency": _summary(
                    [float(row["ideal_parallel_efficiency"]) for row in rows]
                ),
            }
        )
    return {
        "strategy": strategy,
        "call_count": len(call_rows),
        "selection_count": len(call_rows) * TOPK,
        "worker_selection_totals": totals,
        "worker_weight_totals": weighted_totals,
        "aggregate_hottest_worker": totals.index(max(totals)),
        "aggregate_coldest_worker": totals.index(min(totals)),
        "aggregate_hottest_to_coldest_ratio": max(totals) / min(totals),
        "aggregate_range_percent_of_mean": 100.0
        * (max(totals) - min(totals))
        / aggregate_mean,
        "aggregate_max_deviation_percent": 100.0
        * max(abs(value - aggregate_mean) for value in totals)
        / aggregate_mean,
        "critical_worker_selections": _summary(
            [float(row["critical_worker_selections"]) for row in call_rows]
        ),
        "coldest_worker_selections": _summary(
            [float(row["coldest_worker_selections"]) for row in call_rows]
        ),
        "ideal_parallel_efficiency": _summary(
            [float(row["ideal_parallel_efficiency"]) for row in call_rows]
        ),
        "critical_route_weight": _summary(
            [float(row["critical_route_weight"]) for row in call_rows]
        ),
        "critical_load_histogram": dict(
            sorted(
                Counter(
                    str(int(row["critical_worker_selections"])) for row in call_rows
                ).items()
            )
        ),
        "per_step": per_step,
        "per_layer": per_layer,
    }, call_rows


def _expert_hit_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    layer_expert_hits: Counter[tuple[int, int]] = Counter()
    global_expert_id_hits: Counter[int] = Counter()
    unique_per_layer: list[float] = []
    reuse_per_layer: list[float] = []
    for record in records:
        layer = int(record["layer"])
        for expert in record["expert_ids"]:
            layer_expert_hits[(layer, expert)] += 1
            global_expert_id_hits[expert] += 1
    for layer in MOE_LAYERS:
        unique = len(
            {
                expert
                for (observed_layer, expert), count in layer_expert_hits.items()
                if observed_layer == layer and count
            }
        )
        unique_per_layer.append(float(unique))
        reuse_per_layer.append((3 * TOPK) / unique)
    selected_histogram = Counter(str(count) for count in layer_expert_hits.values())
    top_layer_experts = [
        {"layer": layer, "expert": expert, "hits": count}
        for (layer, expert), count in sorted(
            layer_expert_hits.items(), key=lambda item: (-item[1], item[0])
        )[:32]
    ]
    return {
        "layer_expert_opportunities": len(MOE_LAYERS) * EXPERTS,
        "selected_layer_expert_pairs": len(layer_expert_hits),
        "never_selected_layer_expert_pairs": len(MOE_LAYERS) * EXPERTS
        - len(layer_expert_hits),
        "selected_pair_hit_histogram": dict(sorted(selected_histogram.items())),
        "pairs_hit_in_all_three_steps": sum(
            count == 3 for count in layer_expert_hits.values()
        ),
        "unique_selected_experts_per_layer": _summary(unique_per_layer),
        "three_step_effective_reuse_per_layer": _summary(reuse_per_layer),
        "top_layer_specific_experts": top_layer_experts,
        "global_expert_id_hit_summary": _summary(
            [float(global_expert_id_hits[expert]) for expert in range(EXPERTS)]
        ),
        "global_expert_id_hottest": [
            {"expert": expert, "hits_across_layers": hits}
            for expert, hits in global_expert_id_hits.most_common(16)
        ],
        "representativeness_warning": (
            "Each layer contributes only three real token routes; hit frequency can "
            "measure overlap in this trace but cannot establish a persistent hot expert."
        ),
    }


def analyze_sub_layer_routing(
    oracle_routes: Path,
    serial_receipt: Path,
    recovery_receipt: Path,
    output_path: Path,
    csv_path: Path,
    *,
    workers: int = 4,
    cycle_id: str = "H014-SUB-010",
) -> dict[str, Any]:
    """Analyze real route imbalance and held-out ownership without touching CUDA."""
    if workers != 4 or EXPERTS % workers:
        raise ValueError("H014-SUB-010 requires the certified four-worker topology")
    paths = {
        "oracle_routes": oracle_routes.resolve(),
        "serial_receipt": serial_receipt.resolve(),
        "recovery_receipt": recovery_receipt.resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")
    serial = json.loads(paths["serial_receipt"].read_text(encoding="utf-8"))
    recovery = json.loads(paths["recovery_receipt"].read_text(encoding="utf-8"))
    route_sha = _sha256_file(paths["oracle_routes"])
    source_route = Path(serial["raw_artifacts"]["routes"]).resolve()
    integrity = {
        "serial_receipt_pass": serial.get("status") == "PASS",
        "canonical_idot_zero": serial.get("environment_overrides", {}).get("K3_IDOT")
        == "0",
        "route_path_matches_receipt": source_route == paths["oracle_routes"],
        "route_sha_matches_receipt": route_sha == serial["routes"]["route_sha256"],
        "recovery_receipt_pass": recovery.get("status") == "PASS"
        and recovery.get("hypothesis_supported") is True,
    }
    if not all(integrity.values()):
        raise ValueError(f"route provenance gate failed: {integrity}")
    records = _parse_routes(paths["oracle_routes"])
    integrity.update(
        {
            "route_calls_match_receipt": len(records)
            == int(serial["routes"]["route_calls"]),
            "selections_match_receipt": len(records) * TOPK
            == int(serial["routes"]["selection_count"]),
            "three_forward_steps": int(serial["trace"]["forward_steps"]) == 3,
            "all_routes_unique_per_call": all(
                len(set(record["expert_ids"])) == TOPK for record in records
            ),
        }
    )
    if not all(integrity.values()):
        raise ValueError(f"route content gate failed: {integrity}")

    def static_owner(_layer: int, expert: int) -> int:
        return expert % workers

    capacity_mapping, capacity = _capacity_assignment(
        records, workers, frozenset({0, 1})
    )

    def capacity_owner(layer: int, expert: int) -> int:
        return capacity_mapping[layer][expert]

    evaluations: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for strategy, owner in (
        ("static_expert_id_mod_4", static_owner),
        ("capacity_preserving_trained_steps_0_1", capacity_owner),
    ):
        all_metrics, all_rows = _evaluate(records, workers, strategy, owner)
        train_metrics, _ = _evaluate(
            records, workers, strategy, owner, frozenset({0, 1})
        )
        heldout_metrics, _ = _evaluate(
            records, workers, strategy, owner, frozenset({2})
        )
        evaluations[strategy] = {
            "all_steps": all_metrics,
            "training_steps_0_1": train_metrics,
            "heldout_step_2": heldout_metrics,
        }
        csv_rows.extend(all_rows)

    static = evaluations["static_expert_id_mod_4"]
    candidate = evaluations["capacity_preserving_trained_steps_0_1"]
    static_heldout = static["heldout_step_2"]["critical_worker_selections"]
    candidate_heldout = candidate["heldout_step_2"]["critical_worker_selections"]
    mean_improvement = 100.0 * (
        static_heldout["mean"] - candidate_heldout["mean"]
    ) / static_heldout["mean"]
    p95_improvement = 100.0 * (
        static_heldout["p95"] - candidate_heldout["p95"]
    ) / static_heldout["p95"]
    training_static = static["training_steps_0_1"]["critical_worker_selections"]
    training_candidate = candidate["training_steps_0_1"][
        "critical_worker_selections"
    ]
    training_improvement = 100.0 * (
        training_static["mean"] - training_candidate["mean"]
    ) / training_static["mean"]

    worker_rows = recovery["groups"][0]["ready"]["workers"]
    resident_expert_bytes = {int(row["resident_expert_tensor_bytes"]) for row in worker_rows}
    owned_experts = {int(row["owned_expert_count"]) for row in worker_rows}
    if len(resident_expert_bytes) != 1 or owned_experts != {EXPERTS // workers}:
        raise ValueError("recovery receipt has inconsistent expert residency")
    bytes_per_expert = next(iter(resident_expert_bytes)) // (EXPERTS // workers)

    gates = {
        "source_integrity": all(integrity.values()),
        "aggregate_static_range_at_most_5_percent": static["all_steps"]
        ["aggregate_range_percent_of_mean"]
        <= 5.0,
        "equal_224_expert_capacity": capacity["all_layer_worker_capacities"]
        == [224, 224, 224, 224],
        "heldout_mean_improvement_below_10_percent": mean_improvement < 10.0,
    }
    hypothesis_supported = all(gates.values())
    useful_reassignment = mean_improvement >= 10.0 and p95_improvement >= 0.0
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "cycle_id": cycle_id,
        "status": "PASS",
        "hypothesis": (
            "Static modulo-4 ownership has <=5% aggregate selection range, and "
            "equal-capacity ownership learned on steps 0/1 improves held-out step-2 "
            "mean critical load by <10%."
        ),
        "hypothesis_supported": hypothesis_supported,
        "configuration": {
            "workers": workers,
            "experts": EXPERTS,
            "experts_per_worker": EXPERTS // workers,
            "topk": TOPK,
            "layers": len(MOE_LAYERS),
            "positions": 3,
            "training_positions": [0, 1],
            "heldout_positions": [2],
            "heldout_usefulness_threshold_percent": 10.0,
            "aggregate_range_gate_percent": 5.0,
        },
        "sources": {
            name: {"path": str(path), "sha256": _sha256_file(path)}
            for name, path in paths.items()
        },
        "source_context": {
            "prompt": serial["prompt"],
            "prompt_tokens": serial["prompt_tokens"],
            "generated_token_ids": serial["generated_token_ids"],
            "forward_steps": serial["trace"]["forward_steps"],
            "route_calls": serial["routes"]["route_calls"],
            "selection_count": serial["routes"]["selection_count"],
            "route_sha256": route_sha,
            "canonical_activation_semantics": "K3_IDOT=0",
        },
        "integrity_gates": integrity,
        "expert_hits": _expert_hit_analysis(records),
        "evaluations": evaluations,
        "capacity_assignment": capacity,
        "comparison": {
            "training_mean_critical_improvement_percent": training_improvement,
            "heldout_mean_critical_improvement_percent": mean_improvement,
            "heldout_p95_critical_improvement_percent": p95_improvement,
            "heldout_static_mean_critical_selections": static_heldout["mean"],
            "heldout_candidate_mean_critical_selections": candidate_heldout["mean"],
            "heldout_static_p95_critical_selections": static_heldout["p95"],
            "heldout_candidate_p95_critical_selections": candidate_heldout["p95"],
        },
        "memory": {
            "measured_resident_expert_bytes_per_224_experts": next(
                iter(resident_expert_bytes)
            ),
            "measured_resident_bytes_per_expert": bytes_per_expert,
            "capacity_candidate_additional_expert_bytes": 0,
            "capacity_candidate_experts_per_worker": [224, 224, 224, 224],
        },
        "acceptance_gates": gates,
        "inspection": {
            "actual_bottleneck": (
                "instantaneous selected-expert count on the critical worker"
            ),
            "representativeness": (
                "Real complete-graph routes, but only a two-token prompt plus one "
                "decode step; sufficient for instantaneous/fleet imbalance and a "
                "minimal held-out check, insufficient for persistent hot-expert claims."
            ),
            "cuda_activity": "none; immutable retained real-route analysis",
        },
        "decision": {
            "static_modulo_ownership": "RETAIN" if not useful_reassignment else "MODIFY",
            "capacity_assignment": (
                "REJECT_NOT_HELD_OUT_USEFUL"
                if not useful_reassignment
                else "PROMOTE_TO_REAL_CUDA_CERTIFICATION"
            ),
            "hot_expert_replication": (
                "NOT_JUSTIFIED_FROM_THREE_TOKENS"
                if not useful_reassignment
                else "FORM_NEXT_HYPOTHESIS_ONLY"
            ),
            "next_hypothesis": (
                "Separate decode and prefill state/capacity measurement on the retained "
                "static ownership topology."
                if not useful_reassignment
                else "Certify the held-out-useful capacity assignment on real CUDA."
            ),
        },
        "artifacts": {
            "csv": str(csv_path.resolve()),
            "csv_rows": len(csv_rows),
        },
    }
    _atomic_csv(csv_path, csv_rows)
    receipt["artifacts"]["csv_sha256"] = _sha256_file(csv_path.resolve())
    _atomic_json(output_path, receipt)
    return receipt
