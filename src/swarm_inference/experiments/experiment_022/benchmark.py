"""Preregistered A-to-E placement benchmark and mechanical E022 analyses."""

from __future__ import annotations

import copy
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .dynamic import run_dynamic_scenarios
from .evaluator import PlacementEvaluator, common_endpoint_policy
from .io import atomic_write_json, write_csv
from .model_graph import candidate_catalog
from .models import Inventory, ModelGraph, PartitionKind, PlacementPlan, PlannerLevel
from .planner import OptimizerConfiguration, OptimizerResult, SharedPlacementOptimizer
from .service import ResidentServiceModel

LEVELS = (
    PlannerLevel.A,
    PlannerLevel.B,
    PlannerLevel.C,
    PlannerLevel.D,
    PlannerLevel.E,
)


def _is_whole_only(plan: PlacementPlan) -> bool:
    return plan.feasible and all(
        assignment.partition_kind is PartitionKind.WHOLE_LAYER
        for assignment in plan.assignments
    )


def _relabel(plan: PlacementPlan, level: PlannerLevel) -> PlacementPlan:
    value = copy.deepcopy(plan)
    value.planner_level = level
    return value


def _detail_plan(
    model: ModelGraph,
    inventory: Inventory,
    service: ResidentServiceModel,
    plan: PlacementPlan,
) -> None:
    if not plan.feasible:
        return
    endpoint = common_endpoint_policy(model, inventory)
    if endpoint is None:
        raise RuntimeError("selected feasible plan lost its common endpoint policy")
    PlacementEvaluator(model, inventory, service, endpoint).evaluate(
        plan, include_records=True
    )


def _critical_partition(plan: PlacementPlan) -> dict[str, float]:
    if not plan.feasible or not plan.event_receipt:
        return {kind.value: 0.0 for kind in PartitionKind}
    records = {
        row["task_id"]: row for row in plan.event_receipt.get("records", [])
    }
    assignments = {value.layer_id: value.partition_kind for value in plan.assignments}
    totals = {kind.value: 0.0 for kind in PartitionKind}
    overall = 0.0
    for identifier in plan.event_receipt.get("critical_path_task_ids", []):
        row = records.get(identifier)
        if row is None or row.get("layer_id") is None:
            continue
        kind = assignments.get(int(row["layer_id"]))
        if kind is None:
            continue
        duration = float(row["duration_ms"])
        totals[kind.value] += duration
        overall += duration
    return {
        key: value / overall * 100 if overall else 0.0
        for key, value in totals.items()
    }


def plan_row(
    inventory: Inventory,
    result: OptimizerResult,
) -> dict[str, Any]:
    plan = result.plan
    available = sum(
        node.accelerator_memory_bytes for node in inventory.available_nodes
    )
    used = sum(plan.memory_used_by_node.values()) if plan.feasible else 0
    usage = plan.usage_counts() if plan.feasible else {
        kind.value: 0 for kind in PartitionKind
    }
    total_layers = max(1, len(plan.assignments))
    critical = _critical_partition(plan)
    node_map = inventory.node_map()
    projected_cost = sum(node_map[node].cost for node in plan.used_nodes) if plan.feasible else 0.0
    return {
        "inventory_id": inventory.inventory_id,
        "family": inventory.family,
        "scenario": inventory.scenario,
        "planner_level": plan.planner_level.value,
        "feasible": plan.feasible,
        "infeasible_reason": plan.infeasible_reason or "",
        "exact_tok_s_per_user": plan.exact_tok_s_per_user if plan.feasible else "",
        "critical_path_ms": plan.critical_path_ms if plan.feasible else "",
        "total_worker_compute_ms": plan.total_worker_compute_ms if plan.feasible else "",
        "network_bytes_per_target_pass": plan.network_bytes if plan.feasible else "",
        "network_bytes_per_token": plan.network_bytes / 17 if plan.feasible else "",
        "messages_per_target_pass": plan.messages if plan.feasible else "",
        "serial_waits_per_target_pass": plan.serial_waits if plan.feasible else "",
        "worker_seconds_per_token": plan.worker_seconds_per_token if plan.feasible else "",
        "workers_used": len(plan.used_nodes) if plan.feasible else 0,
        "useful_nodes": len(plan.used_nodes) if plan.feasible else 0,
        "available_nodes": len(inventory.available_nodes),
        "available_accelerator_memory_bytes": available,
        "resident_memory_bytes": used,
        "stranded_memory_bytes": available - used,
        "memory_utilization_percent": used / available * 100 if available else 0.0,
        "projected_abstract_node_cost": projected_cost,
        "chunk_rows": plan.chunk_rows,
        "whole_layer_count": usage[PartitionKind.WHOLE_LAYER.value],
        "whole_expert_count": usage[PartitionKind.WHOLE_EXPERT.value],
        "expert_shard_count": usage[PartitionKind.EXPERT_SHARD.value],
        "attention_projection_shard_count": usage[
            PartitionKind.ATTENTION_PROJECTION_SHARD.value
        ],
        "full_mixed_count": usage[PartitionKind.FULL_MIXED_STRIPE.value],
        "whole_layer_percent": usage[PartitionKind.WHOLE_LAYER.value] / total_layers * 100,
        "whole_expert_percent": usage[PartitionKind.WHOLE_EXPERT.value] / total_layers * 100,
        "expert_shard_percent": usage[PartitionKind.EXPERT_SHARD.value] / total_layers * 100,
        "attention_projection_shard_percent": usage[
            PartitionKind.ATTENTION_PROJECTION_SHARD.value
        ]
        / total_layers
        * 100,
        "full_mixed_percent": usage[PartitionKind.FULL_MIXED_STRIPE.value]
        / total_layers
        * 100,
        "critical_path_whole_layer_percent": critical[PartitionKind.WHOLE_LAYER.value],
        "critical_path_whole_expert_percent": critical[PartitionKind.WHOLE_EXPERT.value],
        "critical_path_expert_shard_percent": critical[PartitionKind.EXPERT_SHARD.value],
        "critical_path_attention_projection_percent": critical[
            PartitionKind.ATTENTION_PROJECTION_SHARD.value
        ],
        "critical_path_full_mixed_percent": critical[
            PartitionKind.FULL_MIXED_STRIPE.value
        ],
        "optimizer_elapsed_ms": result.elapsed_ms,
        "optimizer_proposals": result.proposals,
        "optimizer_exact_evaluations": result.exact_evaluations,
        "fallback_retained": result.fallback_retained,
        "objective_primary": "maximize exact K3 target-only tok/s/user",
    }


def _catalog_with_validation(
    model: ModelGraph,
    service: ResidentServiceModel,
) -> dict[str, Any]:
    catalog = candidate_catalog(model)
    eligible = 0
    for row in catalog["candidates"]:
        layer = model.layers[int(row["layer"])]
        kind = PartitionKind(row["partition_type"])
        supported = service.supported_candidate(
            layer, kind, int(row["degree"]), 1
        )
        row["correctness_status"] = "PASS" if supported else "INELIGIBLE_UNVALIDATED"
        row["service_status"] = (
            "VALIDATED_PHYSICAL_OR_FEATURE_INTERPOLATION"
            if supported
            else "NO_HEADLINE_SERVICE"
        )
        row["headline_eligible"] = supported
        row["eligible_chunk_rows"] = [1] if supported and kind is not PartitionKind.WHOLE_LAYER else (
            [1, 2, 4] if supported else []
        )
        eligible += int(supported)
    catalog["headline_eligible_candidates"] = eligible
    catalog["imaginary_candidates_admitted"] = 0
    return catalog


def run_static_suite(
    artifact_root: Path,
    model: ModelGraph,
    inventories: list[Inventory],
    service: ResidentServiceModel,
    configuration: OptimizerConfiguration,
) -> dict[str, Any]:
    optimizer = SharedPlacementOptimizer(model, service, configuration)
    all_rows: list[dict[str, Any]] = []
    convergence: list[dict[str, Any]] = []
    plans: dict[tuple[str, PlannerLevel], PlacementPlan] = {}
    optimizer_results: dict[tuple[str, PlannerLevel], OptimizerResult] = {}
    placement_root = artifact_root / "planner" / "placements"
    for inventory_index, inventory in enumerate(inventories, 1):
        print(
            f"[e022 planner] inventory {inventory_index:02d}/{len(inventories)} "
            f"{inventory.inventory_id}",
            flush=True,
        )
        results: dict[PlannerLevel, OptimizerResult] = {}
        incumbent: PlacementPlan | None = None
        for level in LEVELS:
            result = optimizer.optimize(inventory, level, fallback=incumbent)
            results[level] = result
            if result.plan.feasible:
                incumbent = result.plan

        # If a larger action space discovers a strictly better all-whole
        # placement, promote it into A and rerun the ladder. This is a baseline
        # quality repair, never evidence for sub-layer value.
        whole_discoveries = [
            result.plan
            for result in results.values()
            if _is_whole_only(result.plan)
        ]
        best_whole = max(
            whole_discoveries,
            key=lambda plan: plan.objective_tuple or (),
            default=results[PlannerLevel.A].plan,
        )
        if (
            best_whole.feasible
            and (
                not results[PlannerLevel.A].plan.feasible
                or (best_whole.objective_tuple or ())
                > (results[PlannerLevel.A].plan.objective_tuple or ())
            )
        ):
            promoted = _relabel(best_whole, PlannerLevel.A)
            results[PlannerLevel.A] = OptimizerResult(
                plan=promoted,
                convergence=results[PlannerLevel.A].convergence,
                elapsed_ms=results[PlannerLevel.A].elapsed_ms,
                proposals=results[PlannerLevel.A].proposals,
                exact_evaluations=results[PlannerLevel.A].exact_evaluations,
                fallback_retained=False,
            )
            incumbent = promoted
            for level in LEVELS[1:]:
                results[level] = optimizer.optimize(
                    inventory, level, fallback=incumbent
                )
                if results[level].plan.feasible:
                    incumbent = results[level].plan

        fallback = results[PlannerLevel.A].plan
        for level in LEVELS[1:]:
            results[level].plan.verify_dominance_fallback(fallback)
        for level, result in results.items():
            _detail_plan(model, inventory, service, result.plan)
            row = plan_row(inventory, result)
            all_rows.append(row)
            plans[(inventory.inventory_id, level)] = result.plan
            optimizer_results[(inventory.inventory_id, level)] = result
            convergence.extend(result.convergence)
            atomic_write_json(
                placement_root / f"{inventory.inventory_id}-{level.value}.json",
                result.plan.as_manifest(model, inventory),
            )
    write_csv(artifact_root / "planner" / "ablation-results.csv", all_rows)
    write_csv(
        artifact_root / "planner" / "whole-layer-results.csv",
        [row for row in all_rows if row["planner_level"] == "A"],
    )
    write_csv(
        artifact_root / "planner" / "adaptive-results.csv",
        [row for row in all_rows if row["planner_level"] == "E"],
    )
    write_csv(
        artifact_root / "validation" / "optimizer-convergence.csv",
        convergence,
    )
    atomic_write_json(
        artifact_root / "planner" / "candidate-catalog.json",
        _catalog_with_validation(model, service),
    )
    analysis = analyze_static(inventories, all_rows)
    for name, rows in analysis.items():
        write_csv(artifact_root / "analysis" / f"{name}.csv", rows)
    return {
        "rows": all_rows,
        "plans": plans,
        "optimizer_results": optimizer_results,
        "analysis": analysis,
        "configuration": configuration,
    }


def analyze_static(
    inventories: list[Inventory],
    all_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_key = {
        (str(row["inventory_id"]), str(row["planner_level"])): row
        for row in all_rows
    }
    uplift: list[dict[str, Any]] = []
    unlocks: list[dict[str, Any]] = []
    crossings: list[dict[str, Any]] = []
    usage: list[dict[str, Any]] = []
    memory: list[dict[str, Any]] = []
    critical: list[dict[str, Any]] = []
    for inventory in inventories:
        a = by_key[(inventory.inventory_id, "A")]
        e = by_key[(inventory.inventory_id, "E")]
        if bool(a["feasible"]):
            a_tps = float(a["exact_tok_s_per_user"])
            e_tps = float(e["exact_tok_s_per_user"])
            value = e_tps / a_tps - 1
            uplift.append(
                {
                    "inventory_id": inventory.inventory_id,
                    "family": inventory.family,
                    "whole_tok_s": a_tps,
                    "adaptive_tok_s": e_tps,
                    "throughput_uplift": value,
                    "throughput_uplift_percent": value * 100,
                    "critical_path_reduction_percent": (
                        1
                        - float(e["critical_path_ms"])
                        / float(a["critical_path_ms"])
                    )
                    * 100,
                    "additional_nodes_made_useful": int(e["useful_nodes"])
                    - int(a["useful_nodes"]),
                    "memory_utilization_delta_percent": float(
                        e["memory_utilization_percent"]
                    )
                    - float(a["memory_utilization_percent"]),
                    "stranded_memory_delta_bytes": int(e["stranded_memory_bytes"])
                    - int(a["stranded_memory_bytes"]),
                    "network_delta_bytes_per_token": float(e["network_bytes_per_token"])
                    - float(a["network_bytes_per_token"]),
                    "worker_compute_delta_ms": float(e["total_worker_compute_ms"])
                    - float(a["total_worker_compute_ms"]),
                    "worker_seconds_delta_per_token": float(e["worker_seconds_per_token"])
                    - float(a["worker_seconds_per_token"]),
                    "adaptive_uses_sublayer": float(e["whole_layer_percent"]) < 99.999,
                    "dominance_pass": e_tps >= a_tps * 0.99,
                }
            )
            if a_tps < 5 <= e_tps:
                crossings.append(
                    {
                        "inventory_id": inventory.inventory_id,
                        "family": inventory.family,
                        "whole_tok_s": a_tps,
                        "adaptive_tok_s": e_tps,
                        "crossing": "TARGET_CROSSING_DUE_TO_SUBLAYER",
                    }
                )
        elif bool(e["feasible"]):
            unlocks.append(
                {
                    "inventory_id": inventory.inventory_id,
                    "family": inventory.family,
                    "whole_feasible": False,
                    "adaptive_feasible": True,
                    "adaptive_tok_s": e["exact_tok_s_per_user"],
                    "result": "SUB_LAYER_UNLOCKED_FEASIBILITY",
                }
            )
        usage.append(
            {
                key: e[key]
                for key in (
                    "inventory_id",
                    "family",
                    "exact_tok_s_per_user",
                    "whole_layer_percent",
                    "whole_expert_percent",
                    "expert_shard_percent",
                    "attention_projection_shard_percent",
                    "full_mixed_percent",
                    "critical_path_whole_layer_percent",
                    "critical_path_whole_expert_percent",
                    "critical_path_expert_shard_percent",
                    "critical_path_attention_projection_percent",
                    "critical_path_full_mixed_percent",
                )
            }
        )
        for label, row in (("whole", a), ("adaptive", e)):
            memory.append(
                {
                    "inventory_id": inventory.inventory_id,
                    "family": inventory.family,
                    "planner": label,
                    "feasible": row["feasible"],
                    "resident_memory_bytes": row["resident_memory_bytes"],
                    "stranded_memory_bytes": row["stranded_memory_bytes"],
                    "memory_utilization_percent": row["memory_utilization_percent"],
                }
            )
            critical.append(
                {
                    "inventory_id": inventory.inventory_id,
                    "family": inventory.family,
                    "planner": label,
                    "feasible": row["feasible"],
                    "critical_path_ms": row["critical_path_ms"],
                    "compute_work_ms": row["total_worker_compute_ms"],
                    "network_bytes_per_token": row["network_bytes_per_token"],
                    "serial_waits": row["serial_waits_per_target_pass"],
                    "messages": row["messages_per_target_pass"],
                }
            )
    return {
        "throughput-uplift": uplift,
        "capacity-unlocks": unlocks,
        "target-crossings": crossings,
        "sublayer-usage": usage,
        "memory-utilization": memory,
        "critical-path": critical,
    }


def run_dynamic_suite(
    artifact_root: Path,
    inventories: list[Inventory],
    static: dict[str, Any],
    optimizer: SharedPlacementOptimizer,
) -> dict[str, list[dict[str, Any]]]:
    selected = [inventory for inventory in inventories if inventory.family == "full-mixed"][:5]
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for inventory in selected:
        initial = static["plans"][(inventory.inventory_id, PlannerLevel.E)]
        if not initial.feasible:
            continue
        print(f"[e022 dynamic] {inventory.inventory_id}", flush=True)
        result = run_dynamic_scenarios(inventory, initial, optimizer)
        for name, row in result.items():
            rows[name].append(row)
    names = {
        "JOIN_USEFUL": "join-useful.csv",
        "JOIN_HARMFUL": "join-harmful.csv",
        "SLOWDOWN": "slowdown.csv",
        "NETWORK_DEGRADATION": "network-degradation.csv",
        "NODE_LOSS": "node-loss.csv",
    }
    for name, filename in names.items():
        write_csv(artifact_root / "dynamic" / filename, rows.get(name, []))
    return dict(rows)


def headline_statistics(analysis: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    uplifts = [float(row["throughput_uplift_percent"]) for row in analysis["throughput-uplift"]]
    return {
        "a_feasible_inventory_count": len(uplifts),
        "median_uplift_percent": statistics.median(uplifts) if uplifts else None,
        "p25_uplift_percent": float(np.percentile(uplifts, 25)) if uplifts else None,
        "p75_uplift_percent": float(np.percentile(uplifts, 75)) if uplifts else None,
        "largest_uplift_percent": max(uplifts, default=None),
        "wins_ge_20_percent": sum(value >= 20 for value in uplifts),
        "wins_ge_10_percent": sum(value >= 10 for value in uplifts),
        "regressions_gt_1_percent": sum(value < -1 for value in uplifts),
        "capacity_unlocks": len(analysis["capacity-unlocks"]),
        "target_crossings": len(analysis["target-crossings"]),
    }


__all__ = [
    "LEVELS",
    "analyze_static",
    "headline_statistics",
    "plan_row",
    "run_dynamic_suite",
    "run_static_suite",
]
