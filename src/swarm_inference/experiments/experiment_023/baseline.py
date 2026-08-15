"""Frozen E022 placement loading and deterministic U_STRONG utilities."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_022.evaluator import (
    EndpointPolicy,
    PlacementEvaluator,
)
from swarm_inference.experiments.experiment_022.models import (
    Inventory,
    LayerAssignment,
    ModelGraph,
    PartitionKind,
    PlacementPlan,
    PlannerLevel,
)
from swarm_inference.experiments.experiment_022.service import ResidentServiceModel

from .freeze import FROZEN_CONSTANTS
from .models import Arm, E023Plan
from .replica_memory import abstract_node_cost
from .serving_engine import CONCURRENCY_LEVELS, ServingEngine, ServingRun


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected placement object: {path}")
    return value


def load_frozen_placement(path: Path, inventory: Inventory) -> PlacementPlan:
    """Reconstruct an E022 placement without rerunning its randomized optimizer."""

    value = _read_object(path)
    if value.get("inventory_id") != inventory.inventory_id:
        raise ValueError("placement/inventory ID mismatch")
    feasible = bool(value.get("feasible"))
    endpoint_memory: dict[str, int] = {}
    endpoint_checkpoint: dict[str, int] = {}
    layer_pieces: dict[int, list[tuple[str, dict[str, Any]]]] = {}
    for node in value.get("nodes", ()):
        node_id = str(node["node_id"])
        for piece in node.get("pieces", ()):
            name = str(piece["piece"])
            if name == "common_non_transformer":
                endpoint_memory[node_id] = int(piece["resident_memory_bytes"])
                endpoint_checkpoint[node_id] = int(piece["checkpoint_bytes"])
                continue
            prefix = "transformer_layer_"
            if not name.startswith(prefix):
                raise ValueError(f"unknown frozen placement piece {name}")
            layer_id = int(name[len(prefix) :])
            layer_pieces.setdefault(layer_id, []).append((node_id, dict(piece)))

    assignments: list[LayerAssignment] = []
    for layer_id, rows in sorted(layer_pieces.items()):
        candidate_ids = {str(piece["candidate_id"]) for _, piece in rows}
        kinds = {str(piece["partition_type"]) for _, piece in rows}
        degrees = {int(piece["degree"]) for _, piece in rows}
        if len(candidate_ids) != 1 or len(kinds) != 1 or len(degrees) != 1:
            raise ValueError(f"ambiguous frozen assignment for layer {layer_id}")
        degree = degrees.pop()
        if len(rows) != degree:
            raise ValueError(f"frozen layer {layer_id} does not contain degree workers")
        coordinators = [
            (node_id, piece) for node_id, piece in rows if bool(piece.get("coordinator"))
        ]
        if len(coordinators) != 1:
            raise ValueError(f"frozen layer {layer_id} has ambiguous coordinator")
        coordinator, coordinator_piece = coordinators[0]
        dependencies = tuple(str(node) for node in coordinator_piece["network_dependencies"])
        node_ids = (coordinator, *dependencies)
        concrete = {node_id for node_id, _ in rows}
        if len(node_ids) != degree or set(node_ids) != concrete:
            raise ValueError(f"frozen layer {layer_id} worker ordering is unrecoverable")
        by_node = {node_id: piece for node_id, piece in rows}
        assignments.append(
            LayerAssignment(
                layer_id=layer_id,
                partition_kind=PartitionKind(kinds.pop()),
                degree=degree,
                node_ids=node_ids,
                memory_by_node={
                    node_id: int(by_node[node_id]["resident_memory_bytes"])
                    for node_id in node_ids
                },
                checkpoint_bytes_by_node={
                    node_id: int(by_node[node_id]["checkpoint_bytes"])
                    for node_id in node_ids
                },
                coordinator_node_id=coordinator,
                candidate_id=candidate_ids.pop(),
            )
        )
    if feasible and [row.layer_id for row in assignments] != list(range(93)):
        raise ValueError("feasible frozen placement does not assign layers 0..92")

    objective = dict(value.get("objective", {}))
    memory_used: dict[str, int] = {}
    for node in value.get("nodes", ()):
        assigned = int(node.get("assigned_memory_bytes", 0))
        if assigned:
            memory_used[str(node["node_id"])] = assigned
    plan = PlacementPlan(
        inventory_id=inventory.inventory_id,
        planner_level=PlannerLevel(str(value["planner_level"])),
        chunk_rows=int(value["chunk_rows"]),
        assignments=assignments,
        endpoint_memory_by_node=endpoint_memory,
        endpoint_checkpoint_bytes_by_node=endpoint_checkpoint,
        feasible=feasible,
        infeasible_reason=None if feasible else "frozen E022 manifest infeasible",
        exact_tok_s_per_user=(
            float(objective["exact_tok_s_per_user"])
            if objective.get("exact_tok_s_per_user") is not None
            else None
        ),
        critical_path_ms=(
            float(objective["critical_path_ms"])
            if objective.get("critical_path_ms") is not None
            else None
        ),
        total_worker_compute_ms=(
            float(objective["total_worker_compute_ms"])
            if objective.get("total_worker_compute_ms") is not None
            else None
        ),
        network_bytes=int(objective.get("network_bytes") or 0),
        used_nodes=tuple(sorted(memory_used)),
        memory_used_by_node=memory_used,
    )
    return plan


def clone_placement(plan: PlacementPlan) -> PlacementPlan:
    return copy.deepcopy(plan)


def frozen_placement_paths(repo: Path, inventory_id: str) -> tuple[Path, ...]:
    root = repo / "artifacts/experiment-022/completion/rerun/placements"
    return tuple(root / f"{inventory_id}-{level}.json" for level in "ABCDE")


def _recompute_memory(plan: PlacementPlan) -> None:
    memory = dict(plan.endpoint_memory_by_node)
    for assignment in plan.assignments:
        for node_id, value in assignment.memory_by_node.items():
            memory[node_id] = memory.get(node_id, 0) + value
    plan.memory_used_by_node = {node: value for node, value in memory.items() if value}
    plan.used_nodes = tuple(sorted(plan.memory_used_by_node))


def best_whole_layer_relocation(
    model: ModelGraph,
    inventory: Inventory,
    service: ResidentServiceModel,
    endpoint: EndpointPolicy,
    plan: PlacementPlan,
) -> tuple[PlacementPlan, dict[str, Any]] | None:
    """Exhaustively find the best one-layer WHOLE_LAYER relocation."""

    current = clone_placement(plan)
    _recompute_memory(current)
    evaluator = PlacementEvaluator(model, inventory, service, endpoint)
    evaluator.evaluate(current)
    assert current.exact_tok_s_per_user is not None
    current_objective = current.exact_tok_s_per_user
    best: tuple[float, int, str, PlacementPlan] | None = None
    for assignment in sorted(current.assignments, key=lambda row: row.layer_id):
        if assignment.partition_kind is not PartitionKind.WHOLE_LAYER:
            continue
        source = assignment.node_ids[0]
        layer = model.layers[assignment.layer_id]
        memory_without = dict(current.memory_used_by_node)
        memory_without[source] -= assignment.memory_by_node[source]
        for destination in sorted(inventory.available_nodes, key=lambda row: row.node_id):
            if (
                destination.node_id == source
                or "WHOLE_LAYER" not in destination.runtime_capabilities
                or memory_without.get(destination.node_id, 0) + layer.resident_bytes
                > destination.accelerator_memory_bytes
            ):
                continue
            candidate = clone_placement(current)
            candidate.assignments[assignment.layer_id] = LayerAssignment(
                layer_id=assignment.layer_id,
                partition_kind=PartitionKind.WHOLE_LAYER,
                degree=1,
                node_ids=(destination.node_id,),
                memory_by_node={destination.node_id: layer.resident_bytes},
                checkpoint_bytes_by_node={
                    destination.node_id: layer.checkpoint_bytes
                },
                coordinator_node_id=destination.node_id,
                candidate_id=assignment.candidate_id,
            )
            _recompute_memory(candidate)
            evaluator.evaluate(candidate)
            assert candidate.exact_tok_s_per_user is not None
            gain = candidate.exact_tok_s_per_user / current_objective - 1.0
            value = (gain, assignment.layer_id, destination.node_id, candidate)
            if best is None or (-value[0], value[1], value[2]) < (
                -best[0],
                best[1],
                best[2],
            ):
                best = value
    if best is None:
        return None
    gain, layer_id, destination, candidate = best
    source = current.assignments[layer_id].node_ids[0]
    return candidate, {
        "inventory_id": inventory.inventory_id,
        "layer_id": layer_id,
        "source_node_id": source,
        "destination_node_id": destination,
        "objective_before": current_objective,
        "objective_after": candidate.exact_tok_s_per_user,
        "relative_gain_percent": 100.0 * gain,
    }


def repair_whole_layer_relocations(
    model: ModelGraph,
    inventory: Inventory,
    service: ResidentServiceModel,
    endpoint: EndpointPolicy,
    plan: PlacementPlan,
    *,
    starting_candidate: str,
) -> tuple[PlacementPlan, list[dict[str, Any]]]:
    if not plan.feasible:
        return clone_placement(plan), []
    current = clone_placement(plan)
    accepted: list[dict[str, Any]] = []
    for iteration in range(1, 17):
        result = best_whole_layer_relocation(
            model, inventory, service, endpoint, current
        )
        if result is None:
            break
        candidate, row = result
        if float(row["relative_gain_percent"]) < 0.10:
            break
        row.update(
            {
                "starting_candidate": starting_candidate,
                "relocation_iteration": iteration,
                "accepted": True,
            }
        )
        accepted.append(row)
        current = candidate
    return current, accepted


@dataclass(frozen=True, slots=True)
class UniqueEnvelopeCandidate:
    candidate_name: str
    plan: PlacementPlan
    runs: dict[int, ServingRun]
    peak_target_rows_per_second: float
    peak_concurrency: int
    abstract_node_cost: float
    canonical_plan_sha256: str

    @property
    def peak_rows_per_second_per_abstract_cost(self) -> float:
        return self.peak_target_rows_per_second / self.abstract_node_cost


def evaluate_unique_candidate(
    engine: ServingEngine,
    inventory: Inventory,
    *,
    candidate_name: str,
    plan: PlacementPlan,
    run_cache: dict[tuple[str, int], ServingRun] | None = None,
) -> UniqueEnvelopeCandidate:
    e023 = E023Plan(plan.inventory_id, Arm.U_STRONG.value, clone_placement(plan))
    cache = run_cache if run_cache is not None else {}
    runs: dict[int, ServingRun] = {}
    for concurrency in CONCURRENCY_LEVELS:
        key = (e023.canonical_sha256, concurrency)
        if key not in cache:
            cache[key] = engine.run_closed_loop(e023, concurrency)
        run = cache[key]
        if run.status != "PASS":
            raise RuntimeError("MODEL_INVALID: unique baseline concurrency incomplete")
        runs[concurrency] = run
    peak = max(run.target_rows_per_second for run in runs.values())
    near = [
        concurrency
        for concurrency, run in runs.items()
        if abs(run.target_rows_per_second - peak) <= 1e-12
    ]
    peak_concurrency = min(near)
    return UniqueEnvelopeCandidate(
        candidate_name=candidate_name,
        plan=clone_placement(plan),
        runs=runs,
        peak_target_rows_per_second=peak,
        peak_concurrency=peak_concurrency,
        abstract_node_cost=abstract_node_cost(e023, inventory),
        canonical_plan_sha256=e023.canonical_sha256,
    )


def select_u_strong(
    candidates: list[UniqueEnvelopeCandidate],
) -> tuple[UniqueEnvelopeCandidate, str]:
    if not candidates:
        raise ValueError("U_STRONG envelope has no feasible unique candidate")
    fastest = min(
        candidates,
        key=lambda row: (-row.peak_target_rows_per_second, row.canonical_plan_sha256),
    )
    floor = float(FROZEN_CONSTANTS["economic_fastest_throughput_floor"])
    retained = [
        row
        for row in candidates
        if row.peak_target_rows_per_second
        >= floor * fastest.peak_target_rows_per_second
    ]
    selected = min(
        retained,
        key=lambda row: (
            -row.peak_rows_per_second_per_abstract_cost,
            -row.peak_target_rows_per_second,
            row.abstract_node_cost,
            row.canonical_plan_sha256,
        ),
    )
    return selected, fastest.candidate_name


__all__ = [
    "UniqueEnvelopeCandidate",
    "best_whole_layer_relocation",
    "clone_placement",
    "evaluate_unique_candidate",
    "frozen_placement_paths",
    "load_frozen_placement",
    "repair_whole_layer_relocations",
    "select_u_strong",
]
