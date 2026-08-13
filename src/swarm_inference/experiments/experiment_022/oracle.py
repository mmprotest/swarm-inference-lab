"""Independent exact enumeration for reduced E022 placement problems."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Any

from .evaluator import PlacementEvaluator, common_endpoint_policy
from .model_graph import candidate_memory
from .models import (
    ALLOWED_BY_LEVEL,
    Inventory,
    LayerAssignment,
    LayerSpec,
    ModelGraph,
    PartitionKind,
    PlacementPlan,
    PlannerLevel,
)
from .planner import OptimizerConfiguration, SharedPlacementOptimizer
from .service import ResidentServiceModel


@dataclass(frozen=True, slots=True)
class _ReducedModel:
    model_id: str
    layers: tuple[LayerSpec, ...]
    endpoint_resident_bytes: int
    endpoint_checkpoint_bytes: int


def _reduced(model: ModelGraph, layer_count: int) -> _ReducedModel:
    return _ReducedModel(
        model_id=f"{model.model_id}:reduced-{layer_count}",
        layers=model.layers[:layer_count],
        endpoint_resident_bytes=model.endpoint_resident_bytes,
        endpoint_checkpoint_bytes=model.endpoint_checkpoint_bytes,
    )


def _candidate_assignments(
    model: _ReducedModel,
    inventory: Inventory,
    service: ResidentServiceModel,
    layer_id: int,
    level: PlannerLevel,
) -> list[LayerAssignment]:
    layer = model.layers[layer_id]
    result: list[LayerAssignment] = []
    for kind in ALLOWED_BY_LEVEL[level]:
        if layer_id == 0 and kind is not PartitionKind.WHOLE_LAYER:
            continue
        for degree in ((1,) if kind is PartitionKind.WHOLE_LAYER else (2, 4, 8, 16)):
            if degree > len(inventory.available_nodes):
                continue
            if not service.supported_candidate(layer, kind, degree, 1):
                continue
            resident, checkpoint = candidate_memory(layer, kind, degree)
            required = "WHOLE_LAYER" if kind is PartitionKind.WHOLE_LAYER else (
                "EXPERT_SHARD"
                if kind in {PartitionKind.EXPERT_SHARD, PartitionKind.FULL_MIXED_STRIPE}
                else "PROJECTION_SHARD"
                if kind is PartitionKind.ATTENTION_PROJECTION_SHARD
                else "WHOLE_EXPERT"
            )
            eligible = [
                node
                for node in inventory.available_nodes
                if required in node.runtime_capabilities
            ]
            for nodes in itertools.permutations(eligible, degree):
                if any(
                    value > node.accelerator_memory_bytes
                    for node, value in zip(nodes, resident, strict=True)
                ):
                    continue
                node_ids = tuple(node.node_id for node in nodes)
                result.append(
                    LayerAssignment(
                        layer_id=layer_id,
                        partition_kind=kind,
                        degree=degree,
                        node_ids=node_ids,
                        memory_by_node=dict(zip(node_ids, resident, strict=True)),
                        checkpoint_bytes_by_node=dict(zip(node_ids, checkpoint, strict=True)),
                        coordinator_node_id=node_ids[0],
                        candidate_id=f"layer-{layer_id:02d}:{kind.value}:p{degree}",
                    )
                )
    return result


def _enumerate(
    model: _ReducedModel,
    inventory: Inventory,
    service: ResidentServiceModel,
    level: PlannerLevel,
) -> tuple[PlacementPlan, int, float]:
    endpoint = common_endpoint_policy(model, inventory)  # type: ignore[arg-type]
    if endpoint is None:
        raise RuntimeError("reduced oracle endpoint is infeasible")
    evaluator = PlacementEvaluator(model, inventory, service, endpoint)  # type: ignore[arg-type]
    options = [
        _candidate_assignments(model, inventory, service, layer, level)
        for layer in range(len(model.layers))
    ]
    if any(not rows for rows in options):
        raise RuntimeError("reduced oracle has an unassignable layer")
    capacities = {
        node.node_id: node.accelerator_memory_bytes for node in inventory.available_nodes
    }
    best: PlacementPlan | None = None
    evaluated = 0
    started = time.perf_counter_ns()

    def visit(
        layer: int,
        assignments: list[LayerAssignment],
        memory: dict[str, int],
    ) -> None:
        nonlocal best, evaluated
        if layer == len(model.layers):
            plan = PlacementPlan(
                inventory_id=inventory.inventory_id,
                planner_level=level,
                chunk_rows=1,
                assignments=list(assignments),
                endpoint_memory_by_node=dict(endpoint.memory_by_node),
                endpoint_checkpoint_bytes_by_node=dict(endpoint.checkpoint_bytes_by_node),
                feasible=True,
                memory_used_by_node={node: value for node, value in memory.items() if value},
            )
            evaluator.evaluate(plan)
            evaluated += 1
            if best is None or (plan.objective_tuple or ()) > (best.objective_tuple or ()):
                best = plan
            return
        for assignment in options[layer]:
            updated = dict(memory)
            feasible = True
            for node, value in assignment.memory_by_node.items():
                updated[node] = updated.get(node, 0) + value
                if updated[node] > capacities[node]:
                    feasible = False
                    break
            if feasible:
                visit(layer + 1, [*assignments, assignment], updated)

    visit(0, [], dict(endpoint.memory_by_node))
    if best is None:
        raise RuntimeError("reduced oracle enumerated no feasible placements")
    return best, evaluated, (time.perf_counter_ns() - started) / 1e6


def validate_optimizer_oracle(
    model: ModelGraph,
    inventory: Inventory,
    service: ResidentServiceModel,
    configuration: OptimizerConfiguration,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases = (("whole-four-layer", 4, PlannerLevel.A), ("mixed-two-layer", 2, PlannerLevel.E))
    # Keep exactly four concrete nodes. Their peer advertisements remain valid
    # because extra peer records do not create aggregate compute resources.
    reduced_inventory = Inventory(
        inventory_id=f"{inventory.inventory_id}:oracle-4nodes",
        family="optimizer-oracle",
        seed=inventory.seed,
        nodes=inventory.available_nodes[:4],
        evidence_class=inventory.evidence_class,
        generator_version=inventory.generator_version,
        scenario="reduced exact-enumeration oracle",
    )
    for case, layer_count, level in cases:
        reduced_model = _reduced(model, layer_count)
        exact, enumeration_count, exact_ms = _enumerate(
            reduced_model, reduced_inventory, service, level
        )
        optimizer = SharedPlacementOptimizer(
            reduced_model,  # type: ignore[arg-type]
            service,
            configuration,
        )
        heuristic = optimizer.optimize(reduced_inventory, level)
        exact_value = float(exact.exact_tok_s_per_user or 0)
        heuristic_value = float(heuristic.plan.exact_tok_s_per_user or 0)
        gap = (exact_value - heuristic_value) / exact_value * 100.0
        rows.append(
            {
                "case": case,
                "planner_level": level.value,
                "layers": layer_count,
                "nodes": 4,
                "enumerated_feasible_placements": enumeration_count,
                "exact_tok_s_per_user": exact_value,
                "optimizer_tok_s_per_user": heuristic_value,
                "objective_gap_percent": max(0.0, gap),
                "exact_wall_ms": exact_ms,
                "optimizer_wall_ms": heuristic.elapsed_ms,
                "optimizer_proposals": heuristic.proposals,
                "optimizer_exact_evaluations": heuristic.exact_evaluations,
                "status": "PASS" if gap <= 1.0 + 1e-12 else "FAIL",
            }
        )
    return rows


__all__ = ["validate_optimizer_oracle"]
