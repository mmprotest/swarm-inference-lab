"""Standalone whole-expert-group and plan memory/cost accounting."""

from __future__ import annotations

from dataclasses import dataclass

from swarm_inference.experiments.experiment_022.model_graph import SHARD_BUFFER_BYTES
from swarm_inference.experiments.experiment_022.models import (
    Inventory,
    LayerSpec,
    split_integer,
)

from .models import E023Plan, base_checkpoint_bytes, base_resident_bytes


@dataclass(frozen=True, slots=True)
class StandaloneExpertGroupMemory:
    checkpoint_bytes: int
    resident_bytes: int
    persistent_state_bytes: int = 0


@dataclass(frozen=True, slots=True)
class MemoryReconciliationRow:
    node_id: str
    capacity_bytes: int
    base_resident_bytes: int
    replica_resident_bytes: int
    total_resident_bytes: int
    base_checkpoint_bytes: int
    duplicate_replica_checkpoint_bytes: int
    within_capacity: bool


def standalone_whole_expert_group_memory(
    layer: LayerSpec,
    group_index: int,
    degree: int = 8,
) -> StandaloneExpertGroupMemory:
    """Memory for routed experts only; coordinator fixed bytes are excluded."""

    if degree != 8:
        raise ValueError("E023 admits only WHOLE_EXPERT:p8 replica memory")
    if not 0 <= group_index < degree:
        raise ValueError("logical expert-group index is outside P8")
    routed = int(layer.component_bytes.get("routed_expert", 0))
    if routed <= 0:
        raise ValueError("layer has no routed experts to replicate")
    runtime_ratio = layer.resident_bytes / layer.checkpoint_bytes
    checkpoint_parts = split_integer(routed, degree)
    split_resident_total = round(routed * runtime_ratio)
    resident_parts = split_integer(split_resident_total, degree)
    return StandaloneExpertGroupMemory(
        checkpoint_bytes=checkpoint_parts[group_index],
        resident_bytes=resident_parts[group_index] + SHARD_BUFFER_BYTES,
        persistent_state_bytes=0,
    )


def memory_error_percent(estimated_bytes: int, physical_bytes: int) -> float:
    if estimated_bytes <= 0 or physical_bytes <= 0:
        raise ValueError("memory comparison requires positive byte counts")
    return 100.0 * abs(estimated_bytes - physical_bytes) / physical_bytes


def reconcile_plan_memory(
    plan: E023Plan,
    inventory: Inventory,
) -> tuple[MemoryReconciliationRow, ...]:
    """Prove concrete base plus replica residency fits without virtual VRAM."""

    nodes = inventory.node_map()
    base_resident = base_resident_bytes(plan.base_plan)
    base_checkpoint = base_checkpoint_bytes(plan.base_plan)
    replica_resident: dict[str, int] = {}
    replica_checkpoint: dict[str, int] = {}
    for replica in plan.replicas:
        node_id = replica.alternate_node_id
        if node_id not in nodes:
            raise ValueError(f"replica uses unknown inventory node {node_id}")
        replica_resident[node_id] = replica_resident.get(node_id, 0) + replica.resident_bytes
        replica_checkpoint[node_id] = (
            replica_checkpoint.get(node_id, 0) + replica.checkpoint_bytes
        )
    unknown_base = set(base_resident).difference(nodes)
    if unknown_base:
        raise ValueError("base placement uses unknown inventory nodes")

    rows: list[MemoryReconciliationRow] = []
    for node_id, node in sorted(nodes.items()):
        base = base_resident.get(node_id, 0)
        duplicate = replica_resident.get(node_id, 0)
        total = base + duplicate
        within = total <= node.accelerator_memory_bytes
        row = MemoryReconciliationRow(
            node_id=node_id,
            capacity_bytes=node.accelerator_memory_bytes,
            base_resident_bytes=base,
            replica_resident_bytes=duplicate,
            total_resident_bytes=total,
            base_checkpoint_bytes=base_checkpoint.get(node_id, 0),
            duplicate_replica_checkpoint_bytes=replica_checkpoint.get(node_id, 0),
            within_capacity=within,
        )
        rows.append(row)
        if not within:
            raise ValueError(
                f"node {node_id} capacity exceeded: {total} > "
                f"{node.accelerator_memory_bytes}"
            )
    return tuple(rows)


def abstract_node_cost(plan: E023Plan, inventory: Inventory) -> float:
    """Count each used physical node once, including replica-only nodes."""

    nodes = inventory.node_map()
    unknown = plan.used_nodes.difference(nodes)
    if unknown:
        raise ValueError("abstract cost cannot reconcile unknown used nodes")
    return sum(nodes[node_id].cost for node_id in sorted(plan.used_nodes))


__all__ = [
    "MemoryReconciliationRow",
    "StandaloneExpertGroupMemory",
    "abstract_node_cost",
    "memory_error_percent",
    "reconcile_plan_memory",
    "standalone_whole_expert_group_memory",
]
