"""Canonical records for exact sparse expert optionality."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from swarm_inference.experiments.experiment_022.io import canonical_sha256
from swarm_inference.experiments.experiment_022.models import (
    Inventory,
    ModelGraph,
    PartitionKind,
    PlacementPlan,
)


class NetworkMode(StrEnum):
    LEGACY_DIRECTED_LINK = "LEGACY_DIRECTED_LINK"
    SHARED_NIC = "SHARED_NIC"


class Arm(StrEnum):
    U_STRONG = "U_STRONG"
    FLEX_FREE_NO_ALT = "FLEX_FREE_NO_ALT"
    FLEX_FREE = "FLEX_FREE"
    FLEX_POOL_NO_ALT = "FLEX_POOL_NO_ALT"
    FLEX_POOL = "FLEX_POOL"


@dataclass(frozen=True, slots=True)
class ExpertGroupReplica:
    """One exact alternate resident copy of a stateless P8 expert group."""

    layer_id: int
    logical_group_id: int
    primary_node_id: str
    alternate_node_id: str
    checkpoint_bytes: int
    resident_bytes: int
    persistent_state_bytes: int = 0

    def __post_init__(self) -> None:
        if self.layer_id <= 0:
            raise ValueError("layer 0 and invalid layer IDs cannot be replicated")
        if not 0 <= self.logical_group_id < 8:
            raise ValueError("P8 logical group index must be in [0, 7]")
        if not self.primary_node_id or not self.alternate_node_id:
            raise ValueError("replica requires concrete primary and alternate nodes")
        if self.primary_node_id == self.alternate_node_id:
            raise ValueError("alternate copy cannot share its primary node")
        if self.checkpoint_bytes <= 0 or self.resident_bytes <= 0:
            raise ValueError("replica memory must be positive")
        if self.persistent_state_bytes != 0:
            raise ValueError("stateless expert-group replica persistent state must be zero")

    def validate_partition(self, kind: PartitionKind, degree: int) -> None:
        if kind is not PartitionKind.WHOLE_EXPERT or degree != 8:
            raise ValueError("only stateless WHOLE_EXPERT:p8 groups may be replicated")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def base_resident_bytes(plan: PlacementPlan) -> dict[str, int]:
    """Concrete base-plan residency, including the common endpoint policy."""

    result: dict[str, int] = {}
    for node_id, value in plan.endpoint_memory_by_node.items():
        result[node_id] = result.get(node_id, 0) + int(value)
    for assignment in plan.assignments:
        for node_id, value in assignment.memory_by_node.items():
            result[node_id] = result.get(node_id, 0) + int(value)
    return result


def base_checkpoint_bytes(plan: PlacementPlan) -> dict[str, int]:
    result: dict[str, int] = {}
    for node_id, value in plan.endpoint_checkpoint_bytes_by_node.items():
        result[node_id] = result.get(node_id, 0) + int(value)
    for assignment in plan.assignments:
        for node_id, value in assignment.checkpoint_bytes_by_node.items():
            result[node_id] = result.get(node_id, 0) + int(value)
    return result


@dataclass(slots=True)
class E023Plan:
    """An E022 exact placement plus optional stateless expert-group replicas."""

    inventory_id: str
    arm: str
    base_plan: PlacementPlan
    replicas: tuple[ExpertGroupReplica, ...] = ()
    u_strong_used_nodes: tuple[str, ...] = ()
    planner_actions: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.inventory_id != self.base_plan.inventory_id:
            raise ValueError("E023 plan and base placement inventory differ")
        if not self.base_plan.feasible:
            raise ValueError("E023 plan requires a feasible base placement")
        known_arms = {value.value for value in Arm}
        if self.arm not in known_arms and not self.arm.startswith("DIAGNOSTIC_"):
            raise ValueError(f"unknown E023 arm {self.arm}")
        if self.arm in {
            Arm.U_STRONG.value,
            Arm.FLEX_FREE_NO_ALT.value,
            Arm.FLEX_POOL_NO_ALT.value,
        } and self.replicas:
            raise ValueError(f"{self.arm} cannot retain alternate replicas")

        assignments = {assignment.layer_id: assignment for assignment in self.base_plan.assignments}
        keys: set[tuple[int, int]] = set()
        for replica in self.replicas:
            key = (replica.layer_id, replica.logical_group_id)
            if key in keys:
                raise ValueError("duplicate alternate would create more than two copies")
            keys.add(key)
            try:
                assignment = assignments[replica.layer_id]
            except KeyError as exc:
                raise ValueError("replica layer is absent from base placement") from exc
            replica.validate_partition(assignment.partition_kind, assignment.degree)
            expected_primary = assignment.node_ids[replica.logical_group_id]
            if replica.primary_node_id != expected_primary:
                raise ValueError("replica primary does not match logical P8 group ownership")

        if self.arm == Arm.FLEX_FREE.value:
            allowed = set(self.u_strong_used_nodes)
            if not allowed:
                raise ValueError("FLEX_FREE requires frozen U_STRONG used nodes")
            introduced = self.used_nodes.difference(allowed)
            if introduced:
                raise ValueError(
                    "FLEX_FREE cannot activate a node outside U_STRONG: "
                    + ",".join(sorted(introduced))
                )

    @property
    def used_nodes(self) -> set[str]:
        nodes = set(base_resident_bytes(self.base_plan))
        nodes.update(replica.alternate_node_id for replica in self.replicas)
        return nodes

    @property
    def replica_count(self) -> int:
        return len(self.replicas)

    @property
    def replica_checkpoint_bytes(self) -> int:
        return sum(value.checkpoint_bytes for value in self.replicas)

    @property
    def replica_resident_bytes(self) -> int:
        return sum(value.resident_bytes for value in self.replicas)

    def replica_for(self, layer_id: int, logical_group_id: int) -> ExpertGroupReplica | None:
        for replica in self.replicas:
            if (replica.layer_id, replica.logical_group_id) == (layer_id, logical_group_id):
                return replica
        return None

    def canonical_payload(self) -> dict[str, Any]:
        assignments = [
            {
                "layer_id": value.layer_id,
                "partition_kind": value.partition_kind.value,
                "degree": value.degree,
                "node_ids": list(value.node_ids),
                "memory_by_node": dict(sorted(value.memory_by_node.items())),
                "checkpoint_bytes_by_node": dict(
                    sorted(value.checkpoint_bytes_by_node.items())
                ),
                "coordinator_node_id": value.coordinator_node_id,
                "candidate_id": value.candidate_id,
            }
            for value in sorted(self.base_plan.assignments, key=lambda row: row.layer_id)
        ]
        return {
            "inventory_id": self.inventory_id,
            "chunk_rows": self.base_plan.chunk_rows,
            "assignments": assignments,
            "endpoint_memory_by_node": dict(
                sorted(self.base_plan.endpoint_memory_by_node.items())
            ),
            "endpoint_checkpoint_bytes_by_node": dict(
                sorted(self.base_plan.endpoint_checkpoint_bytes_by_node.items())
            ),
            "replicas": [
                value.as_dict()
                for value in sorted(
                    self.replicas,
                    key=lambda row: (
                        row.layer_id,
                        row.logical_group_id,
                        row.alternate_node_id,
                    ),
                )
            ],
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(self.canonical_payload())

    def as_manifest(self, model: ModelGraph, inventory: Inventory) -> dict[str, Any]:
        base = self.base_plan.as_manifest(model, inventory)
        replica_by_node: dict[str, int] = {}
        replica_checkpoint_by_node: dict[str, int] = {}
        for replica in self.replicas:
            node_id = replica.alternate_node_id
            replica_by_node[node_id] = replica_by_node.get(node_id, 0) + replica.resident_bytes
            replica_checkpoint_by_node[node_id] = (
                replica_checkpoint_by_node.get(node_id, 0) + replica.checkpoint_bytes
            )
        for node in base["nodes"]:
            node_id = str(node["node_id"])
            base_bytes = int(node["assigned_memory_bytes"])
            replica_bytes = replica_by_node.get(node_id, 0)
            node["base_assigned_memory_bytes"] = base_bytes
            node["replica_resident_bytes"] = replica_bytes
            node["assigned_memory_bytes"] = base_bytes + replica_bytes
            node["replica_checkpoint_bytes"] = replica_checkpoint_by_node.get(node_id, 0)
        unique_checkpoint = int(base["checkpoint_reconciliation"]["assigned_checkpoint_bytes"])
        unique_reconciliation = dict(base["checkpoint_reconciliation"])
        duplicate_checkpoint = self.replica_checkpoint_bytes
        base.update(
            {
                "schema_version": "experiment-023-plan-manifest-v1",
                "experiment_id": "023",
                "arm": self.arm,
                "canonical_plan_sha256": self.canonical_sha256,
                "used_nodes": sorted(self.used_nodes),
                "u_strong_used_nodes": list(self.u_strong_used_nodes),
                "replicas": [
                    value.as_dict()
                    for value in sorted(
                        self.replicas,
                        key=lambda row: (row.layer_id, row.logical_group_id),
                    )
                ],
                "maximum_copies_per_logical_expert_group": 2,
                "canonical_reduction_order": list(range(8)),
                "planner_actions": list(self.planner_actions),
                "metadata": self.metadata,
                "checkpoint_reconciliation": {
                    **unique_reconciliation,
                    "unique_model_checkpoint_bytes": unique_checkpoint,
                    "duplicate_replica_checkpoint_bytes": duplicate_checkpoint,
                    "total_resident_checkpoint_bytes": (
                        unique_checkpoint + duplicate_checkpoint
                    ),
                    "model_checkpoint_bytes": model.checkpoint_payload_bytes,
                    "unique_gap_bytes": model.checkpoint_payload_bytes
                    - unique_checkpoint,
                },
            }
        )
        return base


def canonical_group_reduce(
    contributions: dict[int, np.ndarray],
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Reduce all eight logical slots in group order, never arrival order."""

    order = tuple(range(8))
    if set(contributions) != set(order):
        raise ValueError("exact P8 reduction requires logical group slots 0..7")
    values = [np.asarray(contributions[group]) for group in order]
    shape = values[0].shape
    if any(value.shape != shape for value in values):
        raise ValueError("logical expert-group contributions have inconsistent shapes")
    result = values[0].copy()
    for value in values[1:]:
        np.add(result, value, out=result)
    return result, order


__all__ = [
    "Arm",
    "E023Plan",
    "ExpertGroupReplica",
    "NetworkMode",
    "base_checkpoint_bytes",
    "base_resident_bytes",
    "canonical_group_reduce",
]
