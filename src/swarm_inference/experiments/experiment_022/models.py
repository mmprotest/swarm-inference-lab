"""Canonical resource, model, candidate, and placement records for E022."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class LayerType(StrEnum):
    DENSE = "DENSE"
    KDA = "KDA"
    GATED_MLA = "GATED_MLA"


class PartitionKind(StrEnum):
    WHOLE_LAYER = "WHOLE_LAYER"
    WHOLE_EXPERT = "WHOLE_EXPERT"
    EXPERT_SHARD = "EXPERT_SHARD"
    ATTENTION_PROJECTION_SHARD = "ATTENTION_PROJECTION_SHARD"
    FULL_MIXED_STRIPE = "FULL_MIXED_STRIPE"


class PlannerLevel(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"


ALLOWED_BY_LEVEL: dict[PlannerLevel, frozenset[PartitionKind]] = {
    PlannerLevel.A: frozenset({PartitionKind.WHOLE_LAYER}),
    PlannerLevel.B: frozenset(
        {PartitionKind.WHOLE_LAYER, PartitionKind.WHOLE_EXPERT}
    ),
    PlannerLevel.C: frozenset(
        {
            PartitionKind.WHOLE_LAYER,
            PartitionKind.WHOLE_EXPERT,
            PartitionKind.EXPERT_SHARD,
        }
    ),
    PlannerLevel.D: frozenset(
        {
            PartitionKind.WHOLE_LAYER,
            PartitionKind.WHOLE_EXPERT,
            PartitionKind.EXPERT_SHARD,
            PartitionKind.ATTENTION_PROJECTION_SHARD,
        }
    ),
    PlannerLevel.E: frozenset(PartitionKind),
}


@dataclass(frozen=True, slots=True)
class NetworkPeer:
    peer_id: str
    latency_ms: float
    bandwidth_gbps: float
    software_overhead_ms: float = 0.04
    locality_class: str = "unknown"

    def __post_init__(self) -> None:
        if not self.peer_id:
            raise ValueError("network peer requires an identifier")
        if self.latency_ms < 0 or self.bandwidth_gbps <= 0:
            raise ValueError("network latency/bandwidth must be physical values")
        if self.software_overhead_ms < 0:
            raise ValueError("software overhead cannot be negative")

    def transfer_ms(self, payload_bytes: int) -> float:
        if payload_bytes < 0:
            raise ValueError("payload bytes cannot be negative")
        serialization_ms = payload_bytes * 8 / (self.bandwidth_gbps * 1_000_000)
        return self.latency_ms / 2 + serialization_ms + self.software_overhead_ms


@dataclass(frozen=True, slots=True)
class NodeCapability:
    node_id: str
    accelerator_memory_bytes: int
    system_memory_bytes: int
    compute_profile: dict[str, float]
    memory_bandwidth_profile: dict[str, float]
    supported_precisions: tuple[str, ...]
    network_peers: dict[str, NetworkPeer]
    reliability: float
    cost: float
    cached_shards: tuple[str, ...]
    runtime_capabilities: tuple[str, ...]
    locality_group: str
    available: bool = True

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node capability requires node_id")
        if self.accelerator_memory_bytes <= 0 or self.system_memory_bytes <= 0:
            raise ValueError("node memory capacities must be positive")
        if not 0 < self.reliability <= 1:
            raise ValueError("reliability must be in (0, 1]")
        if self.cost < 0:
            raise ValueError("cost cannot be negative")
        if not self.compute_profile:
            raise ValueError("compute_profile cannot be empty")
        for name, multiplier in self.compute_profile.items():
            if not 0 < multiplier <= 1.0:
                raise ValueError(
                    f"controlled compute multiplier {name}={multiplier} must be in (0, 1]"
                )
        if "reference" not in self.compute_profile:
            raise ValueError("compute_profile requires a reference multiplier")

    @property
    def compute_multiplier(self) -> float:
        return float(self.compute_profile["reference"])

    def peer(self, other: str) -> NetworkPeer:
        if other == self.node_id:
            return NetworkPeer(other, 0.0, 1_000_000.0, 0.0, "local")
        try:
            return self.network_peers[other]
        except KeyError as exc:
            raise KeyError(f"{self.node_id} has no measured/shaped peer link to {other}") from exc

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["network_peers"] = {
            key: asdict(peer) for key, peer in sorted(self.network_peers.items())
        }
        value["supported_precisions"] = list(self.supported_precisions)
        value["cached_shards"] = list(self.cached_shards)
        value["runtime_capabilities"] = list(self.runtime_capabilities)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> NodeCapability:
        peers = {
            str(key): NetworkPeer(**peer)
            for key, peer in dict(value.get("network_peers", {})).items()
        }
        return cls(
            node_id=str(value["node_id"]),
            accelerator_memory_bytes=int(value["accelerator_memory_bytes"]),
            system_memory_bytes=int(value["system_memory_bytes"]),
            compute_profile={
                str(key): float(item)
                for key, item in dict(value["compute_profile"]).items()
            },
            memory_bandwidth_profile={
                str(key): float(item)
                for key, item in dict(value["memory_bandwidth_profile"]).items()
            },
            supported_precisions=tuple(value["supported_precisions"]),
            network_peers=peers,
            reliability=float(value["reliability"]),
            cost=float(value["cost"]),
            cached_shards=tuple(value.get("cached_shards", ())),
            runtime_capabilities=tuple(value["runtime_capabilities"]),
            locality_group=str(value["locality_group"]),
            available=bool(value.get("available", True)),
        )


@dataclass(frozen=True, slots=True)
class LayerSpec:
    layer_id: int
    layer_type: LayerType
    checkpoint_bytes: int
    resident_bytes: int
    component_bytes: dict[str, int]
    tensor_count: int
    attnres_snapshot: bool

    def __post_init__(self) -> None:
        if self.layer_id < 0 or self.checkpoint_bytes <= 0 or self.resident_bytes <= 0:
            raise ValueError("invalid K3 layer geometry")
        if sum(self.component_bytes.values()) != self.checkpoint_bytes:
            raise ValueError("layer components must reconcile to checkpoint bytes")


@dataclass(frozen=True, slots=True)
class ModelGraph:
    model_id: str
    checkpoint_index_sha256: str
    checkpoint_payload_bytes: int
    layers: tuple[LayerSpec, ...]
    endpoint_checkpoint_bytes: int
    endpoint_resident_bytes: int
    tensor_count: int

    def __post_init__(self) -> None:
        if len(self.layers) != 93:
            raise ValueError("Experiment 022 requires the complete 93-layer Kimi K3 graph")
        covered = sum(layer.checkpoint_bytes for layer in self.layers)
        if covered + self.endpoint_checkpoint_bytes != self.checkpoint_payload_bytes:
            raise ValueError("model graph does not reconcile to checkpoint payload")

    @property
    def total_resident_bytes(self) -> int:
        return self.endpoint_resident_bytes + sum(
            layer.resident_bytes for layer in self.layers
        )

    @property
    def maximum_layer_resident_bytes(self) -> int:
        return max(layer.resident_bytes for layer in self.layers)


@dataclass(frozen=True, slots=True)
class Inventory:
    inventory_id: str
    family: str
    seed: int
    nodes: tuple[NodeCapability, ...]
    evidence_class: str
    generator_version: str
    scenario: str

    def __post_init__(self) -> None:
        identifiers = [node.node_id for node in self.nodes]
        if not self.inventory_id or len(identifiers) != len(set(identifiers)):
            raise ValueError("inventory node identifiers must be unique")
        expected = set(identifiers)
        for node in self.nodes:
            missing = expected.difference({node.node_id}, node.network_peers)
            if missing:
                raise ValueError(f"node {node.node_id} lacks {len(missing)} peer links")

    @property
    def available_nodes(self) -> tuple[NodeCapability, ...]:
        return tuple(node for node in self.nodes if node.available)

    def node_map(self) -> dict[str, NodeCapability]:
        return {node.node_id: node for node in self.nodes}

    def as_dict(self) -> dict[str, Any]:
        return {
            "inventory_id": self.inventory_id,
            "family": self.family,
            "seed": self.seed,
            "nodes": [node.as_dict() for node in self.nodes],
            "evidence_class": self.evidence_class,
            "generator_version": self.generator_version,
            "scenario": self.scenario,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Inventory:
        return cls(
            inventory_id=str(value["inventory_id"]),
            family=str(value["family"]),
            seed=int(value["seed"]),
            nodes=tuple(NodeCapability.from_dict(node) for node in value["nodes"]),
            evidence_class=str(value["evidence_class"]),
            generator_version=str(value["generator_version"]),
            scenario=str(value["scenario"]),
        )


@dataclass(frozen=True, slots=True)
class LayerAssignment:
    layer_id: int
    partition_kind: PartitionKind
    degree: int
    node_ids: tuple[str, ...]
    memory_by_node: dict[str, int]
    checkpoint_bytes_by_node: dict[str, int]
    coordinator_node_id: str
    candidate_id: str

    def __post_init__(self) -> None:
        if self.degree != len(self.node_ids) or self.degree <= 0:
            raise ValueError("assignment degree must equal concrete node count")
        if self.coordinator_node_id not in self.node_ids:
            raise ValueError("assignment coordinator must be a concrete worker")
        if set(self.memory_by_node) != set(self.node_ids):
            raise ValueError("every assignment worker requires memory accounting")
        if set(self.checkpoint_bytes_by_node) != set(self.node_ids):
            raise ValueError("every assignment worker requires checkpoint ownership")


@dataclass(slots=True)
class PlacementPlan:
    inventory_id: str
    planner_level: PlannerLevel
    chunk_rows: int
    assignments: list[LayerAssignment]
    endpoint_memory_by_node: dict[str, int]
    endpoint_checkpoint_bytes_by_node: dict[str, int]
    feasible: bool
    infeasible_reason: str | None = None
    exact_tok_s_per_user: float | None = None
    critical_path_ms: float | None = None
    total_worker_compute_ms: float | None = None
    network_bytes: int = 0
    serial_waits: int = 0
    messages: int = 0
    used_nodes: tuple[str, ...] = ()
    memory_used_by_node: dict[str, int] = field(default_factory=dict)
    worker_utilization: dict[str, float] = field(default_factory=dict)
    worker_seconds_per_token: float | None = None
    objective_tuple: tuple[float, float, float, float, float] | None = None
    event_receipt: dict[str, Any] | None = None
    optimizer_seed: int = 0
    optimizer_evaluations: int = 0

    def verify_dominance_fallback(self, fallback: PlacementPlan | None) -> None:
        if fallback is None or not fallback.feasible or not self.feasible:
            return
        if self.exact_tok_s_per_user is None or fallback.exact_tok_s_per_user is None:
            raise ValueError("feasible plans require throughput")
        if self.exact_tok_s_per_user + 1e-12 < fallback.exact_tok_s_per_user * 0.99:
            raise ValueError("adaptive plan violated the explicit whole-layer fallback")

    def usage_counts(self) -> dict[str, int]:
        values = {kind.value: 0 for kind in PartitionKind}
        for assignment in self.assignments:
            values[assignment.partition_kind.value] += 1
        return values

    def as_manifest(self, model: ModelGraph, inventory: Inventory) -> dict[str, Any]:
        nodes = inventory.node_map()
        pieces: dict[str, list[dict[str, Any]]] = {node: [] for node in nodes}
        for node_id, memory in self.endpoint_memory_by_node.items():
            pieces[node_id].append(
                {
                    "piece": "common_non_transformer",
                    "partition_type": "IDENTICAL_ENDPOINT_POLICY",
                    "resident_memory_bytes": memory,
                    "checkpoint_bytes": self.endpoint_checkpoint_bytes_by_node[node_id],
                    "cached": "common_non_transformer" in nodes[node_id].cached_shards,
                    "network_dependencies": [],
                }
            )
        for assignment in self.assignments:
            for node_id in assignment.node_ids:
                peers = [other for other in assignment.node_ids if other != node_id]
                pieces[node_id].append(
                    {
                        "piece": f"transformer_layer_{assignment.layer_id:02d}",
                        "candidate_id": assignment.candidate_id,
                        "partition_type": assignment.partition_kind.value,
                        "degree": assignment.degree,
                        "resident_memory_bytes": assignment.memory_by_node[node_id],
                        "checkpoint_bytes": assignment.checkpoint_bytes_by_node[node_id],
                        "coordinator": node_id == assignment.coordinator_node_id,
                        "cached": assignment.candidate_id in nodes[node_id].cached_shards,
                        "network_dependencies": peers,
                    }
                )
        manifest_nodes = []
        for node_id, node in sorted(nodes.items()):
            assigned = pieces[node_id]
            manifest_nodes.append(
                {
                    "node_id": node_id,
                    "available_memory_bytes": node.accelerator_memory_bytes,
                    "assigned_memory_bytes": sum(
                        int(piece["resident_memory_bytes"]) for piece in assigned
                    ),
                    "compute_multiplier": node.compute_multiplier,
                    "locality_group": node.locality_group,
                    "pieces": assigned,
                }
            )
        assigned_checkpoint = sum(self.endpoint_checkpoint_bytes_by_node.values()) + sum(
            sum(value.checkpoint_bytes_by_node.values()) for value in self.assignments
        )
        return {
            "schema_version": "experiment-022-placement-manifest-v1",
            "inventory_id": self.inventory_id,
            "planner_level": self.planner_level.value,
            "feasible": self.feasible,
            "chunk_rows": self.chunk_rows,
            "objective": {
                "exact_tok_s_per_user": self.exact_tok_s_per_user,
                "critical_path_ms": self.critical_path_ms,
                "total_worker_compute_ms": self.total_worker_compute_ms,
                "network_bytes": self.network_bytes,
                "workers_used": len(self.used_nodes),
            },
            "checkpoint_reconciliation": {
                "model_checkpoint_bytes": model.checkpoint_payload_bytes,
                "assigned_checkpoint_bytes": assigned_checkpoint,
                "gap_bytes": model.checkpoint_payload_bytes - assigned_checkpoint,
                "overlap_bytes": 0,
            },
            "nodes": manifest_nodes,
        }


def split_integer(total: int, count: int) -> tuple[int, ...]:
    if total < 0 or count <= 0:
        raise ValueError("invalid integer split")
    bounds = [(total * index) // count for index in range(count + 1)]
    result = tuple(bounds[index + 1] - bounds[index] for index in range(count))
    if sum(result) != total or max(result, default=0) - min(result, default=0) > 1:
        raise RuntimeError("integer split did not reconcile")
    return result


def finite_or_none(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


__all__ = [
    "ALLOWED_BY_LEVEL",
    "Inventory",
    "LayerAssignment",
    "LayerSpec",
    "LayerType",
    "ModelGraph",
    "NetworkPeer",
    "NodeCapability",
    "PartitionKind",
    "PlacementPlan",
    "PlannerLevel",
    "finite_or_none",
    "split_integer",
]
