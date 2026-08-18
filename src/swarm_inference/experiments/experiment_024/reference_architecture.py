"""Controlled concentrated whole-layer reference packing."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_022.io import canonical_sha256
from swarm_inference.experiments.experiment_022.models import ModelGraph

from .freeze import HIDDEN
from .placement import LayerPlacement
from .task_graph import ExecutionTopology, RuntimeNode

REFERENCE_ACCELERATOR_MEMORY_BYTES = 46_627_028_992
CATALOG_RELATIVE_PATH = Path(
    "artifacts/experiment-022/completion/rerun/candidate-catalog.json"
)
MODEL_METADATA_RELATIVE_PATH = Path("artifacts/experiment-022/model-metadata.json")
REFERENCE_TRANSIENT_BYTES = 4 * HIDDEN * 4


@dataclass(frozen=True, slots=True)
class ReferencePacking:
    node_count: int
    layers_by_node: tuple[tuple[int, ...], ...]
    memory_used_by_node: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ReferenceArchitecture:
    node_count: int
    nodes: tuple[RuntimeNode, ...]
    endpoint_node_ids: tuple[str, ...]
    endpoint_memory_by_node: dict[str, int]
    assignments: tuple[LayerPlacement, ...]
    memory_used_by_node: dict[str, int]

    @property
    def architecture_sha256(self) -> str:
        return canonical_sha256(self.as_dict(include_hash=False))

    def as_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": "experiment-024-concentrated-reference-v2",
            "architecture": "CONCENTRATED_FAST",
            "node_count": self.node_count,
            "accelerator_memory_bytes_per_node": REFERENCE_ACCELERATOR_MEMORY_BYTES,
            "compute_multiplier": 1.0,
            "network": {
                "name": "FAST_FABRIC",
                "latency_ms": 0.25,
                "bandwidth_gbps": 25.0,
                "software_overhead_ms": 0.04,
            },
            "endpoint_node_ids": list(self.endpoint_node_ids),
            "endpoint_memory_by_node": self.endpoint_memory_by_node,
            "assignments": [asdict(assignment) for assignment in self.assignments],
            "memory_used_by_node": self.memory_used_by_node,
            "whole_layer_transformer_layer_ids": list(range(93)),
        }
        if include_hash:
            value["architecture_sha256"] = self.architecture_sha256
        return value

    def topology(self) -> ExecutionTopology:
        return ExecutionTopology(
            architecture="CONCENTRATED_FAST",
            nodes=self.nodes,
            endpoint_node_ids=self.endpoint_node_ids,
            assignments=self.assignments,
            commodity_scenario=None,
        )


def contiguous_pack(
    model: ModelGraph,
    *,
    node_count: int,
    accelerator_memory_bytes: int,
    endpoint_memory_by_node: tuple[int, ...],
) -> ReferencePacking | None:
    """Pack layers 0..92 once, never returning to a previous node."""

    if node_count <= 0 or len(endpoint_memory_by_node) != node_count:
        raise ValueError("invalid reference-node definition")
    used = list(endpoint_memory_by_node)
    layers: list[list[int]] = [[] for _ in range(node_count)]
    node = 0
    for layer in model.layers:
        while node < node_count and used[node] + layer.resident_bytes > accelerator_memory_bytes:
            node += 1
        if node == node_count:
            return None
        layers[node].append(layer.layer_id)
        used[node] += layer.resident_bytes
    return ReferencePacking(
        node_count=node_count,
        layers_by_node=tuple(tuple(values) for values in layers),
        memory_used_by_node=tuple(used),
    )


def _split_integer(total: int, count: int) -> tuple[int, ...]:
    bounds = [(total * index) // count for index in range(count + 1)]
    return tuple(bounds[index + 1] - bounds[index] for index in range(count))


def build_reference_architecture(repo_root: Path) -> ReferenceArchitecture:
    """Find the minimum deterministic contiguous packing on the frozen class."""

    repo_root = repo_root.resolve()
    catalog = json.loads(
        (repo_root / CATALOG_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (repo_root / MODEL_METADATA_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    whole = {
        int(row["layer"]): row
        for row in catalog["candidates"]
        if row["candidate_type"] == "WHOLE_LAYER" and int(row["degree"]) == 1
    }
    if set(whole) != set(range(93)):
        raise RuntimeError("reference whole-layer catalog coverage changed")
    endpoint_count = 4
    endpoint_parts = _split_integer(int(metadata["endpoint_resident_bytes"]), 4)
    for node_count in range(endpoint_count, 94):
        nodes = tuple(
            RuntimeNode(f"reference-{index:03d}", index, 1.0)
            for index in range(node_count)
        )
        endpoint_memory = {
            nodes[index].node_id: endpoint_parts[index]
            for index in range(endpoint_count)
        }
        used = {
            node.node_id: endpoint_memory.get(node.node_id, 0) for node in nodes
        }
        assignments: list[LayerPlacement] = []
        node_index = 0
        feasible = True
        for layer in range(93):
            candidate = whole[layer]
            resident = int(candidate["resident_memory_bytes"][0])
            while (
                node_index < node_count
                and used[nodes[node_index].node_id]
                + resident
                + REFERENCE_TRANSIENT_BYTES
                > REFERENCE_ACCELERATOR_MEMORY_BYTES
            ):
                node_index += 1
            if node_index == node_count:
                feasible = False
                break
            node_id = nodes[node_index].node_id
            used[node_id] += resident
            assignments.append(
                LayerPlacement(
                    layer=layer,
                    candidate_id=str(candidate["candidate_id"]),
                    candidate_type="WHOLE_LAYER",
                    degree=1,
                    node_ids=(node_id,),
                    resident_memory_bytes=(resident,),
                    checkpoint_bytes=(int(candidate["checkpoint_bytes"][0]),),
                    coordinator_node_id=node_id,
                )
            )
        if feasible:
            return ReferenceArchitecture(
                node_count=node_count,
                nodes=nodes,
                endpoint_node_ids=tuple(
                    nodes[index].node_id for index in range(endpoint_count)
                ),
                endpoint_memory_by_node=endpoint_memory,
                assignments=tuple(assignments),
                memory_used_by_node=used,
            )
    raise RuntimeError("concentrated reference cannot be packed on 93 nodes")


__all__ = [
    "REFERENCE_ACCELERATOR_MEMORY_BYTES",
    "ReferenceArchitecture",
    "ReferencePacking",
    "build_reference_architecture",
    "contiguous_pack",
]
