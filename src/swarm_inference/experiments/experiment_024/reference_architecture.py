"""Controlled concentrated whole-layer reference packing."""

from __future__ import annotations

from dataclasses import dataclass

from swarm_inference.experiments.experiment_022.models import ModelGraph


@dataclass(frozen=True, slots=True)
class ReferencePacking:
    node_count: int
    layers_by_node: tuple[tuple[int, ...], ...]
    memory_used_by_node: tuple[int, ...]


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


__all__ = ["ReferencePacking", "contiguous_pack"]
