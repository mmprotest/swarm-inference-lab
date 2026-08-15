"""Checkpoint-derived Kimi K3 graph and candidate-memory accounting."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_019.checkpoint import CheckpointCatalog

from .models import LayerSpec, LayerType, ModelGraph, PartitionKind, split_integer

TRANSFORMER_LAYERS = 93
RESIDUAL_BLOCK = 12
SHARD_BUFFER_BYTES = 32 * 1024 * 1024


def _resident_measurements(path: Path) -> dict[int, int]:
    if not path.is_file():
        return {}
    values: dict[int, list[int]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["chunk_rows"]) != 1:
                continue
            values[int(row["layer"])].append(int(row["resident_device_bytes"]))
    return {layer: round(statistics.median(rows)) for layer, rows in values.items()}


def _component_name(role: str, name: str) -> str:
    if role == "routed_expert":
        return "routed_expert"
    if role == "attention":
        if name.endswith(
            (
                ".f_a_proj.weight",
                ".q_a_proj.weight",
                ".q_a_layernorm.weight",
                ".kv_a_proj_with_mqa.weight",
            )
        ):
            return "attention_common"
        if name.endswith((".o_norm.weight", ".kv_a_layernorm.weight")):
            return "attention_replicated"
        return "attention_shardable"
    if role == "shared_expert":
        return "shared_expert"
    if role == "latent_moe_projection":
        return "projection"
    if role == "router":
        return "router"
    if role == "dense_mlp":
        return "dense_mlp"
    if role == "attnres":
        return "attnres"
    return "other"


def build_model_graph(
    checkpoint: Path,
    *,
    whole_layer_service_csv: Path,
) -> ModelGraph:
    """Create the complete model graph from the local checkpoint index.

    Runtime memory is grounded in E018's persistent whole-layer measurements.
    Unmeasured layers use the median measured runtime/checkpoint ratio for their
    exact attention class; this is a memory estimate, never a service-time input.
    """

    catalog = CheckpointCatalog(checkpoint)
    records = catalog.records()
    by_layer: dict[int, list[Any]] = defaultdict(list)
    endpoint = []
    for record in records.values():
        if record.layer_id is None:
            endpoint.append(record)
        else:
            by_layer[record.layer_id].append(record)
    if set(by_layer) != set(range(TRANSFORMER_LAYERS)):
        raise ValueError("checkpoint does not contain exactly layers 0..92")
    measured = _resident_measurements(whole_layer_service_csv)
    types: dict[int, LayerType] = {}
    checkpoint_bytes: dict[int, int] = {}
    for layer in range(TRANSFORMER_LAYERS):
        names = {record.name for record in by_layer[layer]}
        if layer == 0:
            layer_type = LayerType.DENSE
        elif f"language_model.model.layers.{layer}.self_attn.q_proj.weight" in names:
            layer_type = LayerType.KDA
        else:
            layer_type = LayerType.GATED_MLA
        types[layer] = layer_type
        checkpoint_bytes[layer] = sum(record.byte_size for record in by_layer[layer])
    factors: dict[LayerType, list[float]] = defaultdict(list)
    for layer, resident in measured.items():
        if layer in checkpoint_bytes:
            factors[types[layer]].append(resident / checkpoint_bytes[layer])
    general_factors = [value for rows in factors.values() for value in rows]
    fallback_factor = statistics.median(general_factors) if general_factors else 1.0
    type_factor = {
        layer_type: statistics.median(values) if values else fallback_factor
        for layer_type, values in factors.items()
    }
    type_factor.setdefault(LayerType.DENSE, fallback_factor)
    layers: list[LayerSpec] = []
    for layer in range(TRANSFORMER_LAYERS):
        components: dict[str, int] = defaultdict(int)
        for record in by_layer[layer]:
            components[_component_name(record.role, record.name)] += record.byte_size
        resident = measured.get(
            layer,
            round(checkpoint_bytes[layer] * type_factor[types[layer]]),
        )
        layers.append(
            LayerSpec(
                layer_id=layer,
                layer_type=types[layer],
                checkpoint_bytes=checkpoint_bytes[layer],
                resident_bytes=resident,
                component_bytes=dict(sorted(components.items())),
                tensor_count=len(by_layer[layer]),
                attnres_snapshot=bool(layer and layer % RESIDUAL_BLOCK == 0),
            )
        )
    payload = sum(record.byte_size for record in records.values())
    endpoint_bytes = sum(record.byte_size for record in endpoint)
    # Endpoint policy is identical for both planners.  The text path retains
    # checkpoint representation plus bounded buffers; it is not used to create
    # an artificial compute advantage for either action space.
    endpoint_resident = endpoint_bytes + 4 * SHARD_BUFFER_BYTES
    return ModelGraph(
        model_id="moonshotai/Kimi-K3",
        checkpoint_index_sha256=catalog.index_sha256,
        checkpoint_payload_bytes=payload,
        layers=tuple(layers),
        endpoint_checkpoint_bytes=endpoint_bytes,
        endpoint_resident_bytes=endpoint_resident,
        tensor_count=len(records),
    )


def candidate_memory(
    layer: LayerSpec,
    kind: PartitionKind,
    degree: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return resident and checkpoint bytes for each concrete candidate worker."""

    if kind is PartitionKind.WHOLE_LAYER:
        if degree != 1:
            raise ValueError("whole layer candidates have degree one")
        return (layer.resident_bytes,), (layer.checkpoint_bytes,)
    if degree not in (2, 4, 8, 16):
        raise ValueError("validated sub-layer degrees are 2/4/8/16")
    routed = int(layer.component_bytes.get("routed_expert", 0))
    attention_shardable = int(
        layer.component_bytes.get("attention_shardable", 0)
    )
    attention_replicated = int(
        layer.component_bytes.get("attention_replicated", 0)
    )
    projection = int(layer.component_bytes.get("projection", 0))
    split_checkpoint: int
    shared = int(layer.component_bytes.get("shared_expert", 0))
    if kind in (PartitionKind.WHOLE_EXPERT, PartitionKind.EXPERT_SHARD):
        split_checkpoint = routed
    elif kind is PartitionKind.ATTENTION_PROJECTION_SHARD:
        # KDA/MLA projection matrices are classified as attention tensors in
        # the checkpoint. `projection` is the routed latent MoE down/up path
        # and remains whole in this selective candidate.
        split_checkpoint = attention_shardable
    elif kind is PartitionKind.FULL_MIXED_STRIPE:
        split_checkpoint = routed + attention_shardable + projection + shared
    else:
        raise ValueError(f"unsupported candidate {kind}")
    fixed_checkpoint = layer.checkpoint_bytes - split_checkpoint
    split_checkpoint_values = split_integer(split_checkpoint, degree)
    checkpoint = list(split_checkpoint_values)
    checkpoint[0] += fixed_checkpoint
    runtime_ratio = layer.resident_bytes / layer.checkpoint_bytes
    # Buffers are owned by concrete workers.  The coordinator's fixed work is
    # not divided, and every participating worker carries one bounded reusable
    # buffer.  No abstract aggregate memory resource exists.
    split_resident = round(split_checkpoint * runtime_ratio)
    fixed_resident = layer.resident_bytes - split_resident
    resident = [value + SHARD_BUFFER_BYTES for value in split_integer(split_resident, degree)]
    resident[0] += fixed_resident
    if kind in {
        PartitionKind.ATTENTION_PROJECTION_SHARD,
        PartitionKind.FULL_MIXED_STRIPE,
    }:
        # The canonical checkpoint bytes remain coordinator-owned, while every
        # attention worker retains its own hot-path copy of these tiny vectors.
        # The whole-layer resident measurement already includes worker 0's
        # copy, so only additional workers add memory here.
        replicated_resident = round(attention_replicated * runtime_ratio)
        for worker in range(1, degree):
            resident[worker] += replicated_resident
    if sum(checkpoint) != layer.checkpoint_bytes:
        raise RuntimeError("candidate checkpoint accounting did not reconcile")
    if sum(resident) < layer.resident_bytes:
        raise RuntimeError("candidate resident accounting lost model state")
    return tuple(resident), tuple(checkpoint)


def candidate_catalog(model: ModelGraph) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for layer in model.layers:
        for kind in PartitionKind:
            degrees = (1,) if kind is PartitionKind.WHOLE_LAYER else (2, 4, 8, 16)
            if layer.layer_id == 0 and kind in (
                PartitionKind.WHOLE_EXPERT,
                PartitionKind.EXPERT_SHARD,
            ):
                continue
            for degree in degrees:
                resident, checkpoint = candidate_memory(layer, kind, degree)
                starts = [0]
                for value in checkpoint:
                    starts.append(starts[-1] + value)
                ranges = [
                    {
                        "worker_index": worker,
                        "layer_local_start_byte": starts[worker],
                        "layer_local_end_byte_exclusive": starts[worker + 1],
                        "bytes": checkpoint[worker],
                    }
                    for worker in range(degree)
                ]
                split_components = {
                    PartitionKind.WHOLE_LAYER: set(),
                    PartitionKind.WHOLE_EXPERT: {"routed_expert"},
                    PartitionKind.EXPERT_SHARD: {"routed_expert"},
                    PartitionKind.ATTENTION_PROJECTION_SHARD: {
                        "attention_shardable"
                    },
                    PartitionKind.FULL_MIXED_STRIPE: {
                        "attention_shardable",
                        "projection",
                        "routed_expert",
                        "shared_expert",
                    },
                }[kind]
                component_ownership = [
                    {
                        "component": component,
                        "checkpoint_bytes": bytes_,
                        "ownership": (
                            "degree_way_complete_expert_id_groups"
                            if kind is PartitionKind.WHOLE_EXPERT
                            and component == "routed_expert"
                            else "runtime_replica_all_attention_workers_canonical_checkpoint_on_coordinator"
                            if component == "attention_replicated"
                            and kind
                            in {
                                PartitionKind.ATTENTION_PROJECTION_SHARD,
                                PartitionKind.FULL_MIXED_STRIPE,
                            }
                            else "degree_way_exact_tensor_axis_stripe"
                            if component in split_components
                            else "coordinator_whole_component"
                        ),
                    }
                    for component, bytes_ in sorted(layer.component_bytes.items())
                ]
                compute_tasks = {
                    PartitionKind.WHOLE_LAYER: ["whole_layer"],
                    PartitionKind.WHOLE_EXPERT: [
                        "attention_preprocess",
                        "attention_whole",
                        "post_attention_preprocess",
                        "router",
                        "latent_down_whole",
                        "expert_whole_group",
                        "expert_reduction",
                        "routed_norm",
                        "shared_expert_whole",
                        "latent_up_whole",
                        "routed_shared_reduction",
                        "output_state_commit",
                    ],
                    PartitionKind.EXPERT_SHARD: [
                        "attention_preprocess",
                        "attention_whole",
                        "post_attention_preprocess",
                        "router",
                        "latent_down_whole",
                        "expert_stripe",
                        "expert_reduction",
                        "routed_norm",
                        "shared_expert_whole",
                        "latent_up_whole",
                        "routed_shared_reduction",
                        "output_state_commit",
                    ],
                    PartitionKind.ATTENTION_PROJECTION_SHARD: [
                        "attention_preprocess",
                        "attention_common",
                        "attention_shard",
                        "attention_reduction",
                        "post_attention_preprocess",
                        "router",
                        "latent_down_whole",
                        "expert_whole",
                        "routed_norm",
                        "shared_expert_whole",
                        "latent_up_whole",
                        "routed_shared_reduction",
                        "output_state_commit",
                    ],
                    PartitionKind.FULL_MIXED_STRIPE: [
                        "attention_preprocess",
                        "attention_common",
                        "attention_shard",
                        "attention_reduction",
                        "post_attention_preprocess",
                        "router",
                        "latent_down",
                        "expert_stripe",
                        "expert_reduction",
                        "routed_norm",
                        "shared_expert",
                        "shared_reduction",
                        "latent_up",
                        "latent_up_reduction",
                        "routed_shared_reduction",
                        "output_state_commit",
                    ],
                }[kind]
                worker_compute_dag = [
                    {
                        "node_id": f"compute-{index:02d}-{operation}",
                        "operation": operation,
                        "depends_on": (
                            []
                            if index == 0
                            else [
                                f"compute-{index - 1:02d}-{compute_tasks[index - 1]}"
                            ]
                        ),
                    }
                    for index, operation in enumerate(compute_tasks)
                ]
                collective_steps = (
                    []
                    if degree == 1
                    else [
                        "activation_fanout",
                        "local_exact_contribution",
                        "deterministic_reduction",
                    ]
                )
                collective_dag = [
                    {
                        "node_id": f"collective-{index:02d}-{operation}",
                        "operation": operation,
                        "depends_on": (
                            [worker_compute_dag[-1]["node_id"]]
                            if index == 0
                            else [
                                f"collective-{index - 1:02d}-{collective_steps[index - 1]}"
                            ]
                        ),
                    }
                    for index, operation in enumerate(collective_steps)
                ]
                candidates.append(
                    {
                        "candidate_id": f"layer-{layer.layer_id:02d}:{kind.value}:p{degree}",
                        "candidate_type": kind.value,
                        "layer": layer.layer_id,
                        "layer_type": layer.layer_type.value,
                        "partition_type": kind.value,
                        "degree": degree,
                        "partition_degree": degree,
                        "worker_count": degree,
                        "resident_memory_bytes": list(resident),
                        "checkpoint_bytes": list(checkpoint),
                        "checkpoint_byte_ranges": ranges,
                        "checkpoint_range_coordinate": (
                            "canonical layer-local logical payload; semantic tensor axes "
                            "are defined by the named physical primitive"
                        ),
                        "component_ownership": component_ownership,
                        "compute_tasks": compute_tasks,
                        "checkpoint_gap_bytes": layer.checkpoint_bytes - sum(checkpoint),
                        "checkpoint_overlap_bytes": 0,
                        "state_ownership": (
                            "coordinator owns AttnRes/router; attention workers own KDA/MLA state"
                            if degree > 1
                            else "whole layer owner owns all recurrent and AttnRes state"
                        ),
                        "collectives": (
                            []
                            if degree == 1
                            else [
                                "activation fanout",
                                "local exact contribution",
                                "deterministic reduction",
                            ]
                        ),
                        "worker_compute_dag": worker_compute_dag,
                        "collective_dag": collective_dag,
                        "network_payload": (
                            {"kind": "none", "bytes_per_chunk": 0}
                            if degree == 1
                            else {
                                "kind": "explicit event-DAG fanout/gather",
                                "formulas": {
                                    "hidden": "chunk_rows * 7168 * 4",
                                    "attention_common_KDA": "chunk_rows * 128 * 4",
                                    "attention_common_Gated_MLA": (
                                        "chunk_rows * (1536 + 512 + 64) * 4"
                                    ),
                                    "latent": "chunk_rows * 3584 * 4",
                                    "route_metadata": "chunk_rows * 16 * (4 + 4)",
                                },
                                "transport_service_owner": "network event only",
                            }
                        ),
                        "physical_primitive_required": kind.value,
                        "correctness_status": "PENDING_E022_VALIDATION",
                        "service_status": "PENDING_E022_VALIDATION",
                    }
                )
    return {
        "schema_version": "experiment-022-candidate-catalog-v1",
        "model_id": model.model_id,
        "checkpoint_index_sha256": model.checkpoint_index_sha256,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def model_metadata(model: ModelGraph) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    for layer in model.layers:
        counts[layer.layer_type.value] += 1
    return {
        "schema_version": "experiment-022-model-metadata-v1",
        "model": "Kimi K3",
        "model_id": model.model_id,
        "checkpoint_index_sha256": model.checkpoint_index_sha256,
        "checkpoint_payload_bytes": model.checkpoint_payload_bytes,
        "checkpoint_payload_gib": model.checkpoint_payload_bytes / 1024**3,
        "tensor_count": model.tensor_count,
        "transformer_layers": len(model.layers),
        "layer_type_counts": dict(sorted(counts.items())),
        "endpoint_checkpoint_bytes": model.endpoint_checkpoint_bytes,
        "endpoint_resident_bytes": model.endpoint_resident_bytes,
        "total_modeled_resident_bytes": model.total_resident_bytes,
        "maximum_layer_resident_bytes": model.maximum_layer_resident_bytes,
        "memory_basis": (
            "actual checkpoint tensor bytes plus E018 persistent whole-layer resident "
            "memory ratios; no hardware-name branch"
        ),
    }


__all__ = [
    "SHARD_BUFFER_BYTES",
    "build_model_graph",
    "candidate_catalog",
    "candidate_memory",
    "model_metadata",
]
