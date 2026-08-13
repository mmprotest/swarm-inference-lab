"""Independent-machine placement and whole-layer infeasibility receipts."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_019.checkpoint import (
    CheckpointCatalog,
    DirectShardLoader,
)
from swarm_inference.experiments.experiment_019.placement import (
    GIB,
    HIDDEN,
    KDA_HEAD_DIMENSION,
    KDA_HEADS,
    KDA_LAYERS,
    MIB,
    MLA_CACHE_WIDTH,
    PlacementResult,
    PlacementSpec,
    assignments_for,
    build_placement,
    estimate_candidate,
)

from .io import atomic_write_json, canonical_sha256, write_csv, write_csv_stream

MEMORY_TIERS_GIB = (8, 4, 2, 1)
STRIPE_DEGREES = (4, 8, 16, 32)
DEPTH_SPANS = (1, 2, 4, 8)
CHUNK_ROWS = (1, 2, 4)
TRANSFORMER_LAYERS = 93
MAXIMUM_CONTEXT = 4096
BLOCK_CANDIDATES = 16
EXPERT_ALLOCATION_OVERHEAD_FACTOR = 1.0137388288648792


def _layer_memory_components(
    *,
    layer: int,
    static_weight_bytes: int,
    routed_expert_weight_bytes: int,
    chunk_rows: int = 1,
) -> dict[str, int]:
    if layer in KDA_LAYERS:
        recurrent = KDA_HEADS * KDA_HEAD_DIMENSION * KDA_HEAD_DIMENSION * 4
        convolution = 3 * KDA_HEADS * KDA_HEAD_DIMENSION * 4 * 4
        persistent_state = recurrent + convolution
        state_kind = 1
    else:
        persistent_state = MAXIMUM_CONTEXT * MLA_CACHE_WIDTH * 4
        state_kind = 2
    queued_chunks = math.ceil((BLOCK_CANDIDATES + 1) / chunk_rows)
    attnres_cache = queued_chunks * chunk_rows * HIDDEN * 4
    persistent_state += attnres_cache
    expert_scratch = chunk_rows * 16 * 3072 * 2 * 4
    attention_scratch = chunk_rows * 12288 * 8
    activations = chunk_rows * HIDDEN * 4 * 6
    transport_and_reduction = chunk_rows * HIDDEN * 4 * 4
    workspace = max(64 * MIB, min(512 * MIB, static_weight_bytes // 12))
    subtotal = (
        static_weight_bytes
        + persistent_state
        + expert_scratch
        + attention_scratch
        + activations
        + transport_and_reduction
        + workspace
    )
    expert_allocator = math.ceil(
        routed_expert_weight_bytes * (EXPERT_ALLOCATION_OVERHEAD_FACTOR - 1.0)
    )
    allocator = math.ceil(subtotal * 0.03) + expert_allocator
    return {
        "checkpoint_weight_bytes": static_weight_bytes,
        "persistent_state_bytes": persistent_state,
        "attnres_cache_bytes": attnres_cache,
        "activation_bytes": activations,
        "scratch_bytes": expert_scratch + attention_scratch,
        "transport_and_reduction_buffer_bytes": transport_and_reduction,
        "cuda_workspace_bytes": workspace,
        "allocator_overhead_bytes": allocator,
        "runtime_state_kind_code": state_kind,
        "complete_layer_peak_bytes": subtotal + allocator,
    }


def whole_layer_feasibility(
    catalog: CheckpointCatalog,
    *,
    caps_gib: Sequence[int] = MEMORY_TIERS_GIB,
) -> dict[str, Any]:
    """Attempt a literal whole-layer placement using the declared worker caps."""

    layer_bytes = [0] * TRANSFORMER_LAYERS
    expert_bytes = [0] * TRANSFORMER_LAYERS
    tensor_counts = [0] * TRANSFORMER_LAYERS
    for record in catalog.records().values():
        if record.layer_id is None:
            continue
        layer_bytes[record.layer_id] += record.byte_size
        tensor_counts[record.layer_id] += 1
        if record.role == "routed_expert":
            expert_bytes[record.layer_id] += record.byte_size
    layers = []
    for layer in range(TRANSFORMER_LAYERS):
        components = _layer_memory_components(
            layer=layer,
            static_weight_bytes=layer_bytes[layer],
            routed_expert_weight_bytes=expert_bytes[layer],
        )
        layers.append(
            {
                "layer": layer,
                "attention_type": "KDA" if layer in KDA_LAYERS else "GATED_MLA",
                "tensor_count": tensor_counts[layer],
                **components,
                "complete_layer_peak_gib": components["complete_layer_peak_bytes"] / GIB,
                "fits": {
                    str(cap): components["complete_layer_peak_bytes"] <= cap * GIB
                    for cap in caps_gib
                },
            }
        )
    tiers = []
    for cap in caps_gib:
        fitting = [row["layer"] for row in layers if row["fits"][str(cap)]]
        non_fitting = [row["layer"] for row in layers if not row["fits"][str(cap)]]
        tiers.append(
            {
                "worker_cap_gib": cap,
                "worker_cap_bytes": cap * GIB,
                "fitting_whole_layers": fitting,
                "non_fitting_whole_layers": non_fitting,
                "fitting_layer_count": len(fitting),
                "non_fitting_layer_count": len(non_fitting),
                "complete_model_whole_layer_placement_possible": not non_fitting,
                "placement_attempt": [
                    {
                        "layer": row["layer"],
                        "assigned_worker": (
                            f"whole-layer-control-machine-{row['layer']:03d}"
                            if row["fits"][str(cap)]
                            else None
                        ),
                        "placed": bool(row["fits"][str(cap)]),
                    }
                    for row in layers
                ],
            }
        )
    return {
        "schema_version": "experiment-021-whole-layer-feasibility-v1",
        "checkpoint_index": str(catalog.index_path),
        "checkpoint_index_sha256": catalog.index_sha256,
        "runtime_memory_policy": {
            "maximum_context": MAXIMUM_CONTEXT,
            "block_candidates": BLOCK_CANDIDATES,
            "chunk_rows": 1,
            "expert_allocation_overhead_factor": EXPERT_ALLOCATION_OVERHEAD_FACTOR,
            "allocator_fraction": 0.03,
            "workspace_floor_bytes": 64 * MIB,
            "workspace_ceiling_bytes": 512 * MIB,
            "includes": [
                "checkpoint weights and quantization scales",
                "KDA recurrent/conv or MLA cache state",
                "immutable AttnRes cache",
                "activations and scratch",
                "transport and reduction buffers",
                "CUDA workspace",
                "measured/declared allocator overhead",
            ],
        },
        "layers": layers,
        "tiers": tiers,
        "headline_8g_complete_model_whole_layer_placement_possible": next(
            row["complete_model_whole_layer_placement_possible"]
            for row in tiers
            if row["worker_cap_gib"] == 8
        ),
        "status": "PASS",
    }


def search_placements(catalog: CheckpointCatalog) -> tuple[list[dict[str, Any]], dict[int, PlacementSpec]]:
    rows: list[dict[str, Any]] = []
    winners: dict[int, PlacementSpec] = {}
    for cap in MEMORY_TIERS_GIB:
        preferred_degree = {8: 8, 4: 8, 2: 16, 1: 32}[cap]
        candidates: list[
            tuple[tuple[int, int, int, int, int], PlacementSpec, dict[str, Any]]
        ] = []
        for degree in STRIPE_DEGREES:
            for depth in DEPTH_SPANS:
                for chunk in CHUNK_ROWS:
                    spec = PlacementSpec(
                        cap,
                        degree,
                        depth,
                        chunk,
                        hardware_class="INDEPENDENT_MACHINE_RTX5090_SHARD_SERVICE",
                        expert_allocation_overhead_factor=EXPERT_ALLOCATION_OVERHEAD_FACTOR,
                    )
                    estimate = estimate_candidate(catalog, spec)
                    row = {
                        **estimate,
                        "one_worker_per_machine": True,
                        "machines_per_worker": 1,
                        "same_host_collective": False,
                        "candidate_exactly_materialized": False,
                    }
                    rows.append(row)
                    if estimate["capacity_valid"]:
                        # Prefer the physically characterized grouped-stripe
                        # degrees, then fewer machines and smaller peak.  No
                        # throughput result enters this capacity-only search.
                        score = (
                            int(degree != preferred_degree),
                            spec.worker_count,
                            int(estimate["estimated_max_peak_bytes"]),
                            degree,
                            chunk,
                        )
                        candidates.append((score, spec, row))
        if not candidates:
            continue
        _score, winner, winning_row = min(candidates, key=lambda value: value[0])
        winners[cap] = winner
        winning_row["capacity_search_winner"] = True
    return rows, winners


def _independent_worker_id(index: int) -> str:
    return f"machine-{index:04d}.worker"


def independent_manifest(result: PlacementResult) -> dict[str, Any]:
    workers = []
    for index, worker in enumerate(result.workers):
        row = dataclasses.asdict(worker)
        original_group = str(row.pop("pod_id"))
        original_worker = str(row["worker_id"])
        worker_id = _independent_worker_id(index)
        row["worker_id"] = worker_id
        row["machine_id"] = f"machine-{index:04d}"
        row["compute_workers_on_machine"] = 1
        row["depth_group_id"] = original_group.replace("pod-", "depth-group-")
        row["depth_group_has_compute"] = False
        row["depth_group_has_memory"] = False
        row["legacy_internal_worker_key"] = original_worker
        row["gpu_index_within_machine"] = 0
        row.pop("stripe_index", None)
        for cache in row.get("AttnRes_cache_ownership", []):
            cache["owner"] = worker_id
        slices = row.get("assigned_tensor_slices", {})
        slices["worker_filter"] = worker_id
        slices["artifact"] = "placement/checkpoint-coverage.csv"
        row["assigned_tensor_slices"] = slices
        ranges = row.get("checkpoint_byte_ranges", {})
        ranges["worker_filter"] = worker_id
        ranges["artifact"] = "placement/checkpoint-coverage.csv"
        row["checkpoint_byte_ranges"] = ranges
        row["whole_layer"] = row["largest_layer_fraction"] >= 1.0
        row["whole_routed_expert"] = row["largest_expert_fraction"] >= 1.0
        row["whole_shared_expert"] = row["largest_shared_expert_fraction"] >= 1.0
        workers.append(row)
    summary = {
        "worker_count": len(workers),
        "machine_count": len(workers),
        "compute_workers_per_machine": 1,
        "logical_depth_group_count": result.spec.pod_count,
        "logical_depth_groups_are_compute_resources": False,
        "logical_depth_groups_have_memory": False,
        "checkpoint_payload_bytes": result.checkpoint_payload_bytes,
        "total_resident_weight_bytes": result.total_resident_weight_bytes,
        "replicated_weight_bytes": result.replicated_weight_bytes,
        "coverage_tensor_count": result.coverage_tensor_count,
        "coverage_assigned_bytes": result.coverage_assigned_bytes,
        "coverage_gap_bytes": result.coverage_gap_bytes,
        "coverage_overlap_bytes": result.coverage_overlap_bytes,
        "max_worker_peak_bytes": result.max_worker_peak_bytes,
        "max_worker_peak_gib": result.max_worker_peak_bytes / GIB,
        "max_layer_fraction": result.max_layer_fraction,
        "max_expert_fraction": result.max_expert_fraction,
        "max_shared_expert_fraction": result.max_shared_expert_fraction,
        "whole_layer_on_any_worker": result.max_layer_fraction >= 1.0,
        "whole_routed_expert_on_any_worker": result.max_expert_fraction >= 1.0,
        "whole_shared_expert_on_any_worker": result.max_shared_expert_fraction >= 1.0,
        "same_host_pcie_nvlink_nccl_assumed": False,
        "valid": result.valid,
        "invalid_reasons": result.invalid_reasons,
    }
    return {
        "schema_version": "experiment-021-independent-worker-manifest-v1",
        "placement": {
            **dataclasses.asdict(result.spec),
            "one_worker_equals_one_independent_machine": True,
            "same_host_collectives": False,
        },
        "summary": summary,
        "workers": workers,
    }


def iter_coverage_rows(
    catalog: CheckpointCatalog,
    result: PlacementResult,
) -> Iterable[dict[str, Any]]:
    for record in sorted(catalog.records().values(), key=lambda item: item.name):
        owners = []
        for assignment in assignments_for(record, result.spec):
            owners.append(
                {
                    "worker_id": _independent_worker_id(assignment.worker_index),
                    "machine_id": f"machine-{assignment.worker_index:04d}",
                    "axis": assignment.axis,
                    "start": assignment.start,
                    "stop": assignment.stop,
                    "total": assignment.total,
                    "bytes": assignment.bytes,
                    "replicated": assignment.replicated,
                }
            )
        assigned = sum(row["bytes"] for row in owners if not row["replicated"])
        replicated = sum(row["bytes"] for row in owners if row["replicated"])
        yield {
            "tensor": record.name,
            "file": record.file,
            "dtype": record.dtype,
            "shape": "x".join(str(value) for value in record.shape),
            "source_bytes": record.byte_size,
            "layer_id": "" if record.layer_id is None else record.layer_id,
            "expert_id": "" if record.expert_id is None else record.expert_id,
            "role": record.role,
            "partition_count": len(owners),
            "assigned_bytes": assigned,
            "replicated_bytes": replicated,
            "gap_bytes": max(0, record.byte_size - assigned),
            "overlap_bytes": max(0, assigned - record.byte_size),
            "coverage_status": "PASS" if assigned == record.byte_size else "FAIL",
            "owners_json": json.dumps(owners, sort_keys=True, separators=(",", ":")),
            "owners_sha256": canonical_sha256(owners),
        }


def direct_read_audit(catalog: CheckpointCatalog, result: PlacementResult) -> dict[str, Any]:
    loader = DirectShardLoader(catalog)
    selected_roles: dict[str, Any] = {}
    for record in sorted(catalog.records().values(), key=lambda item: (-item.byte_size, item.name)):
        if record.role in selected_roles or len(record.shape) < 2:
            continue
        assignments = assignments_for(record, result.spec)
        assignment = next((value for value in assignments if value.axis is not None), None)
        if assignment is None:
            continue
        loader.load(
            record.name,
            worker_id=_independent_worker_id(assignment.worker_index),
            purpose=f"e021_direct_read_{record.role}",
            axis=assignment.axis,
            start=assignment.start,
            stop=assignment.stop,
        )
        selected_roles[record.role] = record.name
    violations = [
        row
        for row in loader.audit
        if row["full_source_tensor_materialized"]
        and int(row["source_tensor_bytes"]) > 32 * MIB
    ]
    return {
        "schema_version": "experiment-021-direct-read-audit-v1",
        "status": "PASS" if not violations and loader.audit else "FAIL",
        "checkpoint_index_sha256": catalog.index_sha256,
        "representative_roles": selected_roles,
        "request_count": len(loader.audit),
        "bytes_read": sum(int(row["bytes_read"]) for row in loader.audit),
        "source_bytes_addressed": sum(int(row["source_tensor_bytes"]) for row in loader.audit),
        "large_full_tensor_materializations": len(violations),
        "violations": violations,
        "reads": loader.audit,
    }


def materialize_placement_artifacts(
    checkpoint: Path,
    placement_root: Path,
) -> tuple[CheckpointCatalog, dict[int, PlacementResult], dict[str, Any]]:
    catalog = CheckpointCatalog(checkpoint)
    catalog.records()
    feasibility = whole_layer_feasibility(catalog)
    atomic_write_json(placement_root / "whole-layer-feasibility.json", feasibility)
    search_rows, specs = search_placements(catalog)
    exact: dict[int, PlacementResult] = {}
    memory_rows: list[dict[str, Any]] = []
    for cap in MEMORY_TIERS_GIB:
        spec = specs[cap]
        result = build_placement(catalog, spec)
        exact[cap] = result
        manifest = independent_manifest(result)
        atomic_write_json(placement_root / f"worker-manifest-{cap}g.json", manifest)
        memory_rows.append(
            {
                "worker_cap_gib": cap,
                "stripe_degree": spec.stripe_degree,
                "depth_span": spec.depth_span,
                "chunk_rows": spec.chunk_rows,
                "worker_count": len(result.workers),
                "machine_count": len(result.workers),
                "max_peak_bytes": result.max_worker_peak_bytes,
                "max_peak_gib": result.max_worker_peak_bytes / GIB,
                "total_resident_bytes": result.total_resident_weight_bytes,
                "max_layer_fraction": result.max_layer_fraction,
                "max_expert_fraction": result.max_expert_fraction,
                "max_shared_expert_fraction": result.max_shared_expert_fraction,
                "coverage_gap_bytes": result.coverage_gap_bytes,
                "coverage_overlap_bytes": result.coverage_overlap_bytes,
                "one_worker_per_machine": True,
                "valid": result.valid,
            }
        )
    write_csv(placement_root / "worker-memory-tiers.csv", memory_rows)
    write_csv(placement_root / "candidate-search.csv", search_rows)
    coverage_fields = [
        "tensor",
        "file",
        "dtype",
        "shape",
        "source_bytes",
        "layer_id",
        "expert_id",
        "role",
        "partition_count",
        "assigned_bytes",
        "replicated_bytes",
        "gap_bytes",
        "overlap_bytes",
        "coverage_status",
        "owners_json",
        "owners_sha256",
    ]
    coverage_count = write_csv_stream(
        placement_root / "checkpoint-coverage.csv",
        coverage_fields,
        iter_coverage_rows(catalog, exact[8]),
    )
    direct = direct_read_audit(catalog, exact[8])
    atomic_write_json(placement_root / "direct-read-audit.json", direct)
    receipt = {
        "schema_version": "experiment-021-placement-suite-v1",
        "status": (
            "PASS"
            if coverage_count == len(catalog.records())
            and all(value.valid for value in exact.values())
            and direct["status"] == "PASS"
            and not feasibility[
                "headline_8g_complete_model_whole_layer_placement_possible"
            ]
            else "FAIL"
        ),
        "checkpoint_index_sha256": catalog.index_sha256,
        "checkpoint_tensor_count": len(catalog.records()),
        "checkpoint_payload_bytes": sum(
            value.byte_size for value in catalog.records().values()
        ),
        "coverage_row_count": coverage_count,
        "memory_tiers": memory_rows,
        "whole_layer_feasibility_status": feasibility["status"],
        "headline_whole_layer_placement_possible": feasibility[
            "headline_8g_complete_model_whole_layer_placement_possible"
        ],
        "direct_read_status": direct["status"],
    }
    return catalog, exact, receipt


__all__ = [
    "BLOCK_CANDIDATES",
    "CHUNK_ROWS",
    "DEPTH_SPANS",
    "EXPERT_ALLOCATION_OVERHEAD_FACTOR",
    "MEMORY_TIERS_GIB",
    "STRIPE_DEGREES",
    "direct_read_audit",
    "independent_manifest",
    "materialize_placement_artifacts",
    "search_placements",
    "whole_layer_feasibility",
]
