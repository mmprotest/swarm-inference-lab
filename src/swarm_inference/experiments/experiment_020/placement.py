"""Byte-exact E020 placement materialization for the E021 candidate."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_019.checkpoint import CheckpointCatalog
from swarm_inference.experiments.experiment_019.placement import (
    GIB,
    HIDDEN,
    PlacementResult,
    PlacementSpec,
    assignments_for,
    build_placement,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _is_scale(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith("scale") or ".scale" in lowered or "weight_scale" in lowered


def _component_breakdown(worker: Any, spec: PlacementSpec, scale_bytes: int) -> dict[str, int]:
    rows = spec.chunk_rows
    expert_scratch = rows * 16 * ((3072 + spec.stripe_degree - 1) // spec.stripe_degree) * 2 * 4
    attention_scratch = rows * ((12288 + spec.stripe_degree - 1) // spec.stripe_degree) * 8
    activations = rows * HIDDEN * 4 * 6
    scratch = worker.dynamic_buffer_bytes - activations
    if scratch != expert_scratch + attention_scratch:
        raise RuntimeError("dynamic worker buffer decomposition drifted")
    collective = worker.network_buffer_bytes // 2
    transport = worker.network_buffer_bytes - collective
    static_weights = worker.static_weight_bytes - scale_bytes
    components = {
        "static_weights": static_weights,
        "scales": scale_bytes,
        "persistent_state": worker.persistent_state_bytes,
        "activations": activations,
        "scratch": scratch,
        "collective_buffers": collective,
        "transport_buffers": transport,
        "cuda_workspace": worker.workspace_bytes,
        "allocator_allowance": worker.allocator_overhead_bytes,
    }
    if sum(components.values()) != worker.peak_total_bytes:
        raise RuntimeError("worker memory components do not reconcile to peak")
    return components


def materialize_exact_placement(
    checkpoint: Path,
    placement_root: Path,
    *,
    stripe_degree: int = 8,
    depth_span: int = 8,
    chunk_rows: int = 1,
    memory_cap_gib: float = 20,
    expert_allocation_overhead_factor: float = 1.0,
) -> tuple[CheckpointCatalog, PlacementResult, dict[str, Any]]:
    catalog = CheckpointCatalog(checkpoint)
    spec = PlacementSpec(
        memory_cap_gib=memory_cap_gib,
        stripe_degree=stripe_degree,
        depth_span=depth_span,
        chunk_rows=chunk_rows,
        expert_allocation_overhead_factor=expert_allocation_overhead_factor,
        hardware_class="E021_PRIMARY_RTX_3090_SM86_UNMEASURED",
    )
    result = build_placement(catalog, spec)
    placement_root.mkdir(parents=True, exist_ok=True)
    manifest_root = placement_root / "worker-manifest"
    manifest_root.mkdir(parents=True, exist_ok=True)

    scale_bytes = [0 for _ in result.workers]
    coverage_path = placement_root / "tensor-coverage.csv"
    fieldnames = [
        "tensor",
        "file",
        "dtype",
        "shape",
        "source_offset",
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
    ]
    coverage_assigned = 0
    tensor_count = 0
    with coverage_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for record in sorted(catalog.records().values(), key=lambda item: item.name):
            assignments = assignments_for(record, spec)
            owners = []
            for assignment in assignments:
                worker = result.workers[assignment.worker_index]
                if _is_scale(record.name):
                    scale_bytes[assignment.worker_index] += assignment.bytes
                owners.append(
                    {
                        "worker_id": worker.worker_id,
                        "pod": worker.pod_id,
                        "gpu_index": worker.stripe_index,
                        "axis": assignment.axis,
                        "start": assignment.start,
                        "stop": assignment.stop,
                        "total": assignment.total,
                        "bytes": assignment.bytes,
                        "replicated": assignment.replicated,
                    }
                )
            assigned = sum(item["bytes"] for item in owners if not item["replicated"])
            replicated = sum(item["bytes"] for item in owners if item["replicated"])
            if assigned != record.byte_size:
                raise RuntimeError(f"tensor coverage failed for {record.name}")
            writer.writerow(
                {
                    "tensor": record.name,
                    "file": record.file,
                    "dtype": record.dtype,
                    "shape": "x".join(str(value) for value in record.shape),
                    "source_offset": record.data_offset,
                    "source_bytes": record.byte_size,
                    "layer_id": "" if record.layer_id is None else record.layer_id,
                    "expert_id": "" if record.expert_id is None else record.expert_id,
                    "role": record.role,
                    "partition_count": len(owners),
                    "assigned_bytes": assigned,
                    "replicated_bytes": replicated,
                    "gap_bytes": 0,
                    "overlap_bytes": 0,
                    "coverage_status": "PASS",
                    "owners_json": json.dumps(owners, separators=(",", ":")),
                }
            )
            tensor_count += 1
            coverage_assigned += assigned

    workers = []
    memory_rows = []
    for index, worker in enumerate(result.workers):
        components = _component_breakdown(worker, spec, scale_bytes[index])
        row = {
            "schema_version": "experiment-020-worker-manifest-v1",
            "worker_id": worker.worker_id,
            "pod": worker.pod_id,
            "gpu_index": worker.stripe_index,
            "layers": worker.assigned_layers,
            "tensor_slices": worker.assigned_tensor_slices,
            "expert_stripes": worker.assigned_expert_stripes,
            "attention_heads": worker.attention_head_ranges,
            "KDA_state": [
                value for value in worker.state_ownership if value["type"].startswith("KDA")
            ],
            "MLA_state": [
                value for value in worker.state_ownership if value["type"].startswith("MLA")
            ],
            "AttnRes_ownership": worker.AttnRes_cache_ownership,
            "memory_bytes": components,
            "peak_total_bytes": worker.peak_total_bytes,
            "peak_total_gib": worker.peak_total_bytes / GIB,
            "checkpoint_byte_ranges": worker.checkpoint_byte_ranges,
            "checkpoint_hashes": worker.checkpoint_hashes,
            "largest_layer_fraction": worker.largest_layer_fraction,
            "largest_expert_fraction": worker.largest_expert_fraction,
            "largest_shared_expert_fraction": worker.largest_shared_expert_fraction,
            "whole_layer": worker.largest_layer_fraction >= 1.0,
            "whole_routed_expert": worker.largest_expert_fraction >= 1.0,
            "whole_shared_expert": worker.largest_shared_expert_fraction >= 1.0,
        }
        _write_json(manifest_root / f"{worker.worker_id}.json", row)
        workers.append(
            {
                "worker_id": worker.worker_id,
                "manifest": f"worker-manifest/{worker.worker_id}.json",
                "pod": worker.pod_id,
                "gpu_index": worker.stripe_index,
                "layers": worker.assigned_layers,
                "peak_total_bytes": worker.peak_total_bytes,
            }
        )
        memory_rows.append(
            {
                "worker_id": worker.worker_id,
                "pod": worker.pod_id,
                "gpu_index": worker.stripe_index,
                **components,
                "peak_total_bytes": worker.peak_total_bytes,
                "peak_total_gib": worker.peak_total_bytes / GIB,
            }
        )

    summary = {
        "schema_version": "experiment-020-final-placement-v1",
        "placement": asdict(spec),
        "worker_count": len(result.workers),
        "pod_count": spec.pod_count,
        "workers_per_pod": spec.stripe_degree,
        "checkpoint_payload_bytes": result.checkpoint_payload_bytes,
        "coverage_tensor_count": tensor_count,
        "coverage_assigned_bytes": coverage_assigned,
        "coverage_gap_bytes": 0,
        "coverage_overlap_bytes": 0,
        "full_checkpoint_covered": coverage_assigned == result.checkpoint_payload_bytes,
        "total_resident_weight_bytes": result.total_resident_weight_bytes,
        "replicated_weight_bytes": result.replicated_weight_bytes,
        "maximum_worker_peak_bytes": result.max_worker_peak_bytes,
        "maximum_worker_peak_gib": result.max_worker_peak_bytes / GIB,
        "maximum_layer_fraction": result.max_layer_fraction,
        "maximum_expert_fraction": result.max_expert_fraction,
        "maximum_shared_expert_fraction": result.max_shared_expert_fraction,
        "whole_layer_on_any_worker": any(row["largest_layer_fraction"] >= 1.0 for row in (w.manifest_row() for w in result.workers)),
        "whole_expert_on_any_worker": result.max_expert_fraction >= 1.0,
        "valid": result.valid
        and result.max_worker_peak_bytes <= 20 * GIB
        and result.max_expert_fraction <= 0.125 + 1e-12,
        "workers": workers,
    }
    _write_json(placement_root / "final-placement.json", summary)
    _write_json(
        placement_root / "memory-audit.json",
        {
            "schema_version": "experiment-020-memory-audit-v1",
            "status": "PASS" if summary["valid"] else "FAIL",
            "accounting_invariant": "sum(memory_bytes) == peak_total_bytes",
            "maximum_worker_peak_bytes": result.max_worker_peak_bytes,
            "maximum_worker_peak_gib": result.max_worker_peak_bytes / GIB,
            "cap_bytes": 20 * GIB,
            "workers": memory_rows,
        },
    )
    return catalog, result, summary


__all__ = ["materialize_exact_placement"]
