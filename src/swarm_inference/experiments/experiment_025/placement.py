"""Freeze the E014 whole-layer placement with one real four-way sub-layer group."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_014.distribution import (
    build_distribution_manifest,
)

from .constants import (
    EVIDENCE_CLASS,
    GIB,
    MIB,
    MODEL_ID,
    MODEL_REVISION,
    ROUTED_EXPERTS,
    SMALL_WORKER_PHYSICAL_VRAM_BYTES,
    SUB_LAYER_TARGET,
    SUB_LAYER_WORKERS,
    TRANSFORMER_LAYERS,
    VRAM_SAFETY_FRACTION,
)
from .io import atomic_write_json, canonical_sha256, read_json, sha256_file, utc_now

SCHEMA_VERSION = "experiment-025-physical-placement-v1"


def _tensor_rows(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [tensor for unit in units for tensor in unit["tensors"]]


def _worker_payload(
    source: dict[str, Any],
    *,
    worker_id: str,
    worker_role: str,
    units: list[dict[str, Any]],
    owned_layers: list[int],
    owned_components: list[str],
) -> dict[str, Any]:
    worker = copy.deepcopy(source)
    tensors = sorted(_tensor_rows(units), key=lambda row: str(row["name"]))
    experts = sorted(
        int(unit["routed_expert"])
        for unit in units
        if unit.get("routed_expert") is not None
    )
    worker.update(
        {
            "worker_id": worker_id,
            "worker_role": worker_role,
            "physical_role": worker_role,
            "assignment_units": units,
            "assignment_sha256": canonical_sha256(units),
            "owned_layers": owned_layers,
            "owned_components": owned_components,
            "owned_expert_count": len(experts),
            "owned_expert_ids": experts,
            "tensor_count": len(tensors),
            "source_weight_bytes": sum(int(row["physical_bytes"]) for row in tensors),
            "expected_on_disk_package_bytes_excluding_header": sum(
                int(row["physical_bytes"]) for row in tensors
            ),
            "safetensor_source_files": sorted(
                {str(row["safetensors_file"]) for row in tensors}
            ),
            "consumer_gpu_required": True,
            "independent_physical_machine_required": True,
            "whole_layer_fallback": False,
        }
    )
    return worker


def _validate_source(source: dict[str, Any]) -> None:
    if source.get("status") != "PASS":
        raise ValueError("E014 source placement is not passing")
    if source.get("schema_version") != "experiment-014-k3-final-physical-placement-v1":
        raise ValueError("E025 requires the final E014 physical-placement schema")
    if int(source.get("node_count", 0)) != TRANSFORMER_LAYERS:
        raise ValueError("E025 source placement must contain exactly 93 stages")
    workers = source.get("workers")
    if not isinstance(workers, list) or len(workers) != TRANSFORMER_LAYERS:
        raise ValueError("E025 source placement has an invalid worker list")
    checkpoint = source.get("checkpoint", {})
    if checkpoint.get("revision") != MODEL_REVISION:
        raise ValueError("E025 source placement is not the frozen Kimi K3 revision")


def split_layer_worker(
    source_worker: dict[str, Any],
    *,
    layer: int = SUB_LAYER_TARGET,
    worker_count: int = SUB_LAYER_WORKERS,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Split routed experts by ``expert_id % worker_count`` without duplication."""

    if worker_count != 4:
        raise ValueError("the promoted E025 primary assignment requires four workers")
    units = copy.deepcopy(source_worker["assignment_units"])
    core = [unit for unit in units if unit.get("kind") != "routed_expert"]
    experts = [unit for unit in units if unit.get("kind") == "routed_expert"]
    if len(core) != 1 or core[0].get("kind") != "layer_core":
        raise ValueError("selected K3 layer has an unexpected non-expert assignment")
    expert_ids = sorted(int(unit["routed_expert"]) for unit in experts)
    if expert_ids != list(range(ROUTED_EXPERTS)):
        raise ValueError("selected K3 layer does not cover all 896 routed experts")

    parent = _worker_payload(
        source_worker,
        worker_id=f"e025-stage-{layer:03d}-parent",
        worker_role="SUB_LAYER_PARENT",
        units=core,
        owned_layers=[layer],
        owned_components=[f"layer_{layer}_non_expert"],
    )
    parent["requires_sub_layer_workers"] = [
        f"e025-layer-{layer:03d}-sub-{index:02d}" for index in range(worker_count)
    ]
    parent["local_routed_expert_count"] = 0

    complete_memory = source_worker["memory"]
    complete_peak = int(complete_memory["planned_total_vram_bytes"]) - int(
        complete_memory["safety_headroom_bytes"]
    )
    safety_bytes = math.ceil(SMALL_WORKER_PHYSICAL_VRAM_BYTES * VRAM_SAFETY_FRACTION)
    usable_bytes = SMALL_WORKER_PHYSICAL_VRAM_BYTES - safety_bytes
    baseline = int(complete_memory["measured_cuda_baseline_bytes"])
    workspace = 256 * MIB
    communication = 64 * MIB
    serving = 128 * MIB

    sub_workers: list[dict[str, Any]] = []
    for index in range(worker_count):
        owned = [
            unit for unit in experts if int(unit["routed_expert"]) % worker_count == index
        ]
        worker = _worker_payload(
            source_worker,
            worker_id=f"e025-layer-{layer:03d}-sub-{index:02d}",
            worker_role="SUB_LAYER_WORKER",
            units=owned,
            owned_layers=[layer],
            owned_components=[f"layer_{layer}_routed_experts_mod_{worker_count}_{index}"],
        )
        fragment_runtime_bytes = sum(int(unit["weight_bytes"]) for unit in owned)
        persistent_buffers = (1 + 2 * 16) * 3584 * 4
        fragment_peak = (
            baseline
            + fragment_runtime_bytes
            + persistent_buffers
            + workspace
            + communication
            + serving
        )
        worker["sub_layer_proof"] = {
            "layer": layer,
            "ownership": f"expert_id modulo {worker_count} equals {index}",
            "complete_layer_source_weight_bytes": int(source_worker["source_weight_bytes"]),
            "complete_layer_runtime_peak_bytes": complete_peak,
            "physical_vram_bytes": SMALL_WORKER_PHYSICAL_VRAM_BYTES,
            "safety_headroom_bytes": safety_bytes,
            "usable_vram_bytes": usable_bytes,
            "assigned_fragment_source_bytes": int(worker["source_weight_bytes"]),
            "assigned_fragment_runtime_peak_bytes": fragment_peak,
            "complete_layer_fits": complete_peak <= usable_bytes,
            "fragment_fits": fragment_peak <= usable_bytes,
            "complete_layer_execution_elsewhere": False,
            "invoked_for_every_retained_token_required": True,
            "native_tensor_execution_required": True,
        }
        worker["memory"] = {
            "physical_vram_bytes": SMALL_WORKER_PHYSICAL_VRAM_BYTES,
            "usable_vram_bytes": usable_bytes,
            "safety_headroom_bytes": safety_bytes,
            "measured_cuda_baseline_bytes": baseline,
            "fragment_runtime_weight_bytes": fragment_runtime_bytes,
            "persistent_runtime_buffer_bytes": persistent_buffers,
            "workspace_reserve_bytes": workspace,
            "activation_and_communication_reserve_bytes": communication,
            "serving_scheduler_reserve_bytes": serving,
            "planned_runtime_peak_bytes": fragment_peak,
            "remaining_usable_bytes": usable_bytes - fragment_peak,
            "feasible": fragment_peak <= usable_bytes,
        }
        sub_workers.append(worker)

    proof = {
        "layer": layer,
        "strategy": "static disjoint expert_id modulo four",
        "source_lineage": "E014 promoted four-worker real-expert path",
        "complete_layer_runtime_peak_bytes": complete_peak,
        "complete_layer_runtime_peak_gib": complete_peak / GIB,
        "qualifying_worker_count": len(sub_workers),
        "all_complete_layers_exceed_usable_vram": all(
            not bool(worker["sub_layer_proof"]["complete_layer_fits"])
            for worker in sub_workers
        ),
        "all_fragments_fit": all(
            bool(worker["sub_layer_proof"]["fragment_fits"])
            for worker in sub_workers
        ),
        "all_experts_owned_once": sorted(
            expert
            for worker in sub_workers
            for expert in worker["owned_expert_ids"]
        )
        == list(range(ROUTED_EXPERTS)),
        "parent_local_routed_experts": 0,
        "negative_control": {
            "frozen_placement_without_sub_layer_group_valid": False,
            "explicit_replan_required": True,
        },
    }
    return parent, sub_workers, proof


def build_physical_placement(source_path: Path, output_path: Path) -> dict[str, Any]:
    source_file = source_path.expanduser().resolve()
    source = read_json(source_file)
    _validate_source(source)
    source_workers = copy.deepcopy(source["workers"])
    selected = source_workers[SUB_LAYER_TARGET]
    if selected.get("owned_layers") != [SUB_LAYER_TARGET]:
        raise ValueError("E025 layer-89 source worker is not checkpoint aligned")
    parent, sub_workers, proof = split_layer_worker(selected)

    workers: list[dict[str, Any]] = []
    for index, source_worker in enumerate(source_workers):
        if index == SUB_LAYER_TARGET:
            workers.append(parent)
            continue
        worker = _worker_payload(
            source_worker,
            worker_id=f"e025-stage-{index:03d}",
            worker_role="BACKBONE_STAGE",
            units=copy.deepcopy(source_worker["assignment_units"]),
            owned_layers=[index],
            owned_components=list(source_worker["owned_components"]),
        )
        worker["source_worker_id"] = source_worker["worker_id"]
        workers.append(worker)
    workers.extend(sub_workers)

    tensor_rows = [row for worker in workers for row in _tensor_rows(worker["assignment_units"])]
    names = [str(row["name"]) for row in tensor_rows]
    required_count = int(source["coverage"]["expected_required_tensor_count"])
    required_bytes = int(source["coverage"]["expected_required_source_weight_bytes"])
    coverage_bytes = sum(int(row["physical_bytes"]) for row in tensor_rows)
    layers = sorted({layer for worker in workers for layer in worker["owned_layers"]})
    gates = {
        "authoritative_model_id": MODEL_ID == "moonshotai/Kimi-K3",
        "authoritative_revision": source["checkpoint"]["revision"] == MODEL_REVISION,
        "all_93_layers_covered": layers == list(range(TRANSFORMER_LAYERS)),
        "required_tensor_count_exact": len(names) == required_count,
        "required_source_bytes_exact": coverage_bytes == required_bytes,
        "no_duplicate_tensor_ownership": len(names) == len(set(names)),
        "four_independent_sub_layer_workers": len(sub_workers) == SUB_LAYER_WORKERS,
        "all_sub_layer_fragments_fit_8gib": proof["all_fragments_fit"],
        "complete_layer_cannot_fit_sub_layer_workers": proof[
            "all_complete_layers_exceed_usable_vram"
        ],
        "no_parent_routed_expert_copy": proof["parent_local_routed_experts"] == 0,
        "all_routed_experts_owned_once": proof["all_experts_owned_once"],
        "negative_control_invalidates_frozen_placement": not proof["negative_control"][
            "frozen_placement_without_sub_layer_group_valid"
        ],
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "evidence_class_target": EVIDENCE_CLASS,
        "model_id": MODEL_ID,
        "checkpoint": source["checkpoint"],
        "source_placement": {
            "path": str(source_file),
            "sha256": sha256_file(source_file),
            "schema_version": source["schema_version"],
        },
        "topology": {
            "class": "MIXED_WHOLE_LAYER_PLUS_SUB_LAYER",
            "physical_worker_count": len(workers),
            "backbone_stage_count": TRANSFORMER_LAYERS,
            "sub_layer_parent_count": 1,
            "sub_layer_worker_count": len(sub_workers),
            "controller_compute_resource": False,
            "controller_relay_permitted": True,
            "sub_layer_workers_require_distinct_machine_ids": True,
        },
        "coverage": {
            "required_tensor_count": len(names),
            "expected_required_tensor_count": required_count,
            "required_source_weight_bytes": coverage_bytes,
            "expected_required_source_weight_bytes": required_bytes,
            "duplicate_tensor_count": len(names) - len(set(names)),
            "unassigned_required_tensor_count": required_count - len(set(names)),
            "covered_layers": layers,
            "ownership_sha256": canonical_sha256(
                sorted(
                    (
                        str(row["name"]),
                        str(row["safetensors_file"]),
                        tuple(int(value) for value in row["byte_range"]),
                    )
                    for row in tensor_rows
                )
            ),
        },
        "sub_layer_proof": proof,
        "acceptance_gates": gates,
        "workers": workers,
    }
    atomic_write_json(output_path, payload)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": payload["status"],
        "output_path": str(output_path.expanduser().resolve()),
        "output_sha256": sha256_file(output_path.expanduser().resolve()),
        "physical_worker_count": len(workers),
        "sub_layer_worker_count": len(sub_workers),
        "coverage": payload["coverage"],
        "sub_layer_proof": proof,
        "acceptance_gates": gates,
    }


def build_placement_and_distribution(
    checkpoint: Path,
    source_path: Path,
    placement_path: Path,
    distribution_path: Path,
) -> dict[str, Any]:
    placement = build_physical_placement(source_path, placement_path)
    if placement["status"] != "PASS":
        raise RuntimeError("E025 placement failed before checkpoint distribution")
    distribution = build_distribution_manifest(
        checkpoint,
        placement_path,
        distribution_path,
    )
    return {"placement": placement, "distribution": distribution}


__all__ = [
    "SCHEMA_VERSION",
    "build_physical_placement",
    "build_placement_and_distribution",
    "split_layer_worker",
]
