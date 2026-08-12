"""Final 93-worker Kimi K3 placement and immutable ownership evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "experiment-014-k3-final-physical-placement-v1"
GIB = 1024**3
MIB = 1024**2


def _read(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * MIB), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(destination)


def _tensor_rows(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [tensor for unit in units for tensor in unit["tensors"]]


def _attention_type(layer: int) -> str:
    return "Gated_MLA" if layer % 4 == 3 else "KDA"


def _merge_groups() -> list[list[int]]:
    return [[0, 1], *[[old] for old in range(2, 93)], [93, 94, 95]]


def build_final_placement(
    old_placement_path: Path,
    capacity_receipt_path: Path,
    stage_zero_receipt_path: Path,
    kda_batch_receipt_path: Path,
    mla_batch_receipt_path: Path,
    final_stage_receipt_path: Path,
    cuda_library_path: Path,
    promotion_receipt_path: Path,
    output_path: Path,
    *,
    cycle_id: str = "H014-037a",
) -> dict[str, Any]:
    """Merge exact old ownership into the canonical production 93-stage layout."""
    paths = {
        "old_placement": old_placement_path.resolve(),
        "capacity_receipt": capacity_receipt_path.resolve(),
        "stage_zero_receipt": stage_zero_receipt_path.resolve(),
        "kda_batch_receipt": kda_batch_receipt_path.resolve(),
        "mla_batch_receipt": mla_batch_receipt_path.resolve(),
        "final_stage_receipt": final_stage_receipt_path.resolve(),
        "cuda_library": cuda_library_path.resolve(),
        "promotion_receipt": promotion_receipt_path.resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")
    old = _read(paths["old_placement"])
    capacity = _read(paths["capacity_receipt"])
    stage_zero = _read(paths["stage_zero_receipt"])
    kda = _read(paths["kda_batch_receipt"])
    mla = _read(paths["mla_batch_receipt"])
    final = _read(paths["final_stage_receipt"])
    promotion = _read(paths["promotion_receipt"])
    for name, receipt in (
        ("old placement", old),
        ("capacity", capacity),
        ("stage zero", stage_zero),
        ("KDA batch", kda),
        ("MLA batch", mla),
        ("final stage", final),
    ):
        if receipt.get("status") != "PASS":
            raise ValueError(f"{name} receipt is not passing")
    if (
        promotion.get("status") != "PASS"
        or promotion.get("decision") != "PROMOTED_FOR_PRE_CANARY_USE"
    ):
        raise ValueError("CUDA promotion receipt is not passing")
    if int(old.get("node_count", 0)) != 96 or len(old.get("workers", [])) != 96:
        raise ValueError("source placement is not the immutable 96-worker layout")
    recommendation = capacity["recommended_experiment_015_topology"]
    if recommendation["candidate_id"] != "B" or int(recommendation["worker_count"]) != 93:
        raise ValueError("capacity receipt does not select the 93-worker candidate")
    decode_latency = capacity["decode_latency"]
    admission_rtt_ms = float(decode_latency["coarse_admission_rtt_ms"])
    admission_bandwidth_gbps = float(
        decode_latency["coarse_admission_bandwidth_gbps"]
    )
    admission_retention = float(
        decode_latency["coarse_admission_capacity_retention_percent"]
    )
    coarse_source_identity = capacity["sources"]["coarse_network"]
    coarse_source_path = Path(coarse_source_identity["path"])
    if (
        not coarse_source_path.is_file()
        or _sha256(coarse_source_path) != coarse_source_identity["sha256"]
        or admission_retention < 90.0
    ):
        raise ValueError("capacity coarse-network admission provenance is not passing")
    coarse_source = _read(coarse_source_path)
    activation_payload_bytes = int(
        coarse_source["measured_inputs"]["activation_payload_bytes"]
    )

    physical_vram = 24 * GIB
    baseline_bytes = int(
        capacity["memory"]["canonical_93_worker_maximum"][
            "measured_cuda_baseline_bytes"
        ]
    )
    safety_bytes = math.ceil(physical_vram * 0.10)
    workspace_reserve = 256 * MIB
    communication_reserve = 64 * MIB
    serving_reserve = 128 * MIB
    prefill_8k = next(
        row for row in capacity["prefill"] if int(row["context_tokens"]) == 8192
    )
    state_bytes_by_type = {
        "KDA": 8
        * int(
            kda["batches"]["8"]["performance"]["memory"][
                "session_state_bytes_each"
            ]
        ),
        "Gated_MLA": int(
            prefill_8k["maximum_state_bytes_on_one_worker_at_eight_streams"]
        ),
    }
    resident_by_role = {
        "stage_zero": int(stage_zero["stage"]["load"]["resident_device_bytes"]),
        "KDA": int(kda["load"]["resident_device_bytes"]),
        "Gated_MLA": int(mla["load"]["resident_device_bytes"]),
        "final": int(final["lifecycle"]["load"]["resident_device_bytes"]),
    }
    projected_components = capacity["decode_latency"]["compute_components_ms"]
    projected_service_by_role = {
        "stage_zero": float(projected_components["stage_zero"]),
        "KDA": float(projected_components["68_middle_kda_layers"]) / 68.0,
        "Gated_MLA": float(projected_components["23_mla_layers_at_8k"]) / 23.0,
        "final": float(projected_components["final_layer_head"]),
    }
    cuda_sha = _sha256(paths["cuda_library"])
    old_workers = {int(row["worker_index"]): row for row in old["workers"]}
    groups = _merge_groups()
    workers: list[dict[str, Any]] = []
    coverage_names: set[str] = set()
    coverage_digest = hashlib.sha256()
    duplicate_names: list[str] = []
    total_source_bytes = 0
    for worker_index, old_indices in enumerate(groups):
        source_workers = [old_workers[index] for index in old_indices]
        units = sorted(
            [unit for row in source_workers for unit in row["assignment_units"]],
            key=lambda row: str(row["unit_id"]),
        )
        tensors = sorted(_tensor_rows(units), key=lambda row: str(row["name"]))
        for tensor in tensors:
            name = str(tensor["name"])
            if name in coverage_names:
                duplicate_names.append(name)
            coverage_names.add(name)
            total_source_bytes += int(tensor["physical_bytes"])
            coverage_digest.update(
                json.dumps(
                    {
                        "name": name,
                        "source": tensor["safetensors_file"],
                        "byte_range": tensor["byte_range"],
                        "physical_bytes": tensor["physical_bytes"],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
        layers = sorted(
            {
                int(unit["layer"])
                for unit in units
                if unit.get("layer") is not None
            }
        )
        if worker_index == 0:
            role = "stage_zero"
            worker_role = "embedding_dense_stage"
            expected_layers = [0]
            owned_components = ["embedding", "layer_0"]
            state_bytes = state_bytes_by_type["KDA"]
        elif worker_index == 92:
            role = "final"
            worker_role = "final_head_stage"
            expected_layers = [92]
            owned_components = [
                "layer_92",
                "final_attention_residual",
                "final_norm",
                "lm_head",
            ]
            state_bytes = state_bytes_by_type["KDA"]
        else:
            expected_layers = [worker_index]
            role = _attention_type(worker_index)
            worker_role = "whole_layer_stage"
            owned_components = [f"layer_{worker_index}"]
            state_bytes = state_bytes_by_type[role]
        if layers != expected_layers:
            raise ValueError(
                f"worker {worker_index} has layers {layers}, expected {expected_layers}"
            )
        resident_bytes = resident_by_role[role]
        planned_without_safety = (
            baseline_bytes
            + resident_bytes
            + state_bytes
            + workspace_reserve
            + communication_reserve
            + serving_reserve
        )
        planned_total = planned_without_safety + safety_bytes
        remaining = physical_vram - planned_total
        source_bytes = sum(int(tensor["physical_bytes"]) for tensor in tensors)
        source_files = sorted({str(tensor["safetensors_file"]) for tensor in tensors})
        experts = [
            int(unit["routed_expert"])
            for unit in units
            if unit.get("routed_expert") is not None
        ]
        workers.append(
            {
                "worker_index": worker_index,
                "worker_id": f"k3-worker-{worker_index:03d}",
                "worker_role": worker_role,
                "topology_group": f"coarse-stage-{worker_index:03d}",
                "source_worker_indices": old_indices,
                "assignment_sha256": _canonical_sha256(units),
                "owned_components": owned_components,
                "owned_layers": layers,
                "owned_expert_count": len(experts),
                "owned_expert_ids": experts,
                "assignment_units": units,
                "tensor_count": len(tensors),
                "safetensor_source_files": source_files,
                "source_weight_bytes": source_bytes,
                "expected_on_disk_package_bytes_excluding_header": source_bytes,
                "memory": {
                    "physical_vram_bytes": physical_vram,
                    "measured_cuda_baseline_bytes": baseline_bytes,
                    "measured_or_representative_resident_device_bytes": resident_bytes,
                    "production_eight_stream_state_bytes": state_bytes,
                    "workspace_reserve_bytes": workspace_reserve,
                    "activation_and_communication_reserve_bytes": communication_reserve,
                    "serving_scheduler_reserve_bytes": serving_reserve,
                    "safety_headroom_bytes": safety_bytes,
                    "planned_total_vram_bytes": planned_total,
                    "remaining_unallocated_bytes": remaining,
                    "total_headroom_bytes": safety_bytes + remaining,
                    "feasible": remaining >= GIB,
                },
                "expected_compute": {
                    "projected_rtx3090_wall_p50_ms": projected_service_by_role[role],
                    "context_tokens": 8192,
                    "production_batch": 8 if worker_role == "whole_layer_stage" else 1,
                    "physical_status": "PROJECTED_PENDING_3090_CANARY",
                },
                "expected_network": {
                    "incoming_activation_payload_bytes": (
                        0 if worker_index == 0 else activation_payload_bytes
                    ),
                    "outgoing_activation_payload_bytes": (
                        0 if worker_index == 92 else activation_payload_bytes
                    ),
                    "edge_class": "kimi_coarse_stage_fp32_v1",
                    "maximum_rtt_ms": admission_rtt_ms,
                    "minimum_bandwidth_gbps": admission_bandwidth_gbps,
                },
                "model_and_runtime": {
                    "checkpoint_fingerprint": old["checkpoint"]["checkpoint_fingerprint"],
                    "model_revision": old["checkpoint"]["revision"],
                    "cuda_library_sha256": cuda_sha,
                    "cuda_target": "sm_86 SASS + compute_86 PTX",
                    "maximum_certified_routed_expert_batch": 8,
                    "candidate_pending_final_same_binary_regression": False,
                    "promoted_for_pre_canary_use": True,
                },
                "recovery": {
                    "replacement_assignment": f"k3-worker-{worker_index:03d}",
                    "generation_must_match": True,
                    "incomplete_stage_response_policy": "FAIL_CLOSED",
                },
                "checkpoint_fingerprint": old["checkpoint"]["checkpoint_fingerprint"],
            }
        )

    expected_tensor_count = int(old["checkpoint"]["required_tensor_count"])
    expected_source_bytes = int(old["checkpoint"]["required_weight_bytes"])
    layers = sorted({layer for worker in workers for layer in worker["owned_layers"]})
    minimum_remaining = min(
        int(worker["memory"]["remaining_unallocated_bytes"]) for worker in workers
    )
    gates = {
        "exact_93_workers": len(workers) == 93,
        "all_93_layers_covered_once": layers == list(range(93)),
        "required_tensor_count_exact": len(coverage_names) == expected_tensor_count,
        "required_source_bytes_exact": total_source_bytes == expected_source_bytes,
        "no_duplicate_tensor_ownership": not duplicate_names,
        "all_workers_memory_feasible": all(
            bool(worker["memory"]["feasible"]) for worker in workers
        ),
        "minimum_1gib_unallocated_after_safety": minimum_remaining >= GIB,
        "runtime_binary_matches_capacity_candidate": False,
        "runtime_binary_matches_promotion": False,
        "promotion_has_sm86_sass_and_compute86_ptx": False,
        "coarse_network_admission_matches_capacity": (
            admission_retention >= 90.0
            and admission_rtt_ms
            == float(coarse_source["recommendation"]["maximum_tested_rtt_ms"])
            and admission_bandwidth_gbps
            == float(coarse_source["recommendation"]["minimum_tested_bandwidth_gbps"])
        ),
    }
    # Capacity source hashes identify receipts, not their nested CUDA library. Verify it
    # against the exact nested candidate provenance instead.
    expected_cuda_sha = str(
        _read(Path(capacity["sources"]["contextual_route_mix"]["path"]))["sources"][
            "cuda_library"
        ]["sha256"]
    )
    gates["runtime_binary_matches_capacity_candidate"] = cuda_sha == expected_cuda_sha
    promoted_binary = promotion.get("final_binary", {})
    gates["runtime_binary_matches_promotion"] = (
        promoted_binary.get("sha256") == cuda_sha
    )
    gates["promotion_has_sm86_sass_and_compute86_ptx"] = (
        promoted_binary.get("sm86_sass") is True
        and promoted_binary.get("compute86_ptx") is True
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "cycle_id": cycle_id,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "node_count": 93,
        "topology": {
            "class": "WHOLE-LAYER",
            "candidate": "B",
            "worker_count": 93,
            "mapping_from_immutable_96_worker_manifest": groups,
            "normal_generation_rebuilds_topology": False,
            "sub_layer_workers_in_initial_fleet": False,
        },
        "checkpoint": old["checkpoint"],
        "runtime": {
            "cuda_library": str(paths["cuda_library"]),
            "cuda_library_sha256": cuda_sha,
            "target": "sm_86 SASS + compute_86 PTX",
            "status": "PROMOTED_FOR_PRE_CANARY_USE",
        },
        "source_artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "coverage": {
            "required_tensor_count": len(coverage_names),
            "expected_required_tensor_count": expected_tensor_count,
            "required_source_weight_bytes": total_source_bytes,
            "expected_required_source_weight_bytes": expected_source_bytes,
            "duplicate_tensor_count": len(duplicate_names),
            "duplicate_tensor_examples": duplicate_names[:10],
            "unassigned_required_tensor_count": expected_tensor_count - len(coverage_names),
            "covered_layers": layers,
            "canonical_ownership_sha256": coverage_digest.hexdigest(),
        },
        "memory_summary": {
            "physical_vram_bytes": physical_vram,
            "maximum_planned_vram_bytes": max(
                int(worker["memory"]["planned_total_vram_bytes"]) for worker in workers
            ),
            "minimum_remaining_unallocated_bytes": minimum_remaining,
            "minimum_total_headroom_bytes": min(
                int(worker["memory"]["total_headroom_bytes"]) for worker in workers
            ),
            "safety_headroom_bytes_per_worker": safety_bytes,
            "all_workers_feasible": all(
                bool(worker["memory"]["feasible"]) for worker in workers
            ),
        },
        "network_admission": {
            "edge_class": "kimi_coarse_stage_fp32_v1",
            "maximum_rtt_ms": admission_rtt_ms,
            "minimum_bandwidth_gbps": admission_bandwidth_gbps,
            "modeled_capacity_retention_percent": admission_retention,
            "activation_payload_bytes": activation_payload_bytes,
            "fine_edge_class_required": False,
        },
        "acceptance_gates": gates,
        "workers": workers,
    }
    _atomic_json(output_path, manifest)
    return {
        "schema_version": SCHEMA_VERSION,
        "cycle_id": cycle_id,
        "status": manifest["status"],
        "output_path": str(output_path.resolve()),
        "output_sha256": _sha256(output_path.resolve()),
        "worker_count": len(workers),
        "tensor_count": len(coverage_names),
        "source_weight_bytes": total_source_bytes,
        "minimum_remaining_unallocated_bytes": minimum_remaining,
        "canonical_ownership_sha256": coverage_digest.hexdigest(),
        "acceptance_gates": gates,
    }


__all__ = ["build_final_placement"]
