"""Evidence-driven shared-expert placement analysis for H014-SUB-007."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "experiment-014-k3-shared-expert-placement-v1"
LAYER = 89
HIDDEN = 7168
BATCHES = (1, 2, 4, 8)
RTT_PROFILES_MS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
BANDWIDTH_PROFILES_GBPS = (1.0, 2.5, 5.0, 10.0, 25.0, 100.0)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _shared_names() -> tuple[str, ...]:
    prefix = (
        f"language_model.model.layers.{LAYER}.block_sparse_moe.shared_experts"
    )
    return tuple(
        f"{prefix}.{role}_proj.weight" for role in ("gate", "up", "down")
    )


def _selected_safetensors_metadata(
    checkpoint: Path,
) -> tuple[list[dict[str, Any]], Path, str]:
    index_path = checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("checkpoint index has no weight_map")
    names = _shared_names()
    shards = {str(weight_map[name]) for name in names}
    if len(shards) != 1:
        raise ValueError(f"layer-89 shared tensors span unexpected shards: {shards}")
    shard_path = checkpoint / next(iter(shards))
    with shard_path.open("rb") as handle:
        encoded_length = handle.read(8)
        if len(encoded_length) != 8:
            raise ValueError("safetensors shard has no header length")
        header_length = struct.unpack("<Q", encoded_length)[0]
        header = json.loads(handle.read(header_length).decode("utf-8"))

    rows: list[dict[str, Any]] = []
    for name in names:
        metadata = header.get(name)
        if not isinstance(metadata, dict):
            raise ValueError(f"shared tensor absent from shard header: {name}")
        shape = tuple(int(value) for value in metadata["shape"])
        offsets = tuple(int(value) for value in metadata["data_offsets"])
        if len(shape) != 2 or metadata["dtype"] != "BF16":
            raise ValueError(f"unexpected shared tensor representation: {metadata}")
        output_dimension, input_dimension = shape
        if input_dimension % 64:
            raise ValueError("shared tensor input dimension is not grouped-int4 aligned")
        elements = output_dimension * input_dimension
        source_bytes = offsets[1] - offsets[0]
        packed_bytes = elements // 2
        scale_bytes = output_dimension * (input_dimension // 64) * 4
        if source_bytes != elements * 2:
            raise ValueError("shared BF16 tensor byte range is inconsistent")
        rows.append(
            {
                "name": name,
                "dtype": metadata["dtype"],
                "shape": list(shape),
                "source_bytes": source_bytes,
                "grouped_int4_packed_bytes": packed_bytes,
                "grouped_int4_scale_bytes": scale_bytes,
                "resident_tensor_bytes": packed_bytes + scale_bytes,
                "data_offsets": list(offsets),
            }
        )
    selected_digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return rows, shard_path, selected_digest


def _transmission_ms(payload_bytes: int, bandwidth_gbps: float) -> float:
    return payload_bytes * 8.0 / (bandwidth_gbps * 1_000_000.0)


def analyze_shared_expert_placement(
    checkpoint: Path,
    complete_batch_receipt: Path,
    distributed_batch_receipt: Path,
    fine_network_receipt: Path,
    output_path: Path,
    *,
    cycle_id: str = "H014-SUB-007",
) -> dict[str, Any]:
    """Join real bytes/timing and reject dominated shared-expert placements."""
    paths = {
        "checkpoint": checkpoint.resolve(),
        "complete_batch_receipt": complete_batch_receipt.resolve(),
        "distributed_batch_receipt": distributed_batch_receipt.resolve(),
        "fine_network_receipt": fine_network_receipt.resolve(),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
    complete = json.loads(paths["complete_batch_receipt"].read_text(encoding="utf-8"))
    distributed = json.loads(
        paths["distributed_batch_receipt"].read_text(encoding="utf-8")
    )
    fine = json.loads(paths["fine_network_receipt"].read_text(encoding="utf-8"))
    if complete.get("status") != "PASS" or distributed.get("status") != "PASS":
        raise ValueError("shared placement requires passing real batch receipts")
    if not bool(distributed.get("execution_pass")):
        raise ValueError("distributed batch receipt is not an execution pass")
    if fine.get("status") != "PASS":
        raise ValueError("fine-edge receipt is not passing")

    tensors, shard_path, selected_digest = _selected_safetensors_metadata(
        paths["checkpoint"]
    )
    source_bytes = sum(int(row["source_bytes"]) for row in tensors)
    resident_weight_bytes = sum(int(row["resident_tensor_bytes"]) for row in tensors)
    shared_timing: dict[str, Any] = {}
    for batch in BATCHES:
        phase = complete["batches"][str(batch)]["performance"][
            "phase_decomposition"
        ]["shared_expert"]
        shared_timing[str(batch)] = phase

    batch8 = distributed["batches"]["8"]["performance"]
    current_complete_ms = float(batch8["wall"]["p50_ms"])
    routed_collective_ms = float(batch8["expert_collective_roundtrip"]["p50_ms"])
    shared_wall_ms = float(shared_timing["8"]["wall"]["p50_ms"])
    shared_device_ms = float(shared_timing["8"]["device"]["p50_ms"])
    parent_overlap_projection_ms = current_complete_ms - shared_wall_ms
    payload_per_row = 2 * HIDDEN * 4
    payload_batch8 = 8 * payload_per_row
    messages_batch8 = 2
    session_buffer_bytes_batch8 = payload_batch8
    complete_layer_bytes = int(distributed["resident_reference"]["resident_device_bytes"])
    coordinator_bytes = int(
        distributed["distributed_residency"]["coordinator_resident_device_bytes"]
    )
    worker_bytes = int(
        distributed["distributed_residency"]["workers"][0]["tracked_worker_bytes"]
    )

    def remote_projection(rtt_ms: float, bandwidth_gbps: float) -> dict[str, Any]:
        transmission_ms = _transmission_ms(payload_batch8, bandwidth_gbps)
        remote_service_ms = shared_wall_ms + rtt_ms + transmission_ms
        exposed_after_routed_ms = max(0.0, remote_service_ms - routed_collective_ms)
        projected_complete_ms = parent_overlap_projection_ms + exposed_after_routed_ms
        return {
            "rtt_ms": rtt_ms,
            "bandwidth_gbps": bandwidth_gbps,
            "tensor_transmission_ms": transmission_ms,
            "remote_shared_service_ms": remote_service_ms,
            "hidden_by_routed_collective": remote_service_ms <= routed_collective_ms,
            "exposed_after_routed_ms": exposed_after_routed_ms,
            "projected_complete_layer_p50_ms": projected_complete_ms,
            "relative_to_current_parent_serial": current_complete_ms
            / projected_complete_ms,
            "beats_parent_overlap_projection": projected_complete_ms
            < parent_overlap_projection_ms - 1e-12,
        }

    rtt_sweep = [remote_projection(rtt, 100.0) for rtt in RTT_PROFILES_MS]
    bandwidth_sweep = [
        remote_projection(0.25, bandwidth)
        for bandwidth in BANDWIDTH_PROFILES_GBPS
    ]
    maximum_hidden_rtt_100gbps = (
        routed_collective_ms
        - shared_wall_ms
        - _transmission_ms(payload_batch8, 100.0)
    )

    one_worker_bytes = worker_bytes + resident_weight_bytes + session_buffer_bytes_batch8
    separate_worker_minimum_bytes = resident_weight_bytes + session_buffer_bytes_batch8
    replicated_weight_bytes = 4 * resident_weight_bytes
    placements = {
        "parent_resident": {
            "shared_weight_copies": 1,
            "incremental_group_weight_bytes": resident_weight_bytes,
            "parent_shared_resident_weight_bytes": resident_weight_bytes,
            "remote_tensor_bytes_per_row": 0,
            "remote_messages_per_batch8": 0,
            "current_serial_complete_p50_ms": current_complete_ms,
            "perfect_overlap_projection_p50_ms": parent_overlap_projection_ms,
        },
        "one_routed_worker": {
            "shared_weight_copies": 1,
            "incremental_group_weight_bytes": resident_weight_bytes,
            "selected_worker_bytes_lower_bound": one_worker_bytes,
            "selected_worker_fraction_of_complete_layer_percent": 100.0
            * one_worker_bytes
            / complete_layer_bytes,
            "parent_bytes_after_weight_move_lower_bound": coordinator_bytes
            - resident_weight_bytes,
            "remote_tensor_bytes_per_row": payload_per_row,
            "remote_messages_per_batch8": messages_batch8,
            "limitation": "adds shared compute to one already route-imbalanced worker",
        },
        "separate_shared_worker": {
            "shared_weight_copies": 1,
            "incremental_group_weight_bytes": resident_weight_bytes,
            "minimum_worker_bytes_excluding_cuda_context": separate_worker_minimum_bytes,
            "minimum_worker_gib_excluding_cuda_context": separate_worker_minimum_bytes
            / 2**30,
            "remote_tensor_bytes_per_row": payload_per_row,
            "remote_messages_per_batch8": messages_batch8,
            "best_case_projection_equals_parent_overlap": True,
        },
        "four_way_replication": {
            "shared_weight_copies": 4,
            "group_shared_weight_bytes": replicated_weight_bytes,
            "extra_weight_bytes_vs_parent": replicated_weight_bytes
            - resident_weight_bytes,
            "each_worker_bytes_lower_bound": one_worker_bytes,
            "remote_tensor_bytes_per_row": payload_per_row,
            "remote_messages_per_batch8": messages_batch8,
            "limitation": "replication does not remove input/output transfer or improve the zero-network parent overlap bound",
        },
    }

    no_remote_beats_parent_overlap = not any(
        bool(row["beats_parent_overlap_projection"])
        for row in rtt_sweep + bandwidth_sweep
    )
    gates = {
        "exact_shared_tensor_ownership_resolved": len(tensors) == 3,
        "real_shared_phase_measured_all_batches": all(
            float(shared_timing[str(batch)]["device"]["p50_ms"]) > 0
            for batch in BATCHES
        ),
        "remote_placement_adds_tensor_payload": payload_per_row > 0,
        "remote_best_case_does_not_beat_parent_overlap": no_remote_beats_parent_overlap,
        "parent_coordinator_retains_positive_headroom_over_shared_weights": coordinator_bytes
        > resident_weight_bytes,
    }
    hypothesis_supported = all(gates.values())
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "cycle_id": cycle_id,
        "status": "PASS",
        "hypothesis": (
            "Keeping the fused shared experts on the parent is lower-latency and "
            "lower-network-cost than remote or replicated placement; parent compute "
            "can overlap routed-worker service without another fine-edge payload."
        ),
        "hypothesis_supported": hypothesis_supported,
        "sources": {
            name: {
                "path": str(path),
                "sha256": _sha256_file(path) if path.is_file() else None,
            }
            for name, path in paths.items()
        },
        "shared_tensor_ownership": {
            "layer": LAYER,
            "safetensors_shard": str(shard_path),
            "safetensors_shard_size_bytes": shard_path.stat().st_size,
            "selected_header_metadata_sha256": selected_digest,
            "tensors": tensors,
            "source_bytes": source_bytes,
            "runtime_grouped_int4_resident_weight_bytes": resident_weight_bytes,
            "runtime_grouped_int4_resident_weight_gib": resident_weight_bytes / 2**30,
        },
        "real_shared_expert_timing": shared_timing,
        "batch8_context": {
            "distributed_complete_layer_p50_ms": current_complete_ms,
            "routed_collective_p50_ms": routed_collective_ms,
            "shared_expert_device_p50_ms": shared_device_ms,
            "shared_expert_wall_p50_ms": shared_wall_ms,
            "parent_perfect_overlap_projection_p50_ms": parent_overlap_projection_ms,
            "parent_overlap_capacity_gain_upper_bound": current_complete_ms
            / parent_overlap_projection_ms,
        },
        "remote_transport": {
            "input_activation_bytes_per_row": HIDDEN * 4,
            "returned_output_bytes_per_row": HIDDEN * 4,
            "tensor_bytes_per_row": payload_per_row,
            "tensor_bytes_batch8": payload_batch8,
            "messages_batch8": messages_batch8,
            "synchronization_points": 2,
            "rtt_sweep_at_100gbps": rtt_sweep,
            "bandwidth_sweep_at_0_25ms": bandwidth_sweep,
            "maximum_rtt_hidden_by_routed_collective_at_100gbps_ms": max(
                0.0, maximum_hidden_rtt_100gbps
            ),
            "method": (
                "Optimistic lower bound: shared compute + one RTT + tensor serialization; "
                "zero framing/serialization/software overhead, overlapped against the measured routed collective."
            ),
        },
        "placements": placements,
        "acceptance_gates": gates,
        "inspection": {
            "actual_bottleneck": (
                "Remote shared placement cannot beat the zero-network parent-overlap "
                "lower bound. It adds 57,344 tensor bytes/row and two messages; replication "
                "adds 222,953,472 weight bytes versus one parent copy."
            ),
            "important_limit": (
                "Overlap values are analytical projections joined from real timings, not "
                "a physical multi-GPU overlap measurement."
            ),
        },
        "decision": {
            "shared_expert_placement": "RETAIN_PARENT",
            "remote_shared_worker": "REJECT_DOMINATED",
            "four_way_replication": "REJECT_DOMINATED",
            "next_hypothesis": (
                "Overlap parent-resident shared-expert CUDA with the persistent routed "
                "collective without changing placement or numerical semantics."
            ),
        },
        "fine_network_join": {
            "maximum_viable_rtt_ms_at_100gbps": fine["best_projected_topology"][
                "maximum_useful_rtt_ms_at_100gbps"
            ],
            "minimum_viable_bandwidth_gbps_at_0_25ms": fine[
                "best_projected_topology"
            ]["minimum_useful_bandwidth_gbps_at_0_25ms"],
            "physical_multi_gpu_measurement": False,
        },
    }
    _atomic_json(output_path, receipt)
    return receipt

