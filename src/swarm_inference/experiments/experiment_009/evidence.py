"""Inventory reconciliation and evidence helpers for Experiment 009."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import psutil

from swarm_inference.backends.colibri.schemas import (
    ColibriCapabilityReport,
    ExpertInventoryEntry,
    RouteSelection,
    SwarmColibriPlan,
    TensorInventoryEntry,
    TierInventoryEntry,
)
from swarm_inference.microsharding.expert_abi import (
    ExpertMicroshardDescriptor,
    ExpertProjectionSlice,
    executable_microshard_equivalence,
    validate_expert_microshard_set,
)


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def environment_report(repository_root: Path) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        git_commit = None
    return {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "cpu": platform.processor(),
        "physical_cpu_cores": psutil.cpu_count(logical=False),
        "logical_cpu_cores": psutil.cpu_count(logical=True),
        "system_ram_total_bytes": memory.total,
        "system_ram_available_bytes": memory.available,
        "git_commit": git_commit or None,
        "command": list(sys.argv),
    }


def run_colibri_plan(
    *,
    engine_directory: Path,
    model_path: Path,
    capabilities: ColibriCapabilityReport,
    log_path: Path,
) -> dict[str, Any]:
    coli = engine_directory / "coli"
    if not coli.is_file():
        raise FileNotFoundError(f"Colibri CLI is missing: {coli}")
    command = [sys.executable, str(coli), "plan", "--model", str(model_path), "--json"]
    if not (
        capabilities.supports_cuda or capabilities.supports_vulkan or capabilities.supports_metal
    ):
        command.extend(("--gpu", "none"))
    result = subprocess.run(
        command,
        cwd=engine_directory,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"command={json.dumps(command)}\nexit_code={result.returncode}\n\nSTDOUT\n{result.stdout}"
        f"\nSTDERR\n{result.stderr}",
        encoding="utf-8",
    )
    if result.returncode:
        raise RuntimeError(f"coli plan exited {result.returncode}: {result.stderr[-2000:]}")
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise ValueError("Colibri planner did not return a JSON object")
    return value


def hardware_and_tiers(
    *,
    capabilities: ColibriCapabilityReport,
    native_plan: dict[str, Any],
    storage_profile: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[TierInventoryEntry]]:
    hardware_payload = {
        "capability_cpu": capabilities.cpu,
        "capability_gpu_devices": capabilities.gpu_devices,
        "memory": capabilities.memory,
        "storage": capabilities.storage,
        "execution_backends": capabilities.execution_backends,
        "storage_profile": storage_profile,
    }
    fingerprint = canonical_hash(hardware_payload)
    hardware = {"hardware_fingerprint": fingerprint, **hardware_payload}
    tiers = native_plan["tiers"]
    model = native_plan["model"]
    ram = tiers["ram"]
    vram = tiers["vram"]
    memory_total = int(capabilities.memory.get("total_bytes", 0))
    memory_available = int(capabilities.memory.get("available_bytes", 0))
    gpu_total = sum(int(item.get("total_memory_bytes", 0)) for item in capabilities.gpu_devices)
    gpu_available = sum(int(item.get("free_memory_bytes", 0)) for item in capabilities.gpu_devices)
    storage_devices = capabilities.storage.get("devices", [])
    selected_storage = next(
        (item for item in storage_devices if item.get("model_storage")),
        storage_devices[0] if storage_devices else {},
    )
    storage_total = int(selected_storage.get("total_bytes", model.get("model_bytes", 0)))
    storage_available = int(selected_storage.get("available_bytes", 0))
    formats = capabilities.quantization_formats
    results = [
        TierInventoryEntry(
            tier="vram",
            device_id=",".join(
                str(item.get("uuid", item.get("index"))) for item in capabilities.gpu_devices
            )
            or "unavailable",
            total_capacity_bytes=gpu_total,
            available_capacity_bytes=min(gpu_available, gpu_total),
            allocated_model_bytes=0,
            allocated_expert_bytes=int(vram.get("hot_expert_bytes", 0)),
            mutable_state_bytes=0,
            temporary_working_bytes=0,
            measured_bandwidth_bytes_per_second=None,
            measured_latency_ms=None,
            supported_formats=(
                formats
                if capabilities.supports_cuda
                or capabilities.supports_vulkan
                or capabilities.supports_metal
                else []
            ),
            eviction_policy="colibri_hot_expert_tier"
            if int(vram.get("hot_expert_bytes", 0))
            else "not_executable",
        ),
        TierInventoryEntry(
            tier="ram",
            device_id="system-ram",
            total_capacity_bytes=memory_total,
            available_capacity_bytes=min(memory_available, memory_total),
            allocated_model_bytes=int(model.get("dense_bytes", 0)),
            allocated_expert_bytes=int(ram.get("warm_expert_bytes", 0)),
            mutable_state_bytes=int(ram.get("runtime_bytes", 0)),
            temporary_working_bytes=0,
            measured_bandwidth_bytes_per_second=None,
            measured_latency_ms=None,
            supported_formats=formats,
            eviction_policy="colibri_hot_plus_lru_expert_cache",
        ),
        TierInventoryEntry(
            tier="nvme",
            device_id=str(
                selected_storage.get("device", selected_storage.get("mountpoint", "unknown"))
            ),
            total_capacity_bytes=max(storage_total, int(model.get("model_bytes", 0))),
            available_capacity_bytes=min(
                storage_available, max(storage_total, int(model.get("model_bytes", 0)))
            ),
            allocated_model_bytes=int(model.get("dense_bytes", 0)),
            # Colibri keeps immutable recovery bytes on storage even when an
            # expert also has a warm RAM or hot VRAM copy.
            allocated_expert_bytes=int(model.get("expert_bytes", 0)),
            mutable_state_bytes=0,
            temporary_working_bytes=0,
            measured_bandwidth_bytes_per_second=(
                (storage_profile or {})
                .get("random_expert_reads", {})
                .get("bandwidth_bytes_per_second")
            ),
            measured_latency_ms=(storage_profile or {})
            .get("random_expert_reads", {})
            .get("median_latency_ms"),
            supported_formats=formats,
            eviction_policy="immutable_cold_storage",
        ),
    ]
    return hardware, results


def route_tables(selections: list[RouteSelection]) -> dict[str, Any]:
    from swarm_inference.backends.colibri.telemetry import ColibriRouteTraceReader

    summary = ColibriRouteTraceReader.summarize(selections)
    first_seen: dict[tuple[str, int, int], int] = {}
    for row in selections:
        key = (row.phase, row.layer_id, row.expert_id)
        first_seen[key] = min(first_seen.get(key, row.call_index), row.call_index)
    activation = []
    for row in summary["activation"]:
        summary_key = (str(row["phase"]), int(row["layer_id"]), int(row["expert_id"]))
        activation.append({**row, "first_call_index": first_seen[summary_key]})
    tier_counts: dict[str, int] = defaultdict(int)
    for row in selections:
        tier_counts[row.execution_tier or "unknown"] += 1
    tier_hits = [
        {"tier": tier, "expert_selections": count} for tier, count in sorted(tier_counts.items())
    ]
    return {
        "activation": activation,
        "coactivation": summary["coactivation"],
        "transitions": summary["transitions"],
        "tier_hits": tier_hits,
        "summary": summary,
    }


def _segments_for_projection(
    *,
    tensor: TensorInventoryEntry,
    projection: str,
    logical_shape: list[int],
    start: int,
    end: int,
    projection_offset: int = 0,
    element_bytes: int = 1,
) -> tuple[int, int, list[dict[str, int]]]:
    base = tensor.storage_offset + projection_offset
    if projection in {"up", "gate"}:
        row_bytes = logical_shape[1] * element_bytes
        offset = base + start * row_bytes
        length = (end - start) * row_bytes
        return offset, length, [{"offset": offset, "length": length}]
    row_width = logical_shape[1]
    segments = [
        {
            "offset": base + row * row_width * element_bytes + start * element_bytes,
            "length": (end - start) * element_bytes,
        }
        for row in range(logical_shape[0])
    ]
    return (
        min(item["offset"] for item in segments),
        sum(item["length"] for item in segments),
        segments,
    )


def _projection_slice(
    *,
    tensor: TensorInventoryEntry,
    projection: Literal["up", "gate", "down"],
    shape: list[int],
    start: int,
    end: int,
    projection_offset: int = 0,
    element_bytes: int = 1,
) -> ExpertProjectionSlice:
    offset, length, segments = _segments_for_projection(
        tensor=tensor,
        projection=projection,
        logical_shape=shape,
        start=start,
        end=end,
        projection_offset=projection_offset,
        element_bytes=element_bytes,
    )
    return ExpertProjectionSlice(
        tensor_id=tensor.tensor_id,
        tensor_name=tensor.tensor_name,
        projection=projection,
        logical_axis=1 if projection == "down" else 0,
        slice_start=start,
        slice_end=end,
        logical_shape=shape,
        storage_file=tensor.storage_file,
        storage_offset=offset,
        storage_length=length,
        storage_file_size=Path(tensor.storage_file).stat().st_size,
        storage_segments=segments,
        content_hash=f"{tensor.content_hash}:{projection}:{start}:{end}",
    )


def build_microshard_evidence(
    *,
    tensors: list[TensorInventoryEntry],
    experts: list[ExpertInventoryEntry],
    model_config: dict[str, Any],
    model_id: str,
) -> tuple[list[ExpertMicroshardDescriptor], dict[str, Any]]:
    by_id = {tensor.tensor_id: tensor for tensor in tensors}
    selected = next((expert for expert in experts if expert.tensor_ids), None)
    if selected is None:
        return [], {"valid": False, "reason": "no routed expert inventory"}
    items = [by_id[tensor_id] for tensor_id in selected.tensor_ids]
    roles = {item.tensor_role: item for item in items}
    up = roles.get("routed_expert_up_projection")
    gate = roles.get("routed_expert_gate_projection")
    down = roles.get("routed_expert_down_projection")
    effective = model_config.get("text_config", model_config)
    if not isinstance(effective, dict):
        effective = model_config
    hidden = int(
        effective.get("routed_expert_hidden_size", 0) or effective.get("hidden_size", 0) or 0
    )
    intermediate = int(
        effective.get("moe_intermediate_size", 0) or effective.get("intermediate_size", 0) or 0
    )
    merged = roles.get("routed_expert_merged_weight")
    merged_offsets: dict[str, int] = {}
    element_bytes = 1
    if not all((up, gate, down)) and merged is not None and hidden and intermediate:
        up = gate = down = merged
        block = hidden * intermediate
        merged_offsets = {"gate": 0, "up": block, "down": 2 * block}
    if not all((up, gate, down)):
        return [], {"valid": False, "reason": "expert projections cannot be sliced logically"}
    assert up is not None and gate is not None and down is not None
    if not intermediate:
        intermediate = up.logical_shape[0]
    if not hidden:
        hidden = up.logical_shape[1]
    up_shape, gate_shape, down_shape = (
        [intermediate, hidden],
        [intermediate, hidden],
        [hidden, intermediate],
    )
    if not merged_offsets:
        up_shape, gate_shape, down_shape = up.logical_shape, gate.logical_shape, down.logical_shape
        intermediate = up_shape[0]
        hidden = up_shape[1]
        packed_elements = math.prod(up.quantization.packed_shape or up.logical_shape)
        element_bytes = max(1, up.byte_size // max(1, packed_elements))
    quant = (
        up.quantization.model_copy(
            update={"logical_shape": up_shape, "packed_shape": up_shape, "byte_size": up.byte_size}
        )
        if merged_offsets
        else up.quantization
    )
    group = quant.scale_group_size or 1
    midpoint = max(group, (intermediate // 2 // group) * group)
    if midpoint >= intermediate:
        ranges = [(0, intermediate)]
    else:
        ranges = [(0, midpoint), (midpoint, intermediate)]
    descriptors = []
    for _index, (start, end) in enumerate(ranges):
        descriptors.append(
            ExpertMicroshardDescriptor(
                model_id=model_id,
                layer_id=selected.layer_id,
                expert_id=selected.expert_id,
                shard_id=f"L{selected.layer_id}-E{selected.expert_id}-H{start}-{end}",
                hidden_start=start,
                hidden_end=end,
                up_projection=_projection_slice(
                    tensor=up,
                    projection="up",
                    shape=up_shape,
                    start=start,
                    end=end,
                    projection_offset=merged_offsets.get("up", 0),
                    element_bytes=element_bytes,
                ),
                gate_projection=_projection_slice(
                    tensor=gate,
                    projection="gate",
                    shape=gate_shape,
                    start=start,
                    end=end,
                    projection_offset=merged_offsets.get("gate", 0),
                    element_bytes=element_bytes,
                ),
                down_projection=_projection_slice(
                    tensor=down,
                    projection="down",
                    shape=down_shape,
                    start=start,
                    end=end,
                    projection_offset=merged_offsets.get("down", 0),
                    element_bytes=element_bytes,
                ),
                native_quantization=quant,
                required_accumulator="float32_sum",
                supported_backends=[],
                execution_status="unsupported",
            )
        )
    validation: dict[str, Any] = validate_expert_microshard_set(descriptors)
    rng = np.random.default_rng(9009)
    # This is an execution-equivalence fixture, not a stress test of exp() or
    # BLAS summation at unrealistically large random activations.  Small,
    # deterministic values keep the comparison in a representative finite
    # range while retaining non-zero contributions from every logical shard.
    fixture_scale = 0.05
    validation["fixture_equivalence"] = executable_microshard_equivalence(
        inputs=rng.normal(scale=fixture_scale, size=(3, hidden)).astype(np.float32),
        up=rng.normal(scale=fixture_scale, size=(intermediate, hidden)).astype(np.float32),
        gate=rng.normal(scale=fixture_scale, size=(intermediate, hidden)).astype(np.float32),
        down=rng.normal(scale=fixture_scale, size=(hidden, intermediate)).astype(np.float32),
        ranges=ranges,
    )
    validation["real_execution_status"] = "unsupported"
    return descriptors, validation


def plan_tier_rows(plan: SwarmColibriPlan) -> list[dict[str, Any]]:
    rows = []
    for tier, payload in plan.routed_expert_tiers.items():
        if isinstance(payload, dict):
            rows.append(
                {
                    "tier": tier,
                    "allocated_expert_bytes": int(payload.get("bytes", 0)),
                    "role": payload.get("role"),
                    "expert_capacity": payload.get("expert_capacity"),
                    "cache_slots_per_layer": payload.get("cache_slots_per_layer"),
                }
            )
    return rows
