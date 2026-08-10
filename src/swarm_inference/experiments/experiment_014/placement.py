"""Exact Kimi K3 tensor ownership and RTX 3090 node-count solver.

The solver uses exact Safetensors payload bytes.  Routed-expert packed weights
and their scales are indivisible six-tensor units; all non-expert tensors for a
transformer layer form one executable core unit.  This is conservative about
resident bytes and explicit about every non-weight VRAM reserve.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_014.census import (
    _iter_tensors,
    _layer_attention_types,
    _load_json,
    _revision,
    classify_tensor,
)

SCHEMA_VERSION = "experiment-014-k3-placement-v1"
SOLVER_SCHEMA_VERSION = "experiment-014-k3-node-count-solver-v1"
GIB = 1024**3
DEFAULT_CANDIDATES = (64, 72, 73, 80, 88, 96, 112, 128)


class PlacementError(ValueError):
    """Exact checkpoint placement is inconsistent or infeasible."""


@dataclass(slots=True)
class TensorRange:
    name: str
    safetensors_file: str
    byte_range: list[int]
    physical_bytes: int
    dtype: str
    shape: list[int]


@dataclass(slots=True)
class PlacementUnit:
    unit_id: str
    kind: str
    layer: int | None
    attention_type: str | None
    routed_expert: int | None
    tensors: list[TensorRange] = field(default_factory=list)
    weight_bytes: int = 0
    state_bytes: int = 0

    @property
    def effective_bytes(self) -> int:
        return self.weight_bytes + self.state_bytes


@dataclass(frozen=True, slots=True)
class MemoryPolicy:
    physical_vram_bytes: int = 24 * GIB
    cuda_context_bytes: int = 768 * 1024**2
    runtime_library_bytes: int = 512 * 1024**2
    workspace_bytes: int = 1024**3
    activation_buffer_bytes: int = 256 * 1024**2
    reduction_buffer_bytes: int = 256 * 1024**2
    communication_buffer_bytes: int = 512 * 1024**2
    serving_runtime_bytes: int = 256 * 1024**2
    max_context: int = 8192
    max_active_streams: int = 4
    operational_headroom_fraction: float = 0.10

    @property
    def fixed_runtime_reserve_bytes(self) -> int:
        return (
            self.cuda_context_bytes
            + self.runtime_library_bytes
            + self.workspace_bytes
            + self.activation_buffer_bytes
            + self.reduction_buffer_bytes
            + self.communication_buffer_bytes
            + self.serving_runtime_bytes
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_bytes_for_layer(
    config: Mapping[str, Any],
    attention_type: str,
    policy: MemoryPolicy,
) -> int:
    linear = config["linear_attn_config"]
    if attention_type == "kda":
        heads = int(linear["num_heads"])
        head_dim = int(linear["head_dim"])
        projection = heads * head_dim
        convolution = int(linear["short_conv_kernel_size"])
        per_stream = (heads * head_dim * head_dim + 3 * projection * convolution) * 4
    else:
        per_stream = (
            policy.max_context * (int(config["kv_lora_rank"]) + int(config["qk_rope_head_dim"])) * 4
        )
    return per_stream * policy.max_active_streams


def _unit_key(classification: Any) -> tuple[str, int | None, int | None]:
    if classification.routed_expert is not None:
        return ("routed_expert", classification.layer, classification.routed_expert)
    if classification.layer is not None:
        return ("layer_core", classification.layer, None)
    return (classification.component, None, None)


def build_units(
    checkpoint: Path, policy: MemoryPolicy
) -> tuple[list[PlacementUnit], dict[str, Any]]:
    root = checkpoint.expanduser().resolve()
    config = _load_json(root / "config.json")
    text = config.get("text_config")
    if not isinstance(text, Mapping):
        raise PlacementError("checkpoint config has no text_config")
    index = _load_json(root / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise PlacementError("checkpoint index has no weight map")
    shard_names = tuple(sorted({str(value) for value in weight_map.values()}))
    attention = _layer_attention_types(config)
    units: dict[tuple[str, int | None, int | None], PlacementUnit] = {}
    required_names: set[str] = set()
    for shard_name, tensor in _iter_tensors(root, shard_names):
        classification = classify_tensor(tensor.name, config)
        if not classification.required_for_text_generation:
            continue
        if tensor.name in required_names:
            raise PlacementError(f"duplicate required tensor {tensor.name}")
        required_names.add(tensor.name)
        key = _unit_key(classification)
        unit = units.get(key)
        if unit is None:
            kind, layer, expert = key
            suffix = f"-layer-{layer:02d}" if layer is not None else ""
            if expert is not None:
                suffix += f"-expert-{expert:03d}"
            unit = PlacementUnit(
                unit_id=f"{kind}{suffix}",
                kind=kind,
                layer=layer,
                attention_type=attention[layer] if layer is not None else None,
                routed_expert=expert,
            )
            if kind == "layer_core" and layer is not None:
                unit.state_bytes = _state_bytes_for_layer(text, attention[layer], policy)
            units[key] = unit
        record = TensorRange(
            name=tensor.name,
            safetensors_file=shard_name,
            byte_range=[tensor.data_offset, tensor.data_offset + tensor.byte_size],
            physical_bytes=tensor.byte_size,
            dtype=tensor.dtype,
            shape=list(tensor.shape),
        )
        unit.tensors.append(record)
        unit.weight_bytes += tensor.byte_size
    if required_names != set(weight_map) - {
        name
        for name in weight_map
        if name.startswith("vision_tower.") or name.startswith("mm_projector.")
    }:
        missing = sorted(set(weight_map) - required_names)[:10]
        raise PlacementError(
            f"required tensor ownership did not cover the text checkpoint: {missing}"
        )
    ordered = sorted(units.values(), key=lambda unit: (-unit.effective_bytes, unit.unit_id))
    expert_units = [unit for unit in ordered if unit.kind == "routed_expert"]
    expected_experts = (int(text["num_hidden_layers"]) - int(text["first_k_dense_replace"])) * int(
        text["num_experts"]
    )
    if len(expert_units) != expected_experts or any(
        len(unit.tensors) != 6 for unit in expert_units
    ):
        raise PlacementError("routed-expert units are not exact six-tensor checkpoint units")
    identity = {
        "revision": _revision(root, shard_names),
        "config_sha256": _sha256(root / "config.json"),
        "index_sha256": _sha256(root / "model.safetensors.index.json"),
        "required_tensor_count": len(required_names),
        "required_weight_bytes": sum(unit.weight_bytes for unit in ordered),
        "shard_count": len(shard_names),
    }
    identity["checkpoint_fingerprint"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ordered, identity


def _coarse_stage_assignment(
    units: list[PlacementUnit], node_count: int
) -> tuple[list[list[PlacementUnit]], list[int]] | None:
    """Return the checkpoint-aligned 96-stage layout when its invariants hold."""

    if node_count != 96:
        return None
    by_layer: dict[int, list[PlacementUnit]] = {layer: [] for layer in range(93)}
    globals_by_kind: dict[str, list[PlacementUnit]] = {}
    for unit in units:
        if unit.layer is not None:
            by_layer[unit.layer].append(unit)
        else:
            globals_by_kind.setdefault(unit.kind, []).append(unit)
    required_globals = {
        "embedding",
        "final_norm",
        "final_attention_residual",
        "lm_head",
    }
    if any(not rows for rows in by_layer.values()) or set(globals_by_kind) != required_globals:
        return None
    if any(len(globals_by_kind[kind]) != 1 for kind in required_globals):
        return None
    assignments: list[list[PlacementUnit]] = [[] for _ in range(node_count)]
    assignments[0].extend(globals_by_kind["embedding"])
    for layer in range(93):
        assignments[layer + 1].extend(by_layer[layer])
    assignments[94].extend(globals_by_kind["final_attention_residual"])
    assignments[94].extend(globals_by_kind["final_norm"])
    assignments[95].extend(globals_by_kind["lm_head"])
    effective = [sum(unit.effective_bytes for unit in rows) for rows in assignments]
    return assignments, effective


def _assign(
    units: list[PlacementUnit], node_count: int
) -> tuple[list[list[PlacementUnit]], list[int]]:
    if node_count < 1:
        raise PlacementError("node count must be positive")
    coarse = _coarse_stage_assignment(units, node_count)
    if coarse is not None:
        return coarse
    assignments: list[list[PlacementUnit]] = [[] for _ in range(node_count)]
    effective = [0] * node_count
    heap = [(0, node) for node in range(node_count)]
    heapq.heapify(heap)
    for unit in units:
        _, node = heapq.heappop(heap)
        assignments[node].append(unit)
        effective[node] += unit.effective_bytes
        heapq.heappush(heap, (effective[node], node))
    return assignments, effective


def _candidate_result(
    units: list[PlacementUnit],
    node_count: int,
    policy: MemoryPolicy,
) -> dict[str, Any]:
    assignments, effective = _assign(units, node_count)
    strategy = (
        "checkpoint_aligned_coarse_stage"
        if _coarse_stage_assignment(units, node_count) is not None
        else "balanced_fine_grained"
    )
    weights = [sum(unit.weight_bytes for unit in rows) for rows in assignments]
    states = [sum(unit.state_bytes for unit in rows) for rows in assignments]
    scenarios: dict[str, Any] = {}
    for headroom in (0.0, 0.05, 0.10):
        reserved_headroom = math.ceil(policy.physical_vram_bytes * headroom)
        totals = [
            weight + state + policy.fixed_runtime_reserve_bytes + reserved_headroom
            for weight, state in zip(weights, states, strict=True)
        ]
        scenarios[f"{int(headroom * 100)}pct"] = {
            "headroom_fraction": headroom,
            "headroom_bytes": reserved_headroom,
            "feasible": max(totals) <= policy.physical_vram_bytes,
            "worst_total_vram_bytes": max(totals),
            "minimum_remaining_bytes": policy.physical_vram_bytes - max(totals),
        }
    average = sum(effective) / node_count
    return {
        "node_count": node_count,
        "placement_strategy": strategy,
        "memory_feasible_at_operational_headroom": scenarios["10pct"]["feasible"],
        "headroom_scenarios": scenarios,
        "maximum_weight_bytes": max(weights),
        "median_weight_bytes": sorted(weights)[len(weights) // 2],
        "maximum_state_bytes": max(states),
        "effective_load_imbalance_max_over_mean": max(effective) / average,
        "bottleneck_stage": "maximum resident weight/state node",
        "communication": (
            "one hidden-state handoff per layer; selected experts stay local to the layer stage"
            if strategy == "checkpoint_aligned_coarse_stage"
            else "global fine-grained expert dispatch across weight-balanced workers"
        ),
        "expected_service_rate": None,
        "service_rate_status": "NOT_AVAILABLE_UNTIL_VALIDATED_CAPACITY_MODEL",
        "gpu_hourly_cost_usd": node_count * 0.165,
    }


def solve_node_counts(
    units: list[PlacementUnit],
    policy: MemoryPolicy,
    candidates: tuple[int, ...] = DEFAULT_CANDIDATES,
) -> dict[str, Any]:
    search = [
        _candidate_result(units, count, policy)
        for count in range(min(candidates), max(candidates) + 1)
    ]
    feasible = [row for row in search if row["memory_feasible_at_operational_headroom"]]
    absolute_minimum = feasible[0]["node_count"] if feasible else None
    if absolute_minimum is None:
        recommended = None
    else:
        coarse_96 = next((row for row in feasible if row["node_count"] == 96), None)
        if coarse_96 is not None:
            recommended = 96
        else:
            capacity_reserve_target = math.ceil(absolute_minimum / 0.90)
            recommended = next(
                (count for count in candidates if count >= capacity_reserve_target), None
            )
            if recommended is None:
                recommended = capacity_reserve_target
    selected = {row["node_count"]: row for row in search}
    return {
        "schema_version": SOLVER_SCHEMA_VERSION,
        "status": "PASS" if absolute_minimum is not None else "FAIL",
        "memory_policy": asdict(policy),
        "absolute_minimum_node_count": absolute_minimum,
        "recommended_operational_node_count": recommended,
        "recommendation_rule": (
            "prefer the feasible 96-node checkpoint-aligned coarse pipeline (93 transformer "
            "stages plus embedding/final/head); otherwise retain at least 10% node-capacity "
            "reserve above the 10%-VRAM-headroom absolute minimum"
        ),
        "candidates": [selected[count] for count in candidates],
    }


def _node_payload(
    node: int,
    rows: list[PlacementUnit],
    policy: MemoryPolicy,
    checkpoint_identity: Mapping[str, Any],
) -> dict[str, Any]:
    weight_bytes = sum(unit.weight_bytes for unit in rows)
    state_bytes = sum(unit.state_bytes for unit in rows)
    core = sorted(
        unit.layer for unit in rows if unit.kind == "layer_core" and unit.layer is not None
    )
    experts = [
        {"layer": unit.layer, "expert": unit.routed_expert}
        for unit in rows
        if unit.kind == "routed_expert"
    ]
    source_files = sorted({tensor.safetensors_file for unit in rows for tensor in unit.tensors})
    headroom = math.ceil(policy.physical_vram_bytes * policy.operational_headroom_fraction)
    total = weight_bytes + state_bytes + policy.fixed_runtime_reserve_bytes + headroom
    return {
        "worker_index": node,
        "worker_id": f"k3-worker-{node:03d}",
        "worker_role": "stage_and_expert" if core and experts else "stage" if core else "expert",
        "stage": core if core else "expert_pool",
        "owned_layers": core,
        "owned_experts": experts,
        "assignment_units": [
            {
                "unit_id": unit.unit_id,
                "kind": unit.kind,
                "layer": unit.layer,
                "attention_type": unit.attention_type,
                "routed_expert": unit.routed_expert,
                "weight_bytes": unit.weight_bytes,
                "state_bytes": unit.state_bytes,
                "tensors": [asdict(tensor) for tensor in unit.tensors],
            }
            for unit in sorted(rows, key=lambda value: value.unit_id)
        ],
        "tensor_count": sum(len(unit.tensors) for unit in rows),
        "safetensor_source_files": source_files,
        "expected_on_disk_bytes": weight_bytes,
        "expected_resident_weight_bytes": weight_bytes,
        "memory": {
            "resident_weights_bytes": weight_bytes,
            "quantization_scales_included_in_weights": True,
            "runtime_library_bytes": policy.runtime_library_bytes,
            "cuda_context_bytes": policy.cuda_context_bytes,
            "workspace_bytes": policy.workspace_bytes,
            "activation_buffer_bytes": policy.activation_buffer_bytes,
            "reduction_buffer_bytes": policy.reduction_buffer_bytes,
            "communication_buffer_bytes": policy.communication_buffer_bytes,
            "kda_and_gated_mla_state_bytes": state_bytes,
            "serving_runtime_bytes": policy.serving_runtime_bytes,
            "safety_headroom_bytes": headroom,
            "total_planned_vram_bytes": total,
            "physical_vram_bytes": policy.physical_vram_bytes,
            "feasible": total <= policy.physical_vram_bytes,
        },
        "expected_compute_load": {
            "core_layer_count": len(core),
            "routed_expert_count": len(experts),
            "status": "WEIGHT_BALANCED_NOT_PERFORMANCE_CERTIFIED",
        },
        "expected_incoming_tensors": [
            "hidden activation for owned core layer",
            "latent expert activation for selected owned expert",
        ],
        "expected_outgoing_tensors": [
            "next-layer hidden activation",
            "weighted expert contribution to immediate parent reduction",
        ],
        "parent": "coordinator-or-immediate-stage-parent",
        "children": [],
        "recovery_relationship": "manifest-identical replacement and shard rehydration; request restart",
        "checkpoint_fingerprint": checkpoint_identity["checkpoint_fingerprint"],
    }


def write_placement_artifacts(
    checkpoint: Path,
    manifest_path: Path,
    solver_path: Path,
    *,
    manifest_node_count: int = 73,
    policy: MemoryPolicy | None = None,
    candidates: tuple[int, ...] = DEFAULT_CANDIDATES,
) -> dict[str, Any]:
    selected_policy = policy or MemoryPolicy()
    units, identity = build_units(checkpoint, selected_policy)
    solver = solve_node_counts(units, selected_policy, candidates)
    solver.update(
        {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "checkpoint": identity,
            "unit_count": len(units),
        }
    )
    solver_destination = solver_path.expanduser().resolve()
    solver_destination.parent.mkdir(parents=True, exist_ok=True)
    solver_temp = solver_destination.with_suffix(solver_destination.suffix + ".partial")
    solver_temp.write_text(json.dumps(solver, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    solver_temp.replace(solver_destination)

    assignments, _ = _assign(units, manifest_node_count)
    workers = [
        _node_payload(node, rows, selected_policy, identity)
        for node, rows in enumerate(assignments)
    ]
    required_tensor_count = sum(worker["tensor_count"] for worker in workers)
    required_weight_bytes = sum(worker["expected_on_disk_bytes"] for worker in workers)
    all_feasible = all(worker["memory"]["feasible"] for worker in workers)
    layers = sorted({layer for worker in workers for layer in worker["owned_layers"]})
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS" if all_feasible else "FAIL",
        "node_count": manifest_node_count,
        "checkpoint": identity,
        "memory_policy": asdict(selected_policy),
        "coverage": {
            "required_tensor_count": required_tensor_count,
            "expected_required_tensor_count": identity["required_tensor_count"],
            "required_weight_bytes": required_weight_bytes,
            "expected_required_weight_bytes": identity["required_weight_bytes"],
            "unassigned_required_tensors": 0,
            "duplicate_required_tensors": 0,
            "covered_layers": layers,
            "all_93_layers_covered": layers == list(range(93)),
        },
        "distribution_mode": "manifest-driven exact range extraction; no full checkpoint per worker",
        "workers": workers,
    }
    if (
        required_tensor_count != identity["required_tensor_count"]
        or required_weight_bytes != identity["required_weight_bytes"]
    ):
        raise PlacementError("manifest totals changed during assignment")
    destination = manifest_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": manifest["status"],
        "manifest_path": str(destination),
        "manifest_sha256": _sha256(destination),
        "solver_path": str(solver_destination),
        "solver_sha256": _sha256(solver_destination),
        "node_count": manifest_node_count,
        "covered_layers": layers,
        "required_tensor_count": required_tensor_count,
        "absolute_minimum_node_count": solver["absolute_minimum_node_count"],
        "recommended_operational_node_count": solver["recommended_operational_node_count"],
    }


__all__ = [
    "DEFAULT_CANDIDATES",
    "MemoryPolicy",
    "PlacementError",
    "build_units",
    "solve_node_counts",
    "write_placement_artifacts",
]
