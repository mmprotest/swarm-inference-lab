"""OLMoE checkpoint inspection for contiguous-stage planning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from safetensors import safe_open
from transformers import AutoConfig

from swarm_inference.model.partition import LayerCost, ModelPartitionMetadata, StageAssignment

_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def _tensor_nbytes(handle: Any, key: str) -> int:
    tensor_slice = handle.get_slice(key)
    shape = tensor_slice.get_shape()
    dtype = str(tensor_slice.get_dtype())
    if dtype not in _DTYPE_BYTES:
        raise ValueError(f"unsupported safetensors dtype {dtype!r}")
    count = 1
    for dimension in shape:
        count *= int(dimension)
    return count * _DTYPE_BYTES[dtype]


def inspect_olmoe_partition_metadata(
    model_path: Path,
    *,
    model_revision: str,
    tokenizer_revision: str,
    measured_layer_ns: dict[int, int] | None = None,
) -> ModelPartitionMetadata:
    """Inspect only checkpoint metadata needed by the generic planner."""

    model_path = model_path.resolve()
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"OLMoE safetensors index is missing: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map_value = index.get("weight_map")
    if not isinstance(weight_map_value, dict) or not weight_map_value:
        raise ValueError("OLMoE safetensors index has no weight_map")
    weight_map = {str(key): str(value) for key, value in weight_map_value.items()}
    by_shard: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(key)
    sizes: dict[str, int] = {}
    for shard, keys in by_shard.items():
        shard_path = model_path / shard
        if not shard_path.is_file():
            raise FileNotFoundError(f"OLMoE checkpoint shard is missing: {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for key in keys:
                sizes[key] = _tensor_nbytes(handle, key)
    layer_count = int(config.num_hidden_layers)
    weight_bytes = [0] * layer_count
    embedding_weight_bytes = 0
    final_weight_bytes = 0
    for key, size in sizes.items():
        if key.startswith("model.layers."):
            layer_id = int(key.split(".")[2])
            if not 0 <= layer_id < layer_count:
                raise ValueError(f"OLMoE tensor references out-of-range layer {layer_id}")
            weight_bytes[layer_id] += size
        elif key.startswith("model.embed_tokens."):
            embedding_weight_bytes += size
        elif key.startswith("model.norm.") or key.startswith("lm_head."):
            final_weight_bytes += size
    dtype_bytes = 2
    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    kv_heads = int(config.num_key_value_heads)
    kv_bytes_per_layer_token = 2 * kv_heads * head_dim * dtype_bytes
    activation_bytes = int(config.hidden_size) * dtype_bytes
    costs = []
    for layer_id in range(layer_count):
        measured = measured_layer_ns is not None and layer_id in measured_layer_ns
        execution_ns = (
            int(measured_layer_ns[layer_id])
            if measured and measured_layer_ns is not None
            else max(1, int(weight_bytes[layer_id] / 4_000_000_000 * 1e9))
        )
        costs.append(
            LayerCost(
                layer_id=layer_id,
                execution_ns=execution_ns,
                weight_bytes=weight_bytes[layer_id],
                kv_bytes_per_token=kv_bytes_per_layer_token,
                peak_temporary_bytes=max(activation_bytes * 16, weight_bytes[layer_id] // 32),
                activation_bytes=activation_bytes,
                measured=measured,
            )
        )
    identity_payload = json.dumps(
        {
            "model_revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
            "weight_map": weight_map,
            "layer_costs": [asdict(cost) for cost in costs],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return ModelPartitionMetadata(
        layer_costs=tuple(costs),
        embedding_weight_bytes=embedding_weight_bytes,
        final_weight_bytes=final_weight_bytes,
        dtype_bytes=dtype_bytes,
        hidden_size=int(config.hidden_size),
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        metadata_hash=hashlib.sha256(identity_payload).hexdigest(),
    )


def validate_olmoe_stage_assignment(
    model_path: Path,
    *,
    assignment: StageAssignment,
    stage_count: int,
    model_revision: str,
    tokenizer_revision: str,
) -> ModelPartitionMetadata:
    """Verify checkpoint-derived ownership costs without constructing a full model."""

    metadata = inspect_olmoe_partition_metadata(
        model_path,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
    )
    if stage_count > len(metadata.layer_costs):
        raise ValueError("OLMoE topology cannot contain more stages than decoder layers")
    if assignment.layer_end > len(metadata.layer_costs):
        raise ValueError("stage assignment extends beyond the OLMoE decoder layer count")
    selected = metadata.layer_costs[assignment.layer_start : assignment.layer_end]
    expected_weight_bytes = sum(cost.weight_bytes for cost in selected)
    if assignment.stage_id == 0:
        expected_weight_bytes += metadata.embedding_weight_bytes
    if assignment.stage_id == stage_count - 1:
        expected_weight_bytes += metadata.final_weight_bytes
    if assignment.weight_bytes != expected_weight_bytes:
        raise ValueError(
            "stage assignment weight bytes do not match exact OLMoE checkpoint ownership"
        )
    expected_kv_bytes = sum(cost.kv_bytes_per_token for cost in selected)
    if assignment.kv_cache_bytes_per_token != expected_kv_bytes:
        raise ValueError("stage assignment KV cost does not match the OLMoE checkpoint")
    if assignment.peak_temporary_bytes != max(cost.peak_temporary_bytes for cost in selected):
        raise ValueError("stage assignment temporary-memory cost does not match the checkpoint")
    if assignment.activation_bytes != max(cost.activation_bytes for cost in selected):
        raise ValueError("stage assignment activation size does not match the checkpoint")
    return metadata


__all__ = ["inspect_olmoe_partition_metadata", "validate_olmoe_stage_assignment"]
