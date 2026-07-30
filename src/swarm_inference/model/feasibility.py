"""Configuration/index-only feasibility analysis for very large checkpoints."""

from __future__ import annotations

import json
import math
import re
from typing import Any

from swarm_inference.exceptions import IntegrityError
from swarm_inference.model.qwen3 import _DTYPE_BYTES, _normalise_dtype, _weight_map
from swarm_inference.model.shard_builder import ResolvedModel

_ANY_LAYER = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)\.")


def analyse_large_model_manifest(
    resolved: ResolvedModel,
    *,
    node_memory_bytes: int = 32 * 1024**3,
    safety_fraction: float = 0.8,
    measured_proxy_tokens_s_per_stage: float | None = None,
) -> dict[str, Any]:
    """Read config and safetensors headers only; never materialise tensor data."""

    if not 0 < safety_fraction <= 1:
        raise ValueError("safety_fraction must be in (0, 1]")
    config_path = resolved.path / "config.json"
    if not config_path.is_file():
        raise IntegrityError(f"model config is missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    mapping = _weight_map(resolved.path)
    by_file: dict[str, list[str]] = {}
    for name, file in mapping.items():
        by_file.setdefault(file, []).append(name)
    from safetensors import safe_open

    total_bytes = 0
    layer_bytes: dict[int, int] = {}
    expert_bytes = 0
    non_expert_bytes = 0
    dtypes: dict[str, int] = {}
    largest_tensor_name = ""
    largest_tensor_bytes = 0
    for file, names in sorted(by_file.items()):
        with safe_open(resolved.path / file, framework="pt", device="cpu") as handle:
            for name in names:
                slice_ = handle.get_slice(name)
                dtype = _normalise_dtype(slice_.get_dtype())
                width = _DTYPE_BYTES.get(dtype)
                if width is None:
                    raise IntegrityError(f"unsupported dtype {dtype} in tensor {name}")
                size = math.prod(int(value) for value in slice_.get_shape()) * width
                total_bytes += size
                dtypes[dtype] = dtypes.get(dtype, 0) + size
                match = _ANY_LAYER.search(name)
                if match:
                    index = int(match.group(1))
                    layer_bytes[index] = layer_bytes.get(index, 0) + size
                is_expert = any(
                    marker in name
                    for marker in (".experts.", ".expert.", "expert_weights", "experts_")
                )
                if is_expert:
                    expert_bytes += size
                else:
                    non_expert_bytes += size
                if size > largest_tensor_bytes:
                    largest_tensor_name = name
                    largest_tensor_bytes = size
    largest_layer = max(layer_bytes.values(), default=0)
    usable = int(node_memory_bytes * safety_fraction)
    minimum_nodes = math.ceil(total_bytes / node_memory_bytes)
    practical_nodes = math.ceil(total_bytes / usable)
    hidden = int(
        config.get("hidden_size")
        or config.get("n_embd")
        or config.get("text_config", {}).get("hidden_size", 0)
    )
    dominant_dtype = max(dtypes, key=lambda name: dtypes[name]) if dtypes else "unknown"
    activation_width = _DTYPE_BYTES.get(dominant_dtype, 2)
    activation_bytes = hidden * activation_width if hidden else None
    layers = int(
        config.get("num_hidden_layers")
        or config.get("n_layer")
        or config.get("text_config", {}).get("num_hidden_layers", len(layer_bytes))
    )
    kv_heads = int(
        config.get("num_key_value_heads")
        or config.get("num_attention_heads")
        or config.get("text_config", {}).get("num_key_value_heads", 0)
    )
    head_dim = int(
        config.get("head_dim")
        or (
            hidden
            // int(
                config.get("num_attention_heads")
                or config.get("text_config", {}).get("num_attention_heads", 1)
            )
            if hidden
            else 0
        )
    )
    cache_bytes_per_token = (
        2 * kv_heads * head_dim * activation_width * layers
        if kv_heads and head_dim and layers
        else None
    )
    model_type = str(config.get("model_type", "unknown"))
    unsupported: list[str] = []
    if model_type not in {"qwen3", "qwen3_moe"}:
        unsupported.append(
            f"no executable adapter for model_type={model_type}; analysis is index-only"
        )
    if any(
        key in config
        for key in (
            "linear_num_key_heads",
            "mamba_d_state",
            "recurrent_state_size",
        )
    ):
        unsupported.append("recurrent/linear-attention state semantics require backend work")
    if config.get("quantization_config"):
        unsupported.append(
            "quantised stage kernels must be benchmarked for this quantisation format"
        )
    projected = (
        measured_proxy_tokens_s_per_stage if measured_proxy_tokens_s_per_stage is not None else None
    )
    return {
        "analysis_kind": "configuration-and-safetensors-index-only",
        "model_id": resolved.model_id,
        "model_revision": resolved.revision,
        "model_type": model_type,
        "total_model_bytes": total_bytes,
        "largest_tensor": {
            "name": largest_tensor_name,
            "bytes": largest_tensor_bytes,
        },
        "largest_layer_bytes": largest_layer,
        "layer_distribution_bytes": {
            str(index): size for index, size in sorted(layer_bytes.items())
        },
        "expert_bytes": expert_bytes,
        "non_expert_bytes": non_expert_bytes,
        "node_memory_budget_bytes": node_memory_bytes,
        "safety_fraction": safety_fraction,
        "minimum_32gb_node_count": minimum_nodes,
        "practical_node_count_with_safety_margin": practical_nodes,
        "estimated_stage_activation_bytes_per_token": activation_bytes,
        "estimated_cache_bytes_per_token_all_layers": cache_bytes_per_token,
        "estimated_aggregate_throughput_from_measured_proxy": projected,
        "throughput_projection_note": (
            "No projection supplied because measured proxy-model worker results were not provided."
            if projected is None
            else "Projection is a proxy estimate, not measured target-model performance."
        ),
        "unsupported_architectural_components": unsupported,
        "required_backend_work": [
            "validate attention and recurrent-state semantics",
            "validate quantised kernels on each physical backend",
            "measure per-stage service rates",
            "measure activation transport and cache recovery",
        ],
        "claim_of_model_support": False,
    }
