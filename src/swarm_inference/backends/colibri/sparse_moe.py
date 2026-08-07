"""Swarm-owned generic sparse-MoE components for the pinned Colibri path.

This module implements the architecture-independent part of routed expert
execution: deterministic top-k selection, bounded expert residency, exact
weighted accumulation, and standard adapter-described SwiGLU projections.
Attention and router metadata remain architecture-adapter responsibilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from swarm_inference.backends.colibri.architecture import ColibriRoutingDescriptor
from swarm_inference.execution.expert import (
    RoutedComputationKernel,
    RoutedComputationStore,
    RoutedComputationWeights,
    silu,
)


@dataclass(frozen=True, slots=True)
class RoutedSelection:
    """Deterministic expert IDs and exact selected routing weights."""

    expert_ids: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        expert_ids = np.asarray(self.expert_ids)
        weights = np.asarray(self.weights)
        if expert_ids.ndim != 2 or weights.shape != expert_ids.shape:
            raise ValueError("routed selections require matching [rows, top_k] arrays")
        if not np.issubdtype(expert_ids.dtype, np.integer):
            raise ValueError("routed expert IDs must be integers")
        if np.any(weights < 0) or not np.all(np.isfinite(weights)):
            raise ValueError("routed expert weights must be finite and non-negative")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    result = np.empty_like(source)
    positive = source >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-source[positive]))
    exponent = np.exp(source[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def _softmax(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    shifted = source - np.max(source, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return np.asarray(exponent / np.sum(exponent, axis=-1, keepdims=True), dtype=np.float32)


def select_routed_experts(
    router_logits: np.ndarray,
    descriptor: ColibriRoutingDescriptor,
    *,
    correction_bias: np.ndarray | None = None,
    routing_metadata: dict[str, Any] | None = None,
) -> RoutedSelection:
    """Select experts without embedding any model-family or repository-name rule.

    The adapter describes score function, grouping, normalization, and scaling.
    Ties are resolved by ascending expert ID, making replay deterministic across
    workers and NumPy implementations.
    """

    logits = np.asarray(router_logits, dtype=np.float32)
    if logits.ndim != 2 or logits.shape[1] != descriptor.expert_count:
        raise ValueError("router logits must be [rows, adapter expert_count]")
    metadata = routing_metadata or {}
    routing_kind = descriptor.routing_kind.casefold()
    scores = _sigmoid(logits) if "sigmoid" in routing_kind else _softmax(logits)
    selection_scores = scores.copy()
    if correction_bias is not None:
        bias = np.asarray(correction_bias, dtype=np.float32)
        if bias.shape != (descriptor.expert_count,):
            raise ValueError("router correction bias must have one value per expert")
        selection_scores += bias[None, :]

    candidate_mask = np.ones_like(selection_scores, dtype=np.bool_)
    grouped = "group" in routing_kind or int(metadata.get("n_group", 1)) > 1
    if grouped:
        group_count = int(metadata.get("n_group", 0))
        selected_group_count = int(metadata.get("topk_group", 0))
        if (
            group_count <= 0
            or selected_group_count <= 0
            or selected_group_count > group_count
            or descriptor.expert_count % group_count
        ):
            raise ValueError("grouped routing requires valid n_group and topk_group metadata")
        width = descriptor.expert_count // group_count
        reshaped = selection_scores.reshape(logits.shape[0], group_count, width)
        # DeepSeek-style no-aux routing ranks a group by its strongest two
        # corrected expert scores.  Adapters may request max or sum explicitly.
        group_method = str(metadata.get("group_score_method", "top2_sum")).casefold()
        if group_method == "max":
            group_scores = np.max(reshaped, axis=-1)
        elif group_method == "top2_sum":
            take = min(2, width)
            group_scores = np.sum(
                np.partition(reshaped, width - take, axis=-1)[..., -take:], axis=-1
            )
        else:
            raise ValueError(f"unsupported adapter group score method {group_method!r}")
        candidate_mask.fill(False)
        group_ids = np.arange(group_count)
        for row in range(logits.shape[0]):
            order = np.lexsort((group_ids, -group_scores[row]))
            for group in order[:selected_group_count]:
                start = int(group) * width
                candidate_mask[row, start : start + width] = True

    selected_ids = np.empty((logits.shape[0], descriptor.experts_per_token), dtype=np.int64)
    selected_weights = np.empty_like(selected_ids, dtype=np.float32)
    expert_ids = np.arange(descriptor.expert_count)
    for row in range(logits.shape[0]):
        candidates = expert_ids[candidate_mask[row]]
        if candidates.size < descriptor.experts_per_token:
            raise ValueError("adapter routing mask contains fewer candidates than top-k")
        order = np.lexsort((candidates, -selection_scores[row, candidates]))
        chosen = candidates[order[: descriptor.experts_per_token]]
        selected_ids[row] = chosen
        selected_weights[row] = scores[row, chosen]

    normalization = descriptor.normalization.casefold()
    normalize = bool(metadata.get("norm_topk_prob", True))
    if normalization in {"none", "unnormalized"}:
        normalize = False
    if normalize:
        denominator = np.sum(selected_weights, axis=-1, keepdims=True)
        if np.any(denominator <= 0):
            raise ValueError("selected router weights have a zero normalization denominator")
        selected_weights /= denominator
    selected_weights *= np.float32(metadata.get("routed_scaling_factor", 1.0))
    return RoutedSelection(selected_ids, selected_weights)


def _float8_e4m3fn(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.uint8)
    sign = np.where(bits & 0x80, -1.0, 1.0).astype(np.float32)
    exponent = ((bits >> 3) & 0x0F).astype(np.int16)
    mantissa = (bits & 0x07).astype(np.float32)
    normal = sign * np.exp2(exponent.astype(np.float32) - 7.0) * (1.0 + mantissa / 8.0)
    subnormal = sign * np.exp2(np.float32(-6.0)) * (mantissa / 8.0)
    result = np.where(exponent == 0, subnormal, normal).astype(np.float32)
    result[(exponent == 15) & (mantissa == 7)] = np.nan
    return np.asarray(result, dtype=np.float32)


def _as_float32(tensor: np.ndarray, *, native_format: str | None = None) -> np.ndarray:
    source = np.asarray(tensor)
    if source.dtype == np.uint16:
        # Safetensors BF16 values are retained losslessly as their native bits by
        # the generic loader, then widened only at the reference-kernel boundary.
        return np.asarray(
            (source.astype(np.uint32) << np.uint32(16)).view(np.float32),
            dtype=np.float32,
        )
    if (
        source.dtype == np.uint8
        and native_format is not None
        and native_format.casefold().replace("_", "").startswith("float8e4m3")
    ):
        return _float8_e4m3fn(source)
    if np.issubdtype(source.dtype, np.floating):
        return source.astype(np.float32, copy=False)
    raise TypeError(
        f"standard SwiGLU reference kernel cannot decode native {source.dtype}; "
        "a quantization-specific Colibri kernel is required"
    )


def _linear(activation: np.ndarray, weight: np.ndarray) -> np.ndarray:
    source = np.asarray(activation, dtype=np.float32)
    matrix = np.asarray(weight, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("standard expert projections must be rank two after adapter slicing")
    if matrix.shape[1] == source.shape[-1]:
        return source @ matrix.T
    if matrix.shape[0] == source.shape[-1]:
        return source @ matrix
    raise ValueError("expert projection dimensions do not consume the activation width")


def _apply_block_scales(
    weight: np.ndarray,
    scales: np.ndarray,
    *,
    routing_metadata: dict[str, Any],
) -> np.ndarray:
    matrix = np.asarray(weight, dtype=np.float32)
    scale_matrix = _as_float32(scales)
    if matrix.ndim != 2 or scale_matrix.ndim != 2:
        raise ValueError("FP8 block weights and scales must both be rank two")
    configured = routing_metadata.get("weight_block_size", (128, 128))
    if not isinstance(configured, (tuple, list)) or len(configured) != 2:
        raise ValueError("adapter weight_block_size must contain two dimensions")
    row_block, column_block = (int(configured[0]), int(configured[1]))
    if row_block <= 0 or column_block <= 0:
        raise ValueError("adapter weight block dimensions must be positive")
    expected = (
        (matrix.shape[0] + row_block - 1) // row_block,
        (matrix.shape[1] + column_block - 1) // column_block,
    )
    if scale_matrix.shape != expected:
        raise ValueError(
            f"FP8 scale shape {scale_matrix.shape} differs from adapter block grid {expected}"
        )
    expanded = np.repeat(np.repeat(scale_matrix, row_block, axis=0), column_block, axis=1)
    return np.ascontiguousarray(matrix * expanded[: matrix.shape[0], : matrix.shape[1]])


def _decode_symmetric_int4_groups(
    packed: np.ndarray,
    scales: np.ndarray,
    *,
    group_size: int,
) -> np.ndarray:
    """Decode compressed-tensors symmetric INT4 group weights exactly."""

    source = np.asarray(packed)
    if source.dtype not in {np.dtype(np.int32), np.dtype(np.uint32)} or source.ndim != 2:
        raise TypeError("packed INT4 expert weights must be rank-two int32 tensors")
    if group_size <= 0 or group_size % 8:
        raise ValueError("packed INT4 group size must be a positive multiple of eight")
    unsigned = source.view(np.uint32)
    shifts = (np.arange(8, dtype=np.uint32) * np.uint32(4)).reshape(1, 1, 8)
    # compressed-tensors shifts signed values by 2^(bits-1) before packing.
    unpacked = ((unsigned[..., None] >> shifts) & np.uint32(0xF)).astype(np.int8) - 8
    matrix = unpacked.reshape(source.shape[0], source.shape[1] * 8).astype(np.float32)
    scale_matrix = _as_float32(scales)
    expected_groups = (matrix.shape[1] + group_size - 1) // group_size
    if scale_matrix.shape != (matrix.shape[0], expected_groups):
        raise ValueError("INT4 scale shape differs from the adapter-declared output/group grid")
    expanded = np.repeat(scale_matrix, group_size, axis=1)
    return np.ascontiguousarray(matrix * expanded[:, : matrix.shape[1]])


def standard_swiglu_kernel(
    activation: np.ndarray,
    weights: RoutedComputationWeights,
    *,
    routing_metadata: dict[str, Any],
) -> np.ndarray:
    """Execute separate or fused adapter-described SwiGLU expert tensors."""

    entries = tuple(
        (name, weights.tensor_roles[name].casefold(), tensor)
        for name, tensor in weights.tensors.items()
    )

    def projection(kind: str) -> np.ndarray | None:
        data = next(
            (
                (name, tensor)
                for name, role, tensor in entries
                if kind in role and "scale" not in role and "shape" not in role
            ),
            None,
        )
        if data is None:
            return None
        name, tensor = data
        native_format = (weights.tensor_formats or {}).get(name)
        scale = next(
            (value for _name, role, value in entries if kind in role and "scale" in role),
            None,
        )
        normalized_format = (native_format or "").casefold().replace("_", "-")
        if normalized_format.startswith("int4"):
            if scale is None:
                raise ValueError("packed INT4 expert projection has no group scales")
            decoded = _decode_symmetric_int4_groups(
                tensor,
                scale,
                group_size=int(routing_metadata.get("weight_group_size", 32)),
            )
            scale = None
        else:
            decoded = _as_float32(tensor, native_format=native_format)
        return (
            _apply_block_scales(decoded, scale, routing_metadata=routing_metadata)
            if scale is not None
            else decoded
        )

    fused = projection("gate+up")
    gate = projection("gate")
    up = projection("up")
    down = projection("down")
    if down is None:
        raise ValueError("adapter-described standard expert has no down projection")
    if fused is not None:
        projected = _linear(activation, fused)
        if projected.shape[-1] % 2:
            raise ValueError("fused gate/up projection width must be even")
        first, second = np.split(projected, 2, axis=-1)
        if str(routing_metadata.get("fused_projection_order", "gate_up")) == "up_gate":
            up_values, gate_values = first, second
        else:
            gate_values, up_values = first, second
    else:
        if gate is None or up is None:
            raise ValueError("adapter-described standard expert needs gate and up projections")
        gate_values = _linear(activation, gate)
        up_values = _linear(activation, up)
    if gate_values.shape != up_values.shape:
        raise ValueError("expert gate and up projection shapes differ")
    activation_kind = str(routing_metadata.get("activation", "silu")).casefold()
    if activation_kind not in {"silu", "swiglu"}:
        raise ValueError(f"standard Colibri expert kernel does not implement {activation_kind!r}")
    return np.ascontiguousarray(_linear(silu(gate_values) * up_values, down))


def execute_routed_layer(
    activation: np.ndarray,
    router_logits: np.ndarray,
    *,
    layer_id: int,
    routing: ColibriRoutingDescriptor,
    store: RoutedComputationStore,
    correction_bias: np.ndarray | None = None,
    routing_metadata: dict[str, Any] | None = None,
    shared_expert_ids: tuple[int, ...] = (),
    kernel: RoutedComputationKernel = standard_swiglu_kernel,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Execute one exact routed layer using generic residency and accumulation."""

    source = np.ascontiguousarray(activation, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError("routed-layer activation must be [rows, hidden]")
    selection = select_routed_experts(
        router_logits,
        routing,
        correction_bias=correction_bias,
        routing_metadata=routing_metadata,
    )
    before = store.status()
    output = np.zeros_like(source, dtype=np.float32)
    for row in range(source.shape[0]):
        for rank in range(routing.experts_per_token):
            expert_id = int(selection.expert_ids[row, rank])
            contribution = store.execute(
                layer_id=layer_id,
                expert_id=expert_id,
                expert_type="routed",
                activation=source[row : row + 1],
                kernel=kernel,
                routing_metadata=routing_metadata,
            )
            output[row] += np.float32(selection.weights[row, rank]) * contribution[0]
        for expert_id in shared_expert_ids:
            output[row] += store.execute(
                layer_id=layer_id,
                expert_id=expert_id,
                expert_type="shared",
                activation=source[row : row + 1],
                kernel=kernel,
                routing_metadata=routing_metadata,
            )[0]
    after = store.status()
    return output, {
        "selected_expert_ids": selection.expert_ids.tolist(),
        "selected_expert_weights": selection.weights.tolist(),
        "cache_hits": after["cache_hits"] - before["cache_hits"],
        "cache_misses": after["cache_misses"] - before["cache_misses"],
        "expert_movement_bytes": after["bytes_read"] - before["bytes_read"],
        "expert_cache_hit_rate": (
            (after["cache_hits"] - before["cache_hits"])
            / max(
                1,
                after["cache_hits"]
                - before["cache_hits"]
                + after["cache_misses"]
                - before["cache_misses"],
            )
        ),
    }


__all__ = [
    "RoutedSelection",
    "execute_routed_layer",
    "select_routed_experts",
    "standard_swiglu_kernel",
]
