"""Exact decode-context-parallel attention reduction primitives."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class AttentionPartial:
    """Stable softmax sufficient statistics from one non-empty context shard."""

    maximum: np.ndarray
    denominator: np.ndarray
    numerator: np.ndarray
    context_tokens: int
    shard_index: int = 0


def shard_attention(
    scores: np.ndarray,
    values: np.ndarray,
    *,
    shard_index: int = 0,
) -> AttentionPartial:
    """Compute one exact FP64 reference partial for a context shard."""

    score_values = np.asarray(scores, dtype=np.float64)
    value_values = np.asarray(values, dtype=np.float64)
    if score_values.ndim < 1 or value_values.ndim != score_values.ndim + 1:
        raise ValueError("attention shard ranks are incompatible")
    if value_values.shape[:-2] != score_values.shape[:-1]:
        raise ValueError("attention shard leading dimensions differ")
    if value_values.shape[-2] != score_values.shape[-1] or score_values.shape[-1] < 1:
        raise ValueError("attention shard context dimensions differ or are empty")
    if shard_index < 0:
        raise ValueError("attention shard index must be non-negative")
    maximum = np.max(score_values, axis=-1)
    exponentials = np.exp(score_values - maximum[..., None])
    return AttentionPartial(
        maximum=maximum,
        denominator=np.sum(exponentials, axis=-1),
        numerator=np.einsum("...t,...tv->...v", exponentials, value_values),
        context_tokens=int(score_values.shape[-1]),
        shard_index=shard_index,
    )


def combine_attention_partials(
    partials: Sequence[AttentionPartial],
) -> np.ndarray:
    """Combine context shards in deterministic shard-index order."""

    if not partials:
        raise ValueError("at least one attention shard is required")
    ordered = sorted(partials, key=lambda item: item.shard_index)
    if len({item.shard_index for item in ordered}) != len(ordered):
        raise ValueError("attention shard indices must be unique")
    maximum_shape = ordered[0].maximum.shape
    numerator_shape = ordered[0].numerator.shape
    if any(
        item.maximum.shape != maximum_shape
        or item.denominator.shape != maximum_shape
        or item.numerator.shape != numerator_shape
        or item.context_tokens < 1
        for item in ordered
    ):
        raise ValueError("attention shard sufficient statistics are incompatible")
    global_maximum = np.maximum.reduce([item.maximum for item in ordered])
    denominator = np.zeros(maximum_shape, dtype=np.float64)
    numerator = np.zeros(numerator_shape, dtype=np.float64)
    for item in ordered:
        scale = np.exp(item.maximum - global_maximum)
        denominator += item.denominator * scale
        numerator += item.numerator * scale[..., None]
    if np.any(denominator <= 0.0) or not np.isfinite(denominator).all():
        raise ValueError("combined attention denominator is invalid")
    output = numerator / denominator[..., None]
    if not np.isfinite(output).all():
        raise ValueError("combined attention output is non-finite")
    return output


def full_attention(scores: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Reference unsharded attention through the same stable reduction."""

    return combine_attention_partials([shard_attention(scores, values)])


def dcp_partial_payload_bytes(
    *,
    query_rows: int,
    heads: int = 64,
    value_dimension: int = 128,
    scalar_bytes: int = 4,
) -> int:
    """Wire bytes for numerator, maximum and denominator from one shard."""

    if query_rows <= 0 or heads <= 0 or value_dimension <= 0:
        raise ValueError("DCP payload geometry must be positive")
    if scalar_bytes not in (2, 4, 8):
        raise ValueError("DCP scalar size must be 2, 4 or 8 bytes")
    return query_rows * heads * (value_dimension + 2) * scalar_bytes


__all__ = [
    "AttentionPartial",
    "combine_attention_partials",
    "dcp_partial_payload_bytes",
    "full_attention",
    "shard_attention",
]
