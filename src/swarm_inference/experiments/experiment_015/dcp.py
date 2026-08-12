"""Exact log-sum-exp combination for context-sharded MLA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.experiments.experiment_015.evidence import atomic_json


@dataclass(frozen=True, slots=True)
class AttentionPartial:
    """Sufficient statistics from one exact attention context shard."""

    maximum: np.ndarray
    denominator: np.ndarray
    numerator: np.ndarray
    context_tokens: int


def shard_attention(scores: np.ndarray, values: np.ndarray) -> AttentionPartial:
    """Compute stable softmax sufficient statistics for one context shard.

    Scores have shape ``[..., context]`` and values ``[..., context, value]``.
    The leading dimensions must agree. Accumulation is FP64 for a strict local
    reference; production may retain FP32 after its numerical gate passes.
    """
    score_values = np.asarray(scores, dtype=np.float64)
    value_values = np.asarray(values, dtype=np.float64)
    if score_values.ndim < 1 or value_values.ndim != score_values.ndim + 1:
        raise ValueError("attention shard ranks are incompatible")
    if value_values.shape[:-2] != score_values.shape[:-1]:
        raise ValueError("attention shard leading dimensions differ")
    if value_values.shape[-2] != score_values.shape[-1] or score_values.shape[-1] < 1:
        raise ValueError("attention shard context dimensions differ or are empty")
    maximum = np.max(score_values, axis=-1)
    exponentials = np.exp(score_values - maximum[..., None])
    denominator = np.sum(exponentials, axis=-1)
    numerator = np.einsum("...t,...tv->...v", exponentials, value_values)
    return AttentionPartial(maximum, denominator, numerator, score_values.shape[-1])


def combine_attention_partials(partials: list[AttentionPartial]) -> np.ndarray:
    """Combine arbitrary context shards into the exact full softmax output."""
    if not partials:
        raise ValueError("at least one attention shard is required")
    shape = partials[0].maximum.shape
    value_shape = partials[0].numerator.shape
    if any(
        item.maximum.shape != shape
        or item.denominator.shape != shape
        or item.numerator.shape != value_shape
        or item.context_tokens < 1
        for item in partials
    ):
        raise ValueError("attention shard sufficient statistics are incompatible")
    global_maximum = np.maximum.reduce([item.maximum for item in partials])
    denominator = np.zeros(shape, dtype=np.float64)
    numerator = np.zeros(value_shape, dtype=np.float64)
    for item in partials:
        scale = np.exp(item.maximum - global_maximum)
        denominator += item.denominator * scale
        numerator += item.numerator * scale[..., None]
    if np.any(denominator <= 0) or not np.isfinite(denominator).all():
        raise ValueError("combined attention denominator is invalid")
    return numerator / denominator[..., None]


def full_attention(scores: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Reference full-context attention through the same sufficient statistics."""
    return combine_attention_partials([shard_attention(scores, values)])


def dcp_partial_payload_bytes(
    *,
    query_rows: int,
    heads: int = 64,
    value_dimension: int = 128,
    fp32_accumulation: bool = True,
) -> int:
    """Bytes sent by one worker: numerator plus max and denominator per head."""
    if query_rows <= 0 or heads <= 0 or value_dimension <= 0:
        raise ValueError("DCP payload geometry must be positive")
    scalar_bytes = 4 if fp32_accumulation else 2
    return query_rows * heads * (value_dimension + 2) * scalar_bytes


def benchmark_dcp_combination(output_path: Path) -> dict[str, Any]:
    """Execute deterministic component-level DCP correctness on local CPU.

    This verifies the exact sufficient-statistic reduction, including uneven
    shards and repeatability.  It does not claim a complete Kimi MLA execution.
    """
    generator = np.random.default_rng(15006)
    scores = generator.normal(size=(3, 8, 1024)).astype(np.float32)
    values = generator.normal(size=(3, 8, 1024, 32)).astype(np.float32)
    reference = full_attention(scores, values)
    rows: list[dict[str, Any]] = []
    for degree in (1, 2, 4, 8):
        score_shards = np.array_split(scores, degree, axis=-1)
        value_shards = np.array_split(values, degree, axis=-2)
        partials = [
            shard_attention(score_shard, value_shard)
            for score_shard, value_shard in zip(
                score_shards, value_shards, strict=True
            )
        ]
        actual = combine_attention_partials(partials)
        repeated = combine_attention_partials(partials)
        difference = actual - reference
        rows.append(
            {
                "degree": degree,
                "maximum_absolute_error": float(np.max(np.abs(difference))),
                "relative_l2_error": float(
                    np.linalg.norm(difference) / np.linalg.norm(reference)
                ),
                "finite": bool(np.isfinite(actual).all()),
                "repeat_bit_exact": bool(np.array_equal(actual, repeated)),
                "context_tokens": 1024,
                "query_rows": 3,
                "heads": 8,
                "value_dimension": 32,
            }
        )
    maximum = max(float(row["relative_l2_error"]) for row in rows)
    receipt: dict[str, Any] = {
        "schema_version": "experiment-015-dcp-combination-correctness-v1",
        "cycle_id": "H015-006A",
        "status": "PASS" if maximum <= 1e-12 else "FAIL",
        "evidence_class": None,
        "scientific_result": False,
        "scope": (
            "local CPU exact-attention unit diagnostic with synthetic tensors; "
            "outside the four Experiment 015 evidence classes and not complete Kimi MLA"
        ),
        "rows": rows,
        "maximum_relative_l2_error": maximum,
        "complete_kimi_reference_gate": "NOT_RUN",
    }
    atomic_json(output_path, receipt)
    return receipt


__all__ = [
    "AttentionPartial",
    "benchmark_dcp_combination",
    "combine_attention_partials",
    "dcp_partial_payload_bytes",
    "full_attention",
    "shard_attention",
]
