"""Frozen closed-loop decode workload helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .freeze import DECODE_CONCURRENCY_LEVELS


def microbatch_sizes(active_sequence_count: int) -> tuple[int, ...]:
    if active_sequence_count not in DECODE_CONCURRENCY_LEVELS:
        raise ValueError("concurrency is outside the frozen E024 ladder")
    full, remainder = divmod(active_sequence_count, 4)
    values = [4] * full
    if remainder:
        values.append(remainder)
    return tuple(values)


def measured_steps_per_batch(active_sequence_count: int) -> int:
    if active_sequence_count not in DECODE_CONCURRENCY_LEVELS:
        raise ValueError("concurrency is outside the frozen E024 ladder")
    return max(4, math.ceil(256 / active_sequence_count))


def measured_output_token_count(active_sequence_count: int) -> int:
    return active_sequence_count * measured_steps_per_batch(active_sequence_count)


def token_latency_quantiles(
    step_latencies_ms: tuple[tuple[float, int], ...],
) -> tuple[float, float]:
    values: list[float] = []
    for latency, batch_size in step_latencies_ms:
        if not math.isfinite(latency) or latency < 0:
            raise ValueError("decode-step latency must be finite and non-negative")
        if batch_size not in (1, 2, 4):
            raise ValueError("invalid E024 microbatch size")
        values.extend([latency] * batch_size)
    if not values:
        raise ValueError("at least one measured decode-step latency is required")
    array = np.asarray(values, dtype=np.float64)
    return (
        float(np.quantile(array, 0.50, method="linear")),
        float(np.quantile(array, 0.95, method="linear")),
    )


@dataclass(frozen=True, slots=True)
class DecodeMeasurementPlan:
    active_sequence_count: int
    microbatch_sizes: tuple[int, ...]
    warmup_steps_per_batch: int
    measured_steps_per_batch: int
    minimum_measured_output_tokens: int


class DecodeServingEngine:
    """Define the closed-loop run without target-pass terminology."""

    def measurement_plan(self, active_sequence_count: int) -> DecodeMeasurementPlan:
        sizes = microbatch_sizes(active_sequence_count)
        steps = measured_steps_per_batch(active_sequence_count)
        measured = steps * sum(sizes)
        if measured < 256:
            raise RuntimeError("INSUFFICIENT_TOKEN_SAMPLES")
        return DecodeMeasurementPlan(
            active_sequence_count=active_sequence_count,
            microbatch_sizes=sizes,
            warmup_steps_per_batch=2,
            measured_steps_per_batch=steps,
            minimum_measured_output_tokens=measured,
        )


__all__ = [
    "DecodeMeasurementPlan",
    "DecodeServingEngine",
    "measured_output_token_count",
    "measured_steps_per_batch",
    "microbatch_sizes",
    "token_latency_quantiles",
]
