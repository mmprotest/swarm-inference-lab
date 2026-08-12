"""Reference mathematics, replay, traffic and precision study for Experiment 017."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.execution.kda_verification import (
    factorized_kda_block,
    kda_state_traffic_model,
    replay_kda_state,
    serial_kda_block,
    token_factors,
)
from swarm_inference.experiments.experiment_017 import BLOCK_SIZES

SEED = 17017
KEY_DIMENSION = 128
VALUE_DIMENSION = 128
HEADS = 96
EXACT_RELATIVE_L2_GATE = 2e-5
DEFAULT_DTYPE = np.dtype(np.float32)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _relative_l2(reference: np.ndarray, actual: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference.astype(np.float64)))
    numerator = float(np.linalg.norm(actual.astype(np.float64) - reference.astype(np.float64)))
    return numerator / denominator if denominator else numerator


def _metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    difference = actual.astype(np.float64) - reference.astype(np.float64)
    return {
        "maximum_absolute_error": float(np.max(np.abs(difference), initial=0.0)),
        "relative_l2_error": _relative_l2(reference, actual),
        "reference_l2": float(np.linalg.norm(reference.astype(np.float64))),
        "actual_l2": float(np.linalg.norm(actual.astype(np.float64))),
        "nan_count": int(np.isnan(actual).sum()),
        "inf_count": int(np.isinf(actual).sum()),
    }


def _fixture(rows: int, *, dtype: np.dtype[Any] = DEFAULT_DTYPE) -> tuple[Any, ...]:
    rng = np.random.default_rng(SEED + rows)
    state = rng.normal(0.0, 0.04, size=(KEY_DIMENSION, VALUE_DIMENSION)).astype(dtype)
    alphas = np.exp(-rng.uniform(0.005, 0.15, size=(rows, KEY_DIMENSION))).astype(dtype)
    keys = rng.normal(size=(rows, KEY_DIMENSION)).astype(dtype)
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    values = rng.normal(0.0, 0.3, size=(rows, VALUE_DIMENSION)).astype(dtype)
    betas = (1.0 / (1.0 + np.exp(-rng.normal(size=rows)))).astype(dtype)
    queries = rng.normal(size=(rows, KEY_DIMENSION)).astype(dtype)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    return state, token_factors(alphas, keys, values, betas), queries


def _timing(action: Any, *, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        action()
    values: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        action()
        values.append((time.perf_counter_ns() - started) / 1e6)
    ordered = sorted(values)
    return {
        "p50_ms": float(statistics.median(ordered)),
        "minimum_ms": float(ordered[0]),
        "maximum_ms": float(ordered[-1]),
        "mean_ms": float(statistics.fmean(ordered)),
        "retained_iterations": iterations,
        "warmup_iterations": warmup,
    }


def _round_to_bfloat16(value: np.ndarray) -> np.ndarray:
    """Round float32 to BF16 and return the exactly representable float32 value."""

    source = np.ascontiguousarray(value, dtype=np.float32)
    bits = source.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def _bf16_state_fp32_update(
    state: np.ndarray,
    factors: tuple[Any, ...],
    queries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    state = _round_to_bfloat16(state)
    outputs: list[np.ndarray] = []
    for factor, query in zip(factors, queries, strict=True):
        decayed = factor.alpha.astype(np.float32)[:, None] * state
        k_state = factor.key.astype(np.float32) @ decayed
        updated = (
            decayed
            + factor.key.astype(np.float32)[:, None]
            * ((factor.value.astype(np.float32) - k_state) * np.float32(factor.beta))[None, :]
        )
        outputs.append(query.astype(np.float32) @ updated)
        state = _round_to_bfloat16(updated)
    return np.stack(outputs), state


def _operation_model(block_tokens: int) -> dict[str, int | float]:
    k = KEY_DIMENSION
    v = VALUE_DIMENSION
    n = block_tokens
    # Counts are transparent scalar-operation estimates, not hardware counters.
    serial_flops_per_head = n * (8 * k * v + 12 * k + 4 * v)
    factor_output_flops_per_head = n * (2 * k * v) + n * (n + 1) * (k + v)
    accepted_reconstruction_flops_per_head = k * v + 6 * n * k * v + 2 * n * v
    factor_total = factor_output_flops_per_head + accepted_reconstruction_flops_per_head
    return {
        "serial_estimated_flops_all_heads": serial_flops_per_head * HEADS,
        "factor_output_estimated_flops_all_heads": factor_output_flops_per_head * HEADS,
        "accepted_reconstruction_estimated_flops_all_heads": (
            accepted_reconstruction_flops_per_head * HEADS
        ),
        "factor_total_estimated_flops_all_heads": factor_total * HEADS,
        "factor_flop_ratio_vs_serial": factor_total / serial_flops_per_head,
        "note": "analytical scalar-operation estimate; not a GPU counter",
    }


def benchmark(output_path: Path, *, warmup: int = 2, iterations: int = 7) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    exact_pass = True
    for block in BLOCK_SIZES:
        state, factors, queries = _fixture(block)
        serial_output, serial_state, serial_states = serial_kda_block(
            state, factors, queries, retain_states=True
        )
        factor_output, factor_state, prefix = factorized_kda_block(state, factors, queries)
        replay_state = replay_kda_state(state, factors)
        output_metrics = _metrics(serial_output, factor_output)
        state_metrics = _metrics(serial_state, factor_state)
        replay_metrics = _metrics(serial_state, replay_state)
        passed = bool(
            output_metrics["relative_l2_error"] <= EXACT_RELATIVE_L2_GATE
            and state_metrics["relative_l2_error"] <= EXACT_RELATIVE_L2_GATE
            and replay_metrics["relative_l2_error"] <= EXACT_RELATIVE_L2_GATE
            and output_metrics["nan_count"] == 0
            and state_metrics["nan_count"] == 0
        )
        exact_pass &= passed
        serial_timing = _timing(
            lambda state=state, factors=factors, queries=queries: serial_kda_block(
                state, factors, queries
            ),
            warmup=warmup,
            iterations=iterations,
        )
        factor_timing = _timing(
            lambda state=state, factors=factors, queries=queries: factorized_kda_block(
                state, factors, queries
            ),
            warmup=warmup,
            iterations=iterations,
        )
        replay_timing = _timing(
            lambda state=state, factors=factors: replay_kda_state(state, factors),
            warmup=warmup,
            iterations=iterations,
        )
        traffic = kda_state_traffic_model(block_tokens=block)
        rows.append(
            {
                "block_tokens": block,
                "status": "PASS" if passed else "FAIL",
                "output_metrics": output_metrics,
                "state_metrics": state_metrics,
                "replay_metrics": replay_metrics,
                "transition_rank": prefix.transition_rank,
                "additive_rank": prefix.additive_rank,
                "serial_retained_full_state_bytes": sum(item.nbytes for item in serial_states),
                "serial_cpu_timing": serial_timing,
                "factorized_cpu_timing": factor_timing,
                "replay_cpu_timing": replay_timing,
                "cpu_factorized_speedup": (serial_timing["p50_ms"] / factor_timing["p50_ms"]),
                "traffic": traffic,
                "operation_model": _operation_model(block),
            }
        )

    state, factors, queries = _fixture(16)
    exact_output, exact_state, _ = serial_kda_block(state, factors, queries)
    bf16_output, bf16_state = _bf16_state_fp32_update(state, factors, queries)
    precision = {
        "mode": "bf16-state-fp32-update",
        "classification": "APPROXIMATE",
        "exact_control": "float32-state-float32-update",
        "output_metrics": _metrics(exact_output, bf16_output),
        "active_state_metrics": _metrics(exact_state, bf16_state),
        "final_hidden_metrics": _metrics(exact_output[-1], bf16_output[-1]),
        "route_agreement": None,
        "route_agreement_reason": "reference KDA recurrence has no router",
        "logit_divergence": None,
        "top1_token_agreement": None,
        "gpu_latency_ms": None,
        "oracle_tok_s_per_user": None,
        "qualification_status": "LAYER_REFERENCE_ONLY",
        "full_93_layer_qualification_run": False,
        "full_93_layer_qualification_reason": (
            "no GPU implementation and no >=1.5x whole-oracle result"
        ),
    }

    payload = {
        "schema_version": "experiment-017-kda-reference-v1",
        "status": "PASS" if exact_pass else "FAIL",
        "seed": SEED,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "configuration": {
            "block_sizes": list(BLOCK_SIZES),
            "heads": HEADS,
            "key_dimension": KEY_DIMENSION,
            "value_dimension": VALUE_DIMENSION,
            "dtype": "float32",
            "warmup": warmup,
            "iterations": iterations,
            "exact_relative_l2_gate": EXACT_RELATIVE_L2_GATE,
        },
        "derivation": {
            "serial_update": ("H_t = D_t H_(t-1) + beta_t k_t (v_t - k_t^T D_t H_(t-1))^T"),
            "affine_transition": ("A_t = D_t - beta_t k_t (D_t k_t)^T; C_t = beta_t k_t v_t^T"),
            "composition": ("(A2,C2) o (A1,C1) = (A2 A1, A2 C1 + C2)"),
            "factor_rank_growth": "one transition and additive rank per token",
            "candidate_output": ("q_i^T(A_1:i H0 + C_1:i), evaluated without writing H_i"),
            "accepted_state": "materialize Apply(H0, prefix_1:k) once",
        },
        "rows": rows,
        "precision_reference": precision,
        "cuda_prototype_gate": {
            "decision": "SKIP_FACTOR_CUDA_PROTOTYPE",
            "reason": (
                "the existing measured KDA recurrent core is only 0.0431312 ms/token; "
                "at block 7 its complete elimination is a 2.83% real-layer ceiling, "
                "while exact factors add rank-growing arithmetic and accepted-state reconstruction"
            ),
            "measured_core_ms_per_token": 0.04313119888305664,
            "measured_real_layer89_block7_device_ms": 12.201568126678467,
            "free_core_layer_speedup_ceiling": (
                12.201568126678467 / (12.201568126678467 - 8 * 0.04313119888305664)
            ),
        },
    }
    _atomic_json(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=7)
    arguments = parser.parse_args()
    payload = benchmark(arguments.output, warmup=arguments.warmup, iterations=arguments.iterations)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "block7": next(row for row in payload["rows"] if row["block_tokens"] == 7),
                "precision_reference": payload["precision_reference"],
            },
            indent=2,
        )
    )
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
