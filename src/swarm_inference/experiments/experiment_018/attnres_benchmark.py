"""Exact AttnRes future-score cache proof and measured physical upper bound."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.execution.kimi_cuda_runtime import (
    _array_fingerprint,
    _CudaRuntime,
    _numerical_metrics,
    _sha256_file,
)
from swarm_inference.execution.kimi_k3_graph_runtime import _CheckpointReader

SCHEMA_VERSION = "experiment-018-attnres-future-score-v1"
HIDDEN = 7168
EPSILON = 1e-6


def _timing(values: list[float]) -> dict[str, float]:
    ordered = np.asarray(values, dtype=np.float64)
    return {
        "minimum_ms": float(np.min(ordered)),
        "maximum_ms": float(np.max(ordered)),
        "mean_ms": statistics.fmean(values),
        "p50_ms": float(np.percentile(ordered, 50)),
        "p95_ms": float(np.percentile(ordered, 95)),
        "p99_ms": float(np.percentile(ordered, 99)),
        "standard_deviation_ms": float(np.std(ordered)),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _score(value: np.ndarray, query: np.ndarray, epsilon: float) -> np.float32:
    source = np.ascontiguousarray(value, dtype=np.float32)
    weight = np.ascontiguousarray(query, dtype=np.float32)
    square = np.float32(0.0)
    dot = np.float32(0.0)
    for index in range(source.size):
        square = np.float32(square + np.float32(source[index] * source[index]))
        dot = np.float32(dot + np.float32(source[index] * weight[index]))
    denominator = np.sqrt(
        np.float32(square / np.float32(source.size) + np.float32(epsilon))
    )
    return np.float32(dot / denominator)


def _softmax(scores: list[np.float32]) -> np.ndarray:
    maximum = max(scores)
    exponentials = np.empty(len(scores), dtype=np.float32)
    total = np.float32(0.0)
    for index, value in enumerate(scores):
        exponentials[index] = np.exp(np.float32(value - maximum), dtype=np.float32)
        total = np.float32(total + exponentials[index])
    return np.ascontiguousarray(exponentials / total, dtype=np.float32)


def reference_mix(
    prefix: np.ndarray,
    completed: np.ndarray,
    query: np.ndarray,
    *,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    scores = [_score(value, query, epsilon) for value in completed]
    scores.append(_score(prefix, query, epsilon))
    probabilities = _softmax(scores)
    output = np.zeros_like(prefix, dtype=np.float32)
    for index, value in enumerate(completed):
        output = np.asarray(
            output + np.float32(probabilities[index]) * value,
            dtype=np.float32,
        )
    output = np.asarray(
        output + np.float32(probabilities[-1]) * prefix,
        dtype=np.float32,
    )
    return np.ascontiguousarray(output), np.asarray(scores, dtype=np.float32)


def cached_mix(
    prefix: np.ndarray,
    completed: np.ndarray,
    query: np.ndarray,
    cached_completed_scores: np.ndarray,
    *,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    scores = [np.float32(value) for value in cached_completed_scores]
    scores.append(_score(prefix, query, epsilon))
    probabilities = _softmax(scores)
    output = np.zeros_like(prefix, dtype=np.float32)
    for index, value in enumerate(completed):
        output = np.asarray(
            output + np.float32(probabilities[index]) * value,
            dtype=np.float32,
        )
    output = np.asarray(
        output + np.float32(probabilities[-1]) * prefix,
        dtype=np.float32,
    )
    return np.ascontiguousarray(output), np.asarray(scores, dtype=np.float32)


def _future_score_names(reader: _CheckpointReader, start_layer: int) -> list[str]:
    names: list[str] = []
    for layer in range(start_layer + 1, reader.config.layers):
        prefix = f"language_model.model.layers.{layer}"
        for stem in ("self_attention", "mlp"):
            norm = f"{prefix}.{stem}_res_norm.weight"
            projection = f"{prefix}.{stem}_res_proj.weight"
            if norm in reader.weight_map and projection in reader.weight_map:
                names.append(f"__product__::{norm}::{projection}")
    final_norm = "language_model.model.output_attn_res_norm.weight"
    final_projection = "language_model.model.output_attn_res_proj.weight"
    if final_norm in reader.weight_map and final_projection in reader.weight_map:
        names.append(f"__product__::{final_norm}::{final_projection}")
    return names


def _queries(reader: _CheckpointReader, names: list[str]) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for name in names:
        if name.startswith("__product__::"):
            _marker, norm_name, projection_name = name.split("::", 2)
            result.append(
                np.ascontiguousarray(
                    reader.f32(norm_name) * reader.f32(projection_name),
                    dtype=np.float32,
                ).reshape(-1)
            )
        else:
            raise ValueError(f"invalid future-score query descriptor: {name}")
    return result


def _gpu_bound(
    runtime: _CudaRuntime,
    prefix: np.ndarray,
    completed: np.ndarray,
    query: np.ndarray,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    prefix_device = runtime.allocate(prefix.nbytes)
    residual_device = runtime.allocate(completed.nbytes)
    query_device = runtime.allocate(query.nbytes)
    output_device = runtime.allocate(prefix.nbytes)
    try:
        runtime.upload_activation(prefix_device, prefix)
        runtime.upload_activation(residual_device, completed)
        runtime.upload_activation(query_device, query)
        result: dict[str, Any] = {}
        for blocks in (1, 8):
            for _ in range(warmup):
                runtime.execute_attnres_mix(
                    output_device,
                    prefix_device,
                    residual_device,
                    query_device,
                    block_count=blocks,
                    dimension=HIDDEN,
                    epsilon=EPSILON,
                )
            runtime.synchronize()
            wall: list[float] = []
            device: list[float] = []
            for _ in range(iterations):
                started = time.perf_counter_ns()
                runtime.profile_begin()
                runtime.execute_attnres_mix(
                    output_device,
                    prefix_device,
                    residual_device,
                    query_device,
                    block_count=blocks,
                    dimension=HIDDEN,
                    epsilon=EPSILON,
                )
                device.append(runtime.profile_end())
                wall.append((time.perf_counter_ns() - started) / 1e6)
            result[str(blocks)] = {"wall": _timing(wall), "device": _timing(device)}
        p50_delta = max(
            0.0,
            float(result["8"]["device"]["p50_ms"])
            - float(result["1"]["device"]["p50_ms"]),
        )
        result["residual_score_saving_upper_bound_device_ms_per_mix"] = p50_delta
        result["bound_reason"] = (
            "T(block_count=8)-T(block_count=1) also removes seven vector-mix "
            "terms, so it is an upper bound—not a claimed cached-kernel speedup"
        )
        return result
    finally:
        for pointer in (output_device, query_device, residual_device, prefix_device):
            runtime.free(pointer)


def benchmark(
    checkpoint: Path,
    cuda_library: Path,
    service_receipt: Path,
    output_path: Path,
    *,
    start_layer: int = 0,
    bound_layer: int = 84,
    device: int = 0,
    warmup: int = 30,
    iterations: int = 200,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "RUNNING",
        "started_unix_ns": time.time_ns(),
        "configuration": {
            "start_layer": start_layer,
            "physical_bound_layer": bound_layer,
            "warmup": warmup,
            "iterations": iterations,
            "epsilon": EPSILON,
        },
        "sources": {
            "checkpoint_config_sha256": _sha256_file(checkpoint / "config.json"),
            "checkpoint_index_sha256": _sha256_file(
                checkpoint / "model.safetensors.index.json"
            ),
            "cuda_library_sha256": _sha256_file(cuda_library),
            "benchmark_source_sha256": _sha256_file(Path(__file__)),
        },
    }
    _atomic_json(output_path, receipt)
    runtime: _CudaRuntime | None = None
    try:
        reader = _CheckpointReader(checkpoint)
        names = _future_score_names(reader, start_layer)
        queries = _queries(reader, names)
        if not queries or any(query.shape != (HIDDEN,) for query in queries):
            raise RuntimeError("future AttnRes query set is incomplete")
        generator = np.random.default_rng(18018)
        completed = np.ascontiguousarray(
            generator.normal(0.0, 0.2, size=(8, HIDDEN)), dtype=np.float32
        )
        prefix = np.ascontiguousarray(
            generator.normal(0.0, 0.2, size=HIDDEN), dtype=np.float32
        )
        precompute_started = time.perf_counter_ns()
        score_cache = np.empty((len(queries), 8), dtype=np.float32)
        for query_index, query in enumerate(queries):
            for block_index, value in enumerate(completed):
                score_cache[query_index, block_index] = _score(value, query, EPSILON)
        precompute_ms = (time.perf_counter_ns() - precompute_started) / 1e6
        comparisons: list[dict[str, Any]] = []
        for index, query in enumerate(queries):
            reference, reference_scores = reference_mix(
                prefix, completed, query, epsilon=EPSILON
            )
            cached, cached_scores = cached_mix(
                prefix,
                completed,
                query,
                score_cache[index],
                epsilon=EPSILON,
            )
            score_exact = bool(np.array_equal(reference_scores, cached_scores))
            output_exact = bool(np.array_equal(reference, cached))
            comparisons.append(
                {
                    "query_name": names[index],
                    "scores_bit_exact": score_exact,
                    "output_bit_exact": output_exact,
                    "output_metrics": _numerical_metrics(reference, cached),
                }
            )
        runtime = _CudaRuntime(cuda_library, device)
        physical_bound = _gpu_bound(
            runtime,
            prefix,
            completed,
            queries[-1],
            warmup=warmup,
            iterations=iterations,
        )
        service = json.loads(service_receipt.read_text(encoding="utf-8"))
        layer_result = service["layers"].get(str(bound_layer))
        if layer_result is None:
            raise RuntimeError("physical bound layer is absent from service receipt")
        rows: list[dict[str, Any]] = []
        saving_per_mix = float(
            physical_bound["residual_score_saving_upper_bound_device_ms_per_mix"]
        )
        for row_count in (1, 2, 4, 8):
            layer_p50 = float(layer_result["service"][str(row_count)]["wall"]["p50_ms"])
            saving = 2 * row_count * saving_per_mix
            rows.append(
                {
                    "rows": row_count,
                    "measured_full_layer_wall_p50_ms": layer_p50,
                    "physical_saving_upper_bound_ms": saving,
                    "full_layer_upper_bound_fraction": saving / layer_p50,
                    "retained": False,
                }
            )
        maximum_fraction = max(float(row["full_layer_upper_bound_fraction"]) for row in rows)
        receipt.update(
            {
                "future_query_count": len(queries),
                "completed_block_count": 8,
                "score_cache_bytes": score_cache.nbytes,
                "precompute_cpu_wall_ms": precompute_ms,
                "completed_state_fingerprint": _array_fingerprint(completed),
                "score_cache_fingerprint": _array_fingerprint(score_cache),
                "comparisons": comparisons,
                "all_scores_bit_exact": all(
                    bool(row["scores_bit_exact"]) for row in comparisons
                ),
                "all_outputs_bit_exact": all(
                    bool(row["output_bit_exact"]) for row in comparisons
                ),
                "physical_gpu_kernel_bound": physical_bound,
                "full_layer_results": rows,
                "maximum_full_layer_saving_upper_bound_fraction": maximum_fraction,
                "retained": False,
                "kernel_prototype_warranted_by_bound": maximum_fraction > 0.10,
                "retention_rule": (
                    "an upper bound never qualifies for retention; a cached kernel must "
                    "show a measured full-layer wall-time gain"
                ),
            }
        )
        receipt["status"] = (
            "PASS"
            if receipt["all_scores_bit_exact"] and receipt["all_outputs_bit_exact"]
            else "FAIL"
        )
        receipt["finished_unix_ns"] = time.time_ns()
        _atomic_json(output_path, receipt)
        return receipt
    except BaseException as exc:
        receipt["status"] = "FAIL"
        receipt["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        receipt["finished_unix_ns"] = time.time_ns()
        _atomic_json(output_path, receipt)
        raise
    finally:
        if runtime is not None:
            runtime.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cuda-library", type=Path, required=True)
    parser.add_argument("--service-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--bound-layer", type=int, default=84)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    arguments = parser.parse_args()
    benchmark(
        arguments.checkpoint,
        arguments.cuda_library,
        arguments.service_receipt,
        arguments.output,
        start_layer=arguments.start_layer,
        bound_layer=arguments.bound_layer,
        device=arguments.device,
        warmup=arguments.warmup,
        iterations=arguments.iterations,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
