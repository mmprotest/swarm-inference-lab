"""Physical native-MXFP4 Kimi K3 intra-expert microshard benchmark."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.execution.kimi_cuda_runtime import (
    _array_fingerprint,
    _CudaRuntime,
    _load_real_expert,
    _numerical_metrics,
    _sha256_file,
)
from swarm_inference.model.mxfp4 import MXFP4_GROUP_SIZE, MXFP4Tensor

SCHEMA_VERSION = "experiment-018-native-mxfp4-microshard-v1"
SPLIT_DEGREES = (1, 2, 4, 8, 16, 32)
BATCH_ROWS = (1, 2, 4, 8)
RELATIVE_L2_GATE = 2e-6


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


def _timing_p90(values: list[float]) -> dict[str, float]:
    result = _timing(values)
    result["p90_ms"] = float(np.percentile(values, 90))
    return result


def _slice_tensor_rows(tensor: MXFP4Tensor, start: int, end: int) -> MXFP4Tensor:
    return MXFP4Tensor(
        packed=np.ascontiguousarray(tensor.packed[start:end]).copy(),
        scales=np.ascontiguousarray(tensor.scales[start:end]).copy(),
        input_dimension=tensor.input_dimension,
        output_dimension=end - start,
    )


def _slice_tensor_columns(tensor: MXFP4Tensor, start: int, end: int) -> MXFP4Tensor:
    if start % MXFP4_GROUP_SIZE or end % MXFP4_GROUP_SIZE:
        raise ValueError("native MXFP4 microshards cannot split a scale group")
    return MXFP4Tensor(
        packed=np.ascontiguousarray(tensor.packed[:, start // 2 : end // 2]).copy(),
        scales=np.ascontiguousarray(
            tensor.scales[:, start // MXFP4_GROUP_SIZE : end // MXFP4_GROUP_SIZE]
        ).copy(),
        input_dimension=end - start,
        output_dimension=tensor.output_dimension,
    )


def split_native_expert(
    expert: Any,
    split_degree: int,
) -> tuple[tuple[MXFP4Tensor, MXFP4Tensor, MXFP4Tensor], ...]:
    """Return only physically sliced gate/up rows and matching down columns."""

    intermediate = int(expert.gate.output_dimension)
    if split_degree < 1 or intermediate % split_degree:
        raise ValueError("expert intermediate dimension is not divisible by split degree")
    width = intermediate // split_degree
    if width % MXFP4_GROUP_SIZE:
        raise ValueError("microshard width splits native MXFP4 groups")
    shards = []
    for shard in range(split_degree):
        start = shard * width
        end = start + width
        shards.append(
            (
                _slice_tensor_rows(expert.gate, start, end),
                _slice_tensor_rows(expert.up, start, end),
                _slice_tensor_columns(expert.down, start, end),
            )
        )
    return tuple(shards)


def _native_bytes(shard: tuple[MXFP4Tensor, MXFP4Tensor, MXFP4Tensor]) -> int:
    return sum(item.byte_size for item in shard)


def _stable_sum(partials: list[np.ndarray]) -> np.ndarray:
    result = np.zeros_like(partials[0], dtype=np.float32)
    for partial in partials:
        result += np.asarray(partial, dtype=np.float32)
    return result


def _upload_shard(
    runtime: _CudaRuntime,
    shard: tuple[MXFP4Tensor, MXFP4Tensor, MXFP4Tensor],
) -> tuple[Any, Any, Any]:
    gate = runtime.upload(shard[0])
    up = runtime.upload(shard[1])
    down = runtime.upload(shard[2])
    return gate, up, down


def _measure_handles(
    runtime: _CudaRuntime,
    handles: list[tuple[Any, Any, Any]],
    activation: np.ndarray,
    *,
    warmup: int,
    iterations: int,
    route_weight: np.float32,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    for _ in range(warmup):
        _stable_sum([runtime.execute(item, activation) for item in handles])
    wall_values: list[float] = []
    device_values: list[float] = []
    reduction_values: list[float] = []
    weighted_reduction_values: list[float] = []
    shard_wall_values: list[list[float]] = [[] for _ in handles]
    shard_device_values: list[list[float]] = [[] for _ in handles]
    final: np.ndarray | None = None
    weighted_final: np.ndarray | None = None
    for _ in range(iterations):
        started = time.perf_counter_ns()
        partials: list[np.ndarray] = []
        for index, item in enumerate(handles):
            shard_started = time.perf_counter_ns()
            runtime.profile_begin()
            partial = runtime.execute(item, activation)
            device_ms = runtime.profile_end()
            shard_wall_values[index].append(
                (time.perf_counter_ns() - shard_started) / 1e6
            )
            shard_device_values[index].append(device_ms)
            partials.append(partial)
        reduction_started = time.perf_counter_ns()
        final = _stable_sum(partials)
        reduction_values.append((time.perf_counter_ns() - reduction_started) / 1e6)
        weighted_started = time.perf_counter_ns()
        weighted_final = _stable_sum(
            [np.ascontiguousarray(route_weight * item) for item in partials]
        )
        weighted_reduction_values.append(
            (time.perf_counter_ns() - weighted_started) / 1e6
        )
        wall_values.append((time.perf_counter_ns() - started) / 1e6)
        device_values.append(sum(values[-1] for values in shard_device_values))
    if final is None or weighted_final is None:
        raise RuntimeError("microshard benchmark emitted no result")
    return (
        {
            "sequential_one_gpu_wall": _timing_p90(wall_values),
            "sequential_one_gpu_device": _timing_p90(device_values),
            "stable_fp32_reduction_wall": _timing_p90(reduction_values),
            "route_weighted_stable_fp32_reduction_wall": _timing_p90(
                weighted_reduction_values
            ),
            "per_shard_wall": [
                _timing_p90(values) for values in shard_wall_values
            ],
            "per_shard_device": [
                _timing_p90(values) for values in shard_device_values
            ],
            "independent_resource_compute_ceiling_ms": max(
                _timing_p90(values)["p50_ms"] for values in shard_wall_values
            ),
        },
        final,
        weighted_final,
    )


def benchmark(
    checkpoint: Path,
    cuda_library: Path,
    output_path: Path,
    *,
    layer: int = 89,
    expert_id: int = 885,
    device: int = 0,
    warmup: int = 5,
    iterations: int = 30,
) -> dict[str, Any]:
    if not checkpoint.exists() or not cuda_library.exists():
        raise FileNotFoundError("checkpoint or CUDA library is missing")
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "RUNNING",
        "started_unix_ns": time.time_ns(),
        "configuration": {
            "layer": layer,
            "expert_id": expert_id,
            "device": device,
            "split_degrees": list(SPLIT_DEGREES),
            "batch_rows": list(BATCH_ROWS),
            "warmup": warmup,
            "iterations": iterations,
            "relative_l2_gate": RELATIVE_L2_GATE,
        },
        "claim_boundary": {
            "execution": "PHYSICAL sequential execution of real native MXFP4 slices",
            "parallel_speedup": "VALIDATED INDEPENDENT-RESOURCE MODEL only",
            "one_gpu_is_not_claimed_as_independent_workers": True,
        },
        "environment": {
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "sources": {
            "checkpoint_config_sha256": _sha256_file(checkpoint / "config.json"),
            "checkpoint_index_sha256": _sha256_file(
                checkpoint / "model.safetensors.index.json"
            ),
            "cuda_library_sha256": _sha256_file(cuda_library),
            "benchmark_source_sha256": _sha256_file(Path(__file__)),
        },
        "results": [],
    }
    _atomic_json(output_path, receipt)
    runtime: _CudaRuntime | None = None
    try:
        expert = _load_real_expert(checkpoint, layer, expert_id)
        native_full_bytes = expert.gate.byte_size + expert.up.byte_size + expert.down.byte_size
        runtime = _CudaRuntime(cuda_library, device)
        runtime.set_telemetry("minimal")
        runtime.set_fused_gate_up(True)
        generator = np.random.default_rng(1801885)
        activations = {
            rows: np.ascontiguousarray(
                generator.normal(0.0, 0.08, size=(rows, expert.gate.input_dimension)),
                dtype=np.float32,
            )
            for rows in BATCH_ROWS
        }
        receipt["expert"] = {
            "gate_shape": [expert.gate.output_dimension, expert.gate.input_dimension],
            "up_shape": [expert.up.output_dimension, expert.up.input_dimension],
            "down_shape": [expert.down.output_dimension, expert.down.input_dimension],
            "source_shards": expert.source_shards,
            "native_checkpoint_bytes": native_full_bytes,
            "activation_fingerprints": {
                str(rows): _array_fingerprint(value)
                for rows, value in activations.items()
            },
        }
        for degree in SPLIT_DEGREES:
            physical = split_native_expert(expert, degree)
            handles = [_upload_shard(runtime, shard) for shard in physical]
            try:
                actual_runtime_bytes = [
                    sum(runtime.tensor_bytes(handle) for handle in item)
                    for item in handles
                ]
                if sum(actual_runtime_bytes) != native_full_bytes:
                    raise RuntimeError(
                        "runtime slice bytes do not reconstruct checkpoint expert bytes"
                    )
                for rows in BATCH_ROWS:
                    activation = activations[rows]
                    if degree == 1:
                        reference = runtime.execute(handles[0], activation)
                    else:
                        whole_shard = split_native_expert(expert, 1)[0]
                        whole_handles = _upload_shard(runtime, whole_shard)
                        try:
                            reference = runtime.execute(whole_handles, activation)
                        finally:
                            for handle in whole_handles:
                                runtime.release_tensor(handle)
                    route_weight = np.float32(0.073125)
                    measurements, reconstructed, weighted_reconstructed = _measure_handles(
                        runtime,
                        handles,
                        activation,
                        warmup=warmup,
                        iterations=iterations,
                        route_weight=route_weight,
                    )
                    metrics = _numerical_metrics(reference, reconstructed)
                    weighted_reference = np.ascontiguousarray(route_weight * reference)
                    weighted_metrics = _numerical_metrics(
                        weighted_reference, weighted_reconstructed
                    )
                    passed = float(metrics["relative_l2_error"]) <= RELATIVE_L2_GATE
                    weighted_passed = (
                        float(weighted_metrics["relative_l2_error"])
                        <= RELATIVE_L2_GATE
                    )
                    receipt["results"].append(
                        {
                            "evidence_class": "PHYSICAL",
                            "split_degree": degree,
                            "batch_rows": rows,
                            "intermediate_width_per_shard": 3072 // degree,
                            "native_checkpoint_bytes_per_shard": [
                                _native_bytes(shard) for shard in physical
                            ],
                            "runtime_bytes_per_shard": actual_runtime_bytes,
                            "maximum_runtime_bytes_per_worker": max(actual_runtime_bytes),
                            "input_payload_bytes_per_worker": activation.nbytes,
                            "output_contribution_bytes_per_worker": reference.nbytes,
                            "local_intermediate_activation_bytes_per_worker": (
                                rows * (3072 // degree) * 2 * 4
                            ),
                            "local_intermediate_not_transmitted": True,
                            "no_worker_owns_complete_expert": degree > 1,
                            "metrics": metrics,
                            "route_weighted_reduction": {
                                "route_weight": float(route_weight),
                                "metrics": weighted_metrics,
                                "pass": weighted_passed,
                                "reference_fingerprint": _array_fingerprint(
                                    weighted_reference
                                ),
                                "reconstructed_fingerprint": _array_fingerprint(
                                    weighted_reconstructed
                                ),
                                "order": "shard index ascending; FP32 stable accumulation",
                            },
                            "output_bit_exact": bool(
                                np.array_equal(reference, reconstructed)
                            ),
                            "reference_fingerprint": _array_fingerprint(reference),
                            "reconstructed_fingerprint": _array_fingerprint(
                                reconstructed
                            ),
                            "measurements": measurements,
                            "pass": passed and weighted_passed,
                        }
                    )
                    _atomic_json(output_path, receipt)
                    print(
                        f"[h018-microshard] degree={degree} rows={rows} "
                        f"rel_l2={metrics['relative_l2_error']:.3e} pass={passed}",
                        flush=True,
                    )
            finally:
                for item in handles:
                    for handle in item:
                        runtime.release_tensor(handle)
        highest = max(
            int(row["split_degree"])
            for row in receipt["results"]
            if bool(row["pass"])
        )
        receipt["highest_exact_degree"] = highest
        receipt["degree_32_pass"] = all(
            bool(row["pass"])
            for row in receipt["results"]
            if int(row["split_degree"]) == 32
        )
        receipt["degree_64"] = {
            "status": "INCOMPATIBLE_WITH_NATIVE_MXFP4_GROUPING",
            "reason": "3072 / 64 = 48 values, which splits 32-value scale groups",
        }
        receipt["status"] = "PASS" if receipt["degree_32_pass"] else "FAIL"
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=89)
    parser.add_argument("--expert", type=int, default=885)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    arguments = parser.parse_args()
    benchmark(
        arguments.checkpoint,
        arguments.cuda_library,
        arguments.output,
        layer=arguments.layer,
        expert_id=arguments.expert,
        device=arguments.device,
        warmup=arguments.warmup,
        iterations=arguments.iterations,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
