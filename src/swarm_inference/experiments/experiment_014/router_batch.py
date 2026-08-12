"""Incremental real-weight batched-router certification for H014-030a."""

from __future__ import annotations

import ctypes
import json
import os
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.experiments.experiment_014.batch_safety import _health_snapshot
from swarm_inference.experiments.experiment_014.cuda import (
    KimiCudaError,
    _array_fingerprint,
    _CudaRuntime,
    _device_identity,
    _load_safetensor_f32,
    _numerical_metrics,
    _sha256_file,
)
from swarm_inference.experiments.experiment_014.full_cuda import _pointer_offset

SCHEMA_VERSION = "experiment-014-k3-batched-router-v1"
TARGET_BATCHES = (1, 2, 4, 8)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _timing(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(percent: float) -> float:
        position = (len(ordered) - 1) * percent
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    return {
        "minimum_ms": min(ordered),
        "maximum_ms": max(ordered),
        "mean_ms": statistics.fmean(ordered),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "standard_deviation_ms": statistics.pstdev(ordered),
    }


def _load_router(checkpoint: Path, layer: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    index_path = checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    prefix = f"language_model.model.layers.{layer}.block_sparse_moe.gate"
    router, router_evidence = _load_safetensor_f32(
        checkpoint, index, f"{prefix}.weight"
    )
    bias, bias_evidence = _load_safetensor_f32(
        checkpoint, index, f"{prefix}.e_score_correction_bias"
    )
    if router.shape != (896, 7168) or bias.shape != (896,):
        raise KimiCudaError(
            f"unexpected router geometry W={router.shape}, b={bias.shape}"
        )
    return (
        np.ascontiguousarray(router, dtype=np.float32),
        np.ascontiguousarray(bias, dtype=np.float32),
        {"router": router_evidence, "bias": bias_evidence},
    )


def _activation_rows(trace_path: Path) -> tuple[np.ndarray, tuple[int, ...]]:
    trace = np.memmap(trace_path, mode="r", dtype="<f4", shape=(3 * 94, 7168))
    row_indices = (0, 13, 47, 89, 94, 137, 188, 275)
    rows = np.ascontiguousarray(trace[list(row_indices)], dtype=np.float32)
    return rows, row_indices


def _upload_fixture(
    runtime: _CudaRuntime,
    rows: np.ndarray,
    router: np.ndarray,
    bias: np.ndarray,
) -> tuple[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]:
    x_device = runtime.allocate(rows.nbytes)
    w_device = runtime.allocate(router.nbytes)
    b_device = runtime.allocate(bias.nbytes)
    runtime.upload_activation(x_device, rows)
    runtime.upload_activation(w_device, router)
    runtime.upload_activation(b_device, bias)
    runtime.synchronize()
    return x_device, w_device, b_device


def _free_fixture(
    runtime: _CudaRuntime,
    pointers: tuple[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p],
) -> None:
    for pointer in reversed(pointers):
        runtime.free(pointer)


def _serial_routes(
    runtime: _CudaRuntime,
    pointers: tuple[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p],
    *,
    batch: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    effective: list[int] = []
    for row in range(batch):
        row_indices, row_weights, row_effective = runtime.route(
            _pointer_offset(pointers[0], row * 7168),
            pointers[1],
            pointers[2],
            hidden=7168,
            experts=896,
            topk=16,
        )
        indices.append(row_indices)
        weights.append(row_weights)
        effective.append(row_effective)
    return (
        np.stack(indices),
        np.stack(weights),
        np.asarray(effective, dtype=np.int32),
    )


def _measure_serial(
    runtime: _CudaRuntime,
    pointers: tuple[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p],
    *,
    batch: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        _serial_routes(runtime, pointers, batch=batch)
    wall_ms: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        _serial_routes(runtime, pointers, batch=batch)
        wall_ms.append((time.perf_counter_ns() - started) / 1e6)
    return {"wall": _timing(wall_ms), "native_calls_per_batch": batch}


def _measure_batch(
    runtime: _CudaRuntime,
    pointers: tuple[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p],
    *,
    batch: int,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    for _ in range(warmup):
        runtime.route_batch(
            pointers[0],
            pointers[1],
            pointers[2],
            batch=batch,
            hidden=7168,
            experts=896,
            topk=16,
        )
    wall_ms: list[float] = []
    observed: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    for _ in range(iterations):
        started = time.perf_counter_ns()
        observed = runtime.route_batch(
            pointers[0],
            pointers[1],
            pointers[2],
            batch=batch,
            hidden=7168,
            experts=896,
            topk=16,
        )
        wall_ms.append((time.perf_counter_ns() - started) / 1e6)
    runtime.set_telemetry("detailed")
    runtime.reset_router_stats()
    for _ in range(min(50, iterations)):
        runtime.route_batch(
            pointers[0],
            pointers[1],
            pointers[2],
            batch=batch,
            hidden=7168,
            experts=896,
            topk=16,
            accumulate_stats=True,
        )
    detailed = runtime.router_stats()
    runtime.set_telemetry("minimal")
    if observed is None:
        raise RuntimeError("batched router produced no retained output")
    return (
        {
            "wall": _timing(wall_ms),
            "native_calls_per_batch": 1,
            "detailed_device": detailed,
        },
        observed,
    )


def _guard_batch_three(
    runtime: _CudaRuntime,
    pointers: tuple[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p],
) -> dict[str, Any]:
    native = runtime.router_batch_function
    if native is None:
        raise RuntimeError("batched-router sentinel requires its native export")
    attempts = 0

    def sentinel(*_args: Any) -> int:
        nonlocal attempts
        attempts += 1
        raise AssertionError("unsupported router batch reached native CUDA")

    error: str | None = None
    try:
        runtime.router_batch_function = sentinel
        runtime.route_batch(
            pointers[0],
            pointers[1],
            pointers[2],
            batch=3,
            hidden=7168,
            experts=896,
            topk=16,
        )
    except KimiCudaError as exc:
        error = str(exc)
    finally:
        runtime.router_batch_function = native
    return {
        "requested_batch": 3,
        "native_call_attempts": attempts,
        "error": error,
        "pass": attempts == 0 and error is not None and "requested=3" in error,
    }


def benchmark_batched_router(
    checkpoint: Path,
    candidate_library: Path,
    prior_library: Path,
    oracle_trace: Path,
    output_path: Path,
    *,
    layer: int = 89,
    device: int = 0,
    warmup: int = 30,
    iterations: int = 200,
    cycle_id: str = "H014-030a",
) -> dict[str, Any]:
    """Certify one native batched router incrementally through eight rows."""
    if warmup < 10 or iterations < 100:
        raise ValueError("batched-router certification requires >=10/100 calls")
    paths = {
        "checkpoint": checkpoint.resolve(),
        "candidate_library": candidate_library.resolve(),
        "prior_library": prior_library.resolve(),
        "oracle_trace": oracle_trace.resolve(),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "cycle_id": cycle_id,
        "status": "RUNNING",
        "hypothesis": {
            "prediction": (
                "One batched router logits/select launch pair and one D2H result "
                "transfer preserve exact per-row routes and cut batch-8 router "
                "service by at least 2x versus eight serial router calls."
            ),
            "minimum_batch8_speedup": 2.0,
        },
        "implementation": {
            "logits_grid": "[ceil(896/128), rows] with unchanged per-logit accumulation order",
            "selection_grid": "one unchanged deterministic 256-thread block per row",
            "d2h_transfers_per_batch": 1,
            "attempted_batches": list(TARGET_BATCHES),
            "batch16_executed": False,
        },
        "configuration": {
            "layer": layer,
            "device": device,
            "warmup_calls": warmup,
            "retained_calls": iterations,
        },
        "sources": {
            name: {
                "path": str(path),
                "sha256": _sha256_file(path) if path.is_file() else None,
            }
            for name, path in paths.items()
        },
        "progress": [],
        "batches": {},
    }

    def retain(phase: str, **evidence: Any) -> None:
        receipt["progress"].append({"phase": phase, **evidence})
        _atomic_json(output_path, receipt)

    retain("preregistered")
    candidate_runtime: _CudaRuntime | None = None
    prior_runtime: _CudaRuntime | None = None
    candidate_pointers: tuple[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p] | None = None
    prior_pointers: tuple[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p] | None = None
    try:
        health_before = _health_snapshot(device)
        receipt["gpu_health_before"] = health_before
        receipt["device_identity"] = _device_identity(device)
        retain("gpu_health_before", status=health_before["status"])
        if health_before["status"] != "MEASURED":
            raise RuntimeError("nvidia-smi unavailable before batched-router CUDA")

        router, bias, weight_evidence = _load_router(paths["checkpoint"], layer)
        rows, trace_rows = _activation_rows(paths["oracle_trace"])
        receipt["fixture"] = {
            "real_checkpoint_weights": True,
            "real_activation_trace": True,
            "natural_layer89_boundary": False,
            "qualification": (
                "Eight distinct real Kimi hidden-trace rows are routed through real "
                "layer-89 weights to avoid a duplicated-row cache-only result."
            ),
            "trace_row_indices": list(trace_rows),
            "activation_shape": list(rows.shape),
            "activation_fingerprint": _array_fingerprint(rows),
            "router_fingerprint": _array_fingerprint(router),
            "bias_fingerprint": _array_fingerprint(bias),
            "weight_evidence": weight_evidence,
        }

        prior_runtime = _CudaRuntime(paths["prior_library"], device)
        prior_runtime.set_telemetry("minimal")
        prior_pointers = _upload_fixture(prior_runtime, rows, router, bias)
        prior_serial = _serial_routes(prior_runtime, prior_pointers, batch=8)
        receipt["prior_control"] = {
            "binary_sha256": prior_runtime.sha256,
            "indices_fingerprint": _array_fingerprint(prior_serial[0]),
            "weights_fingerprint": _array_fingerprint(prior_serial[1]),
            "effective": prior_serial[2].tolist(),
        }
        _free_fixture(prior_runtime, prior_pointers)
        prior_pointers = None
        prior_runtime.close()
        prior_runtime = None
        retain("prior_serial_control")

        candidate_runtime = _CudaRuntime(paths["candidate_library"], device)
        candidate_runtime.set_telemetry("minimal")
        if candidate_runtime.router_batch_function is None:
            raise RuntimeError("candidate omitted the batched-router export")
        candidate_pointers = _upload_fixture(candidate_runtime, rows, router, bias)
        candidate_serial = _serial_routes(candidate_runtime, candidate_pointers, batch=8)
        serial_binary_exact = all(
            np.array_equal(candidate, prior)
            for candidate, prior in zip(candidate_serial, prior_serial, strict=True)
        )
        receipt["candidate"] = {
            "binary_sha256": candidate_runtime.sha256,
            "supported_batches": list(candidate_runtime.expert_supported_batches),
            "router_batch_export": True,
            "serial_path_bit_exact_to_prior": serial_binary_exact,
        }
        retain("candidate_serial_control", bit_exact=serial_binary_exact)
        if not serial_binary_exact:
            raise RuntimeError("candidate changed the retained serial router output")

        for batch in TARGET_BATCHES:
            retain(
                f"armed_batch_{batch}",
                prior_batches_passed=[
                    size
                    for size in TARGET_BATCHES
                    if size < batch and receipt["batches"].get(str(size), {}).get("status") == "PASS"
                ],
            )
            serial_expected = tuple(value[:batch].copy() for value in candidate_serial)
            serial_timing = _measure_serial(
                candidate_runtime,
                candidate_pointers,
                batch=batch,
                warmup=warmup,
                iterations=iterations,
            )
            batch_timing, observed = _measure_batch(
                candidate_runtime,
                candidate_pointers,
                batch=batch,
                warmup=warmup,
                iterations=iterations,
            )
            ids_equal = bool(np.array_equal(observed[0], serial_expected[0]))
            weights_equal = bool(np.array_equal(observed[1], serial_expected[1]))
            effective_equal = bool(np.array_equal(observed[2], serial_expected[2]))
            weight_metrics = _numerical_metrics(serial_expected[1], observed[1])
            speedup = float(serial_timing["wall"]["p50_ms"]) / float(
                batch_timing["wall"]["p50_ms"]
            )
            candidate_runtime.synchronize()
            sticky_ok = candidate_runtime.error_state_ok()
            safe_ids, safe_weights, safe_effective = candidate_runtime.route(
                candidate_pointers[0],
                candidate_pointers[1],
                candidate_pointers[2],
                hidden=7168,
                experts=896,
                topk=16,
            )
            safe_exact = (
                np.array_equal(safe_ids, candidate_serial[0][0])
                and np.array_equal(safe_weights, candidate_serial[1][0])
                and safe_effective == int(candidate_serial[2][0])
            )
            health = _health_snapshot(device)
            row = {
                "correctness": {
                    "ids_bit_exact": ids_equal,
                    "weights_bit_exact": weights_equal,
                    "effective_count_exact": effective_equal,
                    "weight_metrics": weight_metrics,
                    "selected_ids": observed[0].tolist(),
                    "all_16_unique_per_row": all(
                        len(set(item.tolist())) == 16 for item in observed[0]
                    ),
                },
                "serial": serial_timing,
                "batched": batch_timing,
                "aggregate_speedup_vs_serial": speedup,
                "aggregate_rows_per_second": batch * 1000.0
                / float(batch_timing["wall"]["p50_ms"]),
                "d2h_bytes_per_batch": batch * (16 * (4 + 4) + 4),
                "post_batch": {
                    "cuda_synchronize": "PASS",
                    "cuda_error_state_ok": sticky_ok,
                    "known_safe_serial_exact": safe_exact,
                    "free_vram_bytes": candidate_runtime.mem_info()["free_bytes"],
                    "nvidia_smi": health,
                },
            }
            row["status"] = (
                "PASS"
                if ids_equal
                and weights_equal
                and effective_equal
                and safe_exact
                and sticky_ok
                and health["status"] == "MEASURED"
                else "FAIL"
            )
            receipt["batches"][str(batch)] = row
            retain(
                f"batch_{batch}_persisted_and_checked",
                status=row["status"],
                speedup=speedup,
                nvidia_smi=health["status"],
            )
            if row["status"] != "PASS":
                raise RuntimeError(f"batched router failed at batch {batch}")

        unsupported = _guard_batch_three(candidate_runtime, candidate_pointers)
        receipt["unsupported_batch_preflight"] = unsupported
        retain("unsupported_batch_3_rejected", passed=unsupported["pass"])
        batch8_speedup = float(
            receipt["batches"]["8"]["aggregate_speedup_vs_serial"]
        )
        receipt["hypothesis_supported"] = batch8_speedup >= 2.0
        receipt["inspection"] = {
            "batch8_speedup": batch8_speedup,
            "batch8_serial_p50_ms": receipt["batches"]["8"]["serial"]["wall"][
                "p50_ms"
            ],
            "batch8_batched_p50_ms": receipt["batches"]["8"]["batched"]["wall"][
                "p50_ms"
            ],
            "actual_bottleneck": (
                "The batched router is logits-compute bound; complete-stage "
                "attention/pre-MoE and routed-expert service must now be measured."
            ),
        }
        receipt["decision"] = (
            "RETAIN_FOR_COMPLETE_STAGE_TEST"
            if receipt["hypothesis_supported"]
            else "MODIFY_OR_REVERT"
        )
        receipt["gpu_health_after"] = _health_snapshot(device)
        receipt["status"] = (
            "PASS"
            if all(row["status"] == "PASS" for row in receipt["batches"].values())
            and unsupported["pass"]
            and receipt["gpu_health_after"]["status"] == "MEASURED"
            else "FAIL"
        )
        retain("complete", status=receipt["status"])
        return receipt
    except Exception as exc:
        receipt["status"] = "FAIL"
        receipt["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "gpu_health": _health_snapshot(device),
        }
        _atomic_json(output_path, receipt)
        raise
    finally:
        if prior_runtime is not None:
            if prior_pointers is not None:
                _free_fixture(prior_runtime, prior_pointers)
            prior_runtime.close()
        if candidate_runtime is not None:
            if candidate_pointers is not None:
                _free_fixture(candidate_runtime, candidate_pointers)
            candidate_runtime.close()
