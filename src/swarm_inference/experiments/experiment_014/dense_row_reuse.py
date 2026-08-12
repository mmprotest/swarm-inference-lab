"""Real-weight row-cooperative dense projection benchmark for H014-030d."""

from __future__ import annotations

import json
import os
import statistics
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.experiments.experiment_014.batch_safety import _health_snapshot
from swarm_inference.experiments.experiment_014.cuda import (
    KimiCudaError,
    _array_fingerprint,
    _numerical_metrics,
    _sha256_file,
)
from swarm_inference.experiments.experiment_014.full_cuda import (
    KimiCudaGraphRunner,
    _LayerResources,
)

SCHEMA_VERSION = "experiment-014-k3-dense-row-reuse-v1"
TARGET_BATCHES = (1, 2, 4, 8)
TRACE_ROWS = (0, 13, 47, 89, 94, 137, 188, 275)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _timing(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "minimum_ms": min(ordered),
        "maximum_ms": max(ordered),
        "mean_ms": statistics.fmean(ordered),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "standard_deviation_ms": statistics.pstdev(ordered),
    }


def _measure(
    runner: KimiCudaGraphRunner,
    action: Callable[[], None],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    runtime = runner.runtime
    for _ in range(warmup):
        action()
    runtime.synchronize()
    device_values: list[float] = []
    wall_values: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        runtime.profile_begin()
        action()
        device_values.append(runtime.profile_end())
        wall_values.append((time.perf_counter_ns() - started) / 1e6)
    return {"device": _timing(device_values), "wall": _timing(wall_values)}


def benchmark_dense_row_reuse(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    output_path: Path,
    *,
    layer: int = 89,
    role: str = "q",
    device: int = 0,
    warmup: int = 30,
    iterations: int = 200,
    cycle_id: str = "H014-030d",
) -> dict[str, Any]:
    """Compare current and weight-reusing kernels through exact batch 8."""
    if warmup < 10 or iterations < 100:
        raise ValueError("dense row-reuse certification requires >=10/100 calls")
    if layer != 89 or role != "q":
        raise ValueError("H014-030d is scoped to real layer-89 KDA q projection")
    sources = {
        "checkpoint": checkpoint.resolve(),
        "cuda_library": cuda_library.resolve(),
        "oracle_trace": oracle_trace.resolve(),
    }
    for name, path in sources.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "cycle_id": cycle_id,
        "status": "RUNNING",
        "hypothesis": {
            "prediction": (
                "A batch-8 dense kernel that loads each real layer-89 KDA q "
                "projection weight once for eight rows is bit exact and at least "
                "2x faster than the current independent-row batch kernel."
            ),
            "minimum_batch8_speedup": 2.0,
        },
        "implementation": {
            "row_accumulators": "2/4/8 compile-time specializations",
            "weight_loads_per_input_per_output": 1,
            "same_thread_traversal_and_reduction_tree": True,
            "attempted_batches": list(TARGET_BATCHES),
            "batch16_executed": False,
        },
        "configuration": {
            "layer": layer,
            "role": role,
            "device": device,
            "warmup_calls": warmup,
            "retained_calls": iterations,
        },
        "sources": {
            name: {
                "path": str(path),
                "sha256": _sha256_file(path) if path.is_file() else None,
            }
            for name, path in sources.items()
        },
        "progress": [],
        "batches": {},
    }

    def retain(phase: str, **evidence: Any) -> None:
        receipt["progress"].append({"phase": phase, **evidence})
        _atomic_json(output_path, receipt)

    retain("preregistered")
    runner: KimiCudaGraphRunner | None = None
    resources: _LayerResources | None = None
    try:
        health_before = _health_snapshot(device)
        receipt["gpu_health_before"] = health_before
        retain("gpu_health_before", status=health_before["status"])
        if health_before["status"] != "MEASURED":
            raise RuntimeError("nvidia-smi unavailable before dense CUDA benchmark")

        runner = KimiCudaGraphRunner(checkpoint, cuda_library, device)
        runtime = runner.runtime
        if runtime.dense_rows_reuse_function is None:
            raise RuntimeError("candidate lacks row-reuse dense export")
        resources = _LayerResources(runtime)
        tensor_name = (
            f"language_model.model.layers.{layer}.self_attn.q_proj.weight"
        )
        tensor = runner._upload_int4(resources, tensor_name)
        trace = np.memmap(oracle_trace, dtype=np.float32, mode="r")
        if trace.size % runner.config.hidden:
            raise KimiCudaError("oracle trace is not aligned to Kimi hidden size")
        trace = trace.reshape(-1, runner.config.hidden)
        if max(TRACE_ROWS) >= trace.shape[0]:
            raise KimiCudaError("oracle trace does not contain selected diverse rows")
        rows = np.ascontiguousarray(trace[list(TRACE_ROWS)], dtype=np.float32)
        outputs = runner.config.kda_projection
        source_dev = resources.allocate(rows.size)
        baseline_dev = resources.allocate(len(TRACE_ROWS) * outputs)
        reuse_dev = resources.allocate(len(TRACE_ROWS) * outputs)
        runtime.upload_activation(source_dev, rows)
        receipt["fixture"] = {
            "tensor_name": tensor_name,
            "tensor_input": runner.config.hidden,
            "tensor_output": outputs,
            "resident_tensor_bytes": runtime.tensor_bytes(tensor),
            "runtime_format": "grouped_int4",
            "trace_rows": list(TRACE_ROWS),
            "activation_fingerprint": _array_fingerprint(rows),
            "qualification": (
                "Eight distinct real Kimi hidden-trace rows; this isolates the "
                "projection kernel and is not claimed as a natural layer-89 boundary."
            ),
        }
        retain("real_fixture_resident", **receipt["fixture"])

        for batch in TARGET_BATCHES:
            retain(
                f"armed_batch_{batch}",
                prior_batches_passed=[
                    size
                    for size in TARGET_BATCHES
                    if size < batch
                    and receipt["batches"].get(str(size), {}).get("status") == "PASS"
                ],
            )
            baseline = _measure(
                runner,
                lambda b=batch: runtime.execute_dense(
                    tensor, baseline_dev, source_dev, b
                ),
                warmup=warmup,
                iterations=iterations,
            )
            runtime.execute_dense(tensor, baseline_dev, source_dev, batch)
            runtime.synchronize()
            expected = runtime.download_activation(baseline_dev, (batch, outputs))

            reuse = _measure(
                runner,
                lambda b=batch: runtime.execute_dense_rows_reuse(
                    tensor, reuse_dev, source_dev, b
                ),
                warmup=warmup,
                iterations=iterations,
            )
            runtime.execute_dense_rows_reuse(tensor, reuse_dev, source_dev, batch)
            runtime.synchronize()
            observed = runtime.download_activation(reuse_dev, (batch, outputs))
            bit_exact = bool(np.array_equal(expected, observed))
            metrics = _numerical_metrics(expected, observed)
            speedup = baseline["device"]["p50_ms"] / reuse["device"]["p50_ms"]

            runtime.synchronize()
            sticky_ok = runtime.error_state_ok()
            runtime.execute_dense(tensor, baseline_dev, source_dev, 1)
            runtime.synchronize()
            safe = runtime.download_activation(baseline_dev, (1, outputs))
            safe_ok = bool(np.isfinite(safe).all())
            health = _health_snapshot(device)
            row = {
                "baseline_independent_rows": baseline,
                "row_reuse": reuse,
                "speedup": speedup,
                "aggregate_rows_per_second": (
                    batch * 1000.0 / reuse["device"]["p50_ms"]
                ),
                "correctness": {
                    "bit_exact": bit_exact,
                    "metrics": metrics,
                    "baseline_fingerprint": _array_fingerprint(expected),
                    "observed_fingerprint": _array_fingerprint(observed),
                },
                "post_batch": {
                    "cuda_synchronize": "PASS",
                    "cuda_error_state_ok": sticky_ok,
                    "known_safe_fixture_finite": safe_ok,
                    "free_vram_bytes": runtime.mem_info()["free_bytes"],
                    "nvidia_smi": health,
                },
            }
            row["status"] = (
                "PASS"
                if bit_exact
                and sticky_ok
                and safe_ok
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
                raise RuntimeError(f"row-reuse dense failed at batch {batch}")

        attempts = 0
        native = runtime.dense_rows_reuse_function

        def sentinel(*_args: object) -> int:
            nonlocal attempts
            attempts += 1
            return 0

        runtime.dense_rows_reuse_function = sentinel
        error: str | None = None
        try:
            runtime.execute_dense_rows_reuse(tensor, reuse_dev, source_dev, 3)
        except KimiCudaError as exc:
            error = str(exc)
        finally:
            runtime.dense_rows_reuse_function = native
        receipt["unsupported_batch_preflight"] = {
            "requested_batch": 3,
            "native_call_attempts": attempts,
            "error": error,
            "pass": attempts == 0 and error is not None and "requested=3" in error,
        }
        batch8 = receipt["batches"]["8"]
        receipt["hypothesis_supported"] = bool(batch8["speedup"] >= 2.0)
        receipt["inspection"] = {
            "batch8_baseline_device_p50_ms": batch8[
                "baseline_independent_rows"
            ]["device"]["p50_ms"],
            "batch8_row_reuse_device_p50_ms": batch8["row_reuse"]["device"][
                "p50_ms"
            ],
            "batch8_speedup": batch8["speedup"],
            "actual_bottleneck": (
                "pending compiled-resource and complete-stage inspection"
            ),
        }
        receipt["decision"] = (
            "RETAIN_FOR_ATTENTION_BATCH_INTEGRATION"
            if receipt["hypothesis_supported"]
            else "REVERT_OR_REDESIGN"
        )
        receipt["gpu_health_after"] = _health_snapshot(device)
        receipt["status"] = (
            "PASS"
            if all(item["status"] == "PASS" for item in receipt["batches"].values())
            and receipt["unsupported_batch_preflight"]["pass"]
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
        if resources is not None:
            resources.close()
        if runner is not None:
            runner.close()
