"""Fail-closed post-reboot validation for Experiment 014 H014-027w."""

from __future__ import annotations

import json
import os
import subprocess
import time
import traceback
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.experiments.experiment_014.cuda import (
    KimiCudaError,
    _array_fingerprint,
    _CudaRuntime,
    _device_identity,
    _load_real_expert,
    _numerical_metrics,
    _percentiles,
    _sha256_file,
)

SCHEMA_VERSION = "experiment-014-k3-failclosed-batch-validation-v1"
EXPECTED_REJECTION = (
    "Kimi CUDA expert batch rejected before launch: requested=4, certified_max=2"
)
MEMORY_STABILITY_TOLERANCE_BYTES = 4 * 1024**2
PERFORMANCE_REGRESSION_LIMIT = 1.20


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _health_snapshot(device: int) -> dict[str, Any]:
    fields = (
        "index,name,uuid,pci.bus_id,driver_version,compute_cap,memory.total,"
        "memory.free,memory.used,temperature.gpu,pstate"
    )
    command = [
        "nvidia-smi",
        f"--id={device}",
        f"--query-gpu={fields}",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=20
    )
    if completed.returncode:
        return {
            "status": "UNAVAILABLE",
            "returncode": completed.returncode,
            "stderr": completed.stderr.strip(),
        }
    values = [value.strip() for value in completed.stdout.strip().split(",")]
    names = fields.split(",")
    if len(values) != len(names):
        return {"status": "UNPARSEABLE", "stdout": completed.stdout.strip()}
    return {"status": "MEASURED", **dict(zip(names, values, strict=True))}


def _source_integrity(candidate_receipt: dict[str, Any]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name, source in candidate_receipt["sources"].items():
        path = Path(source["path"])
        actual = _sha256_file(path) if path.is_file() else None
        rows[name] = {
            "path": path.as_posix(),
            "expected_sha256": source["sha256"],
            "actual_sha256": actual,
            "match": actual == source["sha256"],
        }
    return {
        "artifacts": rows,
        "all_match": all(row["match"] for row in rows.values()),
    }


def _upload_expert(runtime: _CudaRuntime, expert: Any) -> tuple[Any, Any, Any]:
    return (
        runtime.upload(expert.gate),
        runtime.upload(expert.up),
        runtime.upload(expert.down),
    )


def _run_mode(
    runtime: _CudaRuntime,
    handles: tuple[Any, Any, Any],
    rows: np.ndarray,
    *,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, Any], np.ndarray]:
    source = np.ascontiguousarray(rows, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != 3584:
        raise ValueError(f"expected [batch,3584] source, got {source.shape}")
    batch = int(source.shape[0])
    memory_before = runtime.mem_info()
    input_pointer = runtime.allocate(source.nbytes)
    output_pointer = runtime.allocate(source.nbytes)
    try:
        runtime.upload_activation(input_pointer, source)
        memory_after_allocation = runtime.mem_info()
        for _ in range(warmup):
            runtime.execute_resident(handles, output_pointer, input_pointer, batch)
        runtime.synchronize()
        memory_after_warmup = runtime.mem_info()
        device_ms: list[float] = []
        wall_ms: list[float] = []
        for _ in range(iterations):
            wall_started = time.perf_counter_ns()
            runtime.profile_begin()
            runtime.execute_resident(handles, output_pointer, input_pointer, batch)
            device_ms.append(runtime.profile_end())
            wall_ms.append((time.perf_counter_ns() - wall_started) / 1e6)
        output = runtime.download_activation(output_pointer, source.shape)
    finally:
        runtime.free(output_pointer)
        runtime.free(input_pointer)
    runtime.synchronize()
    memory_after_free = runtime.mem_info()
    return (
        {
            "batch": batch,
            "warmup_calls": warmup,
            "retained_calls": iterations,
            "device": _percentiles(device_ms),
            "wall": _percentiles(wall_ms),
            "input_fingerprint": _array_fingerprint(source),
            "output_fingerprint": _array_fingerprint(output),
            "output_finite": bool(np.isfinite(output).all()),
            "timed_h2d_bytes": 0,
            "timed_d2h_bytes": 0,
            "memory": {
                "free_before_bytes": memory_before["free_bytes"],
                "free_after_allocation_bytes": memory_after_allocation["free_bytes"],
                "free_after_warmup_bytes": memory_after_warmup["free_bytes"],
                "free_after_free_bytes": memory_after_free["free_bytes"],
                "explicit_io_bytes": source.nbytes * 2,
                "persistent_growth_bytes": max(
                    0,
                    int(memory_before["free_bytes"])
                    - int(memory_after_free["free_bytes"]),
                ),
            },
        },
        output,
    )


def _guarded_batch4_rejection(
    runtime: _CudaRuntime,
    handles: tuple[Any, Any, Any],
    rows: np.ndarray,
) -> dict[str, Any]:
    """Exercise Python preflight while a sentinel prevents any native call."""
    source = np.ascontiguousarray(rows, dtype=np.float32)
    if source.shape != (4, 3584):
        raise ValueError(f"expected [4,3584] rejection source, got {source.shape}")
    memory_before = runtime.mem_info()
    input_pointer = runtime.allocate(source.nbytes)
    output_pointer = runtime.allocate(source.nbytes)
    native_name = "coli_cuda_kimi_expert_mlp_dev"
    native_function = getattr(runtime._library, native_name)
    native_call_attempts = 0

    def _native_sentinel(*_args: Any) -> int:
        nonlocal native_call_attempts
        native_call_attempts += 1
        raise AssertionError("unsafe native batch-4 call reached the CUDA boundary")

    try:
        runtime.upload_activation(input_pointer, source)
        runtime.synchronize()
        memory_before_request = runtime.mem_info()
        stats_before = runtime.stats()
        setattr(runtime._library, native_name, _native_sentinel)
        error: str | None = None
        try:
            runtime.execute_resident(handles, output_pointer, input_pointer, 4)
        except KimiCudaError as exc:
            error = str(exc)
        finally:
            setattr(runtime._library, native_name, native_function)
        runtime.synchronize()
        memory_after_request = runtime.mem_info()
        stats_after = runtime.stats()
    finally:
        setattr(runtime._library, native_name, native_function)
        runtime.free(output_pointer)
        runtime.free(input_pointer)
    runtime.synchronize()
    memory_after_free = runtime.mem_info()
    request_growth = max(
        0,
        int(memory_before_request["free_bytes"])
        - int(memory_after_request["free_bytes"]),
    )
    persistent_growth = max(
        0,
        int(memory_before["free_bytes"])
        - int(memory_after_free["free_bytes"]),
    )
    passed = (
        error == EXPECTED_REJECTION
        and native_call_attempts == 0
        and stats_before == stats_after
        and request_growth <= MEMORY_STABILITY_TOLERANCE_BYTES
        and persistent_growth <= MEMORY_STABILITY_TOLERANCE_BYTES
    )
    return {
        "requested_batch": 4,
        "certified_max": runtime.expert_max_certified_batch,
        "error": error,
        "expected_error": EXPECTED_REJECTION,
        "native_call_attempts": native_call_attempts,
        "native_sentinel_armed": True,
        "cuda_synchronize_after_rejection": "PASS",
        "stats_before": stats_before,
        "stats_after": stats_after,
        "stats_unchanged": stats_before == stats_after,
        "memory": {
            "explicit_caller_io_bytes": source.nbytes * 2,
            "free_before_bytes": memory_before["free_bytes"],
            "free_before_request_bytes": memory_before_request["free_bytes"],
            "free_after_request_bytes": memory_after_request["free_bytes"],
            "free_after_free_bytes": memory_after_free["free_bytes"],
            "request_growth_bytes": request_growth,
            "persistent_growth_bytes": persistent_growth,
            "stability_tolerance_bytes": MEMORY_STABILITY_TOLERANCE_BYTES,
        },
        "pass": passed,
    }


def _comparison(
    candidate: dict[str, tuple[dict[str, Any], np.ndarray]],
    retained: dict[str, tuple[dict[str, Any], np.ndarray]],
) -> dict[str, Any]:
    output_rows = np.concatenate(
        (candidate["batch1_pre"][1], candidate["batch1_post"][1]), axis=0
    )
    row_semantics = _numerical_metrics(candidate["batch2"][1], output_rows)
    candidate_b1_p50 = float(
        np.mean(
            [
                candidate["batch1_pre"][0]["device"]["p50_ms"],
                candidate["batch1_post"][0]["device"]["p50_ms"],
            ]
        )
    )
    retained_b1_p50 = float(
        np.mean(
            [
                retained["batch1_pre"][0]["device"]["p50_ms"],
                retained["batch1_post"][0]["device"]["p50_ms"],
            ]
        )
    )
    batch1_ratio = candidate_b1_p50 / retained_b1_p50
    batch2_ratio = (
        float(candidate["batch2"][0]["device"]["p50_ms"])
        / float(retained["batch2"][0]["device"]["p50_ms"])
    )
    exact = {
        key: bool(np.array_equal(candidate[key][1], retained[key][1]))
        for key in candidate
    }
    return {
        "candidate_batch2_vs_candidate_serial_rows": {
            **row_semantics,
            "bit_exact": bool(np.array_equal(candidate["batch2"][1], output_rows)),
        },
        "candidate_vs_retained_bit_exact": exact,
        "performance": {
            "candidate_batch1_device_p50_ms": candidate_b1_p50,
            "retained_batch1_device_p50_ms": retained_b1_p50,
            "candidate_over_retained_batch1_ratio": batch1_ratio,
            "candidate_batch2_device_p50_ms": candidate["batch2"][0]["device"][
                "p50_ms"
            ],
            "retained_batch2_device_p50_ms": retained["batch2"][0]["device"][
                "p50_ms"
            ],
            "candidate_over_retained_batch2_ratio": batch2_ratio,
            "maximum_allowed_regression_ratio": PERFORMANCE_REGRESSION_LIMIT,
        },
        "pass": (
            bool(np.array_equal(candidate["batch2"][1], output_rows))
            and all(exact.values())
            and batch1_ratio <= PERFORMANCE_REGRESSION_LIMIT
            and batch2_ratio <= PERFORMANCE_REGRESSION_LIMIT
        ),
    }


def _serializable_modes(
    modes: dict[str, tuple[dict[str, Any], np.ndarray]],
) -> dict[str, dict[str, Any]]:
    return {name: record for name, (record, _output) in modes.items()}


def validate_failclosed_batch_candidate(
    checkpoint: Path,
    candidate_library: Path,
    retained_library: Path,
    candidate_receipt_path: Path,
    output_path: Path,
    *,
    device: int = 0,
    layer: int = 89,
    expert_id: int = 803,
    warmup: int = 20,
    iterations: int = 100,
) -> dict[str, Any]:
    """Run the bounded H014-027w sequence and persist evidence after each step."""
    if warmup < 3 or iterations < 20:
        raise ValueError("H014-027w requires >=3 warmups and >=20 retained calls")
    candidate_receipt = json.loads(candidate_receipt_path.read_text(encoding="utf-8"))
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "cycle_id": "H014-027w",
        "hypothesis": (
            "The max_certified_batch=2 contract preserves real expert batch 1/2, "
            "rejects batch 4 before the native CUDA boundary, and leaves the GPU "
            "healthy for a subsequent real batch-1 fixture."
        ),
        "implementation": (
            "Quarantined candidate; production API sequence 1 -> 2 -> rejected 4 -> 1; "
            "a native-call sentinel makes the rejection test physically incapable of "
            "launching batch 4; retained binary executed separately at only batch 1/2."
        ),
        "configuration": {
            "device": device,
            "layer": layer,
            "expert_id": expert_id,
            "warmup_calls": warmup,
            "retained_calls": iterations,
            "batch_sequence": [1, 2, 4, 1],
            "unsafe_multi_size_sweep_run": False,
            "cpu_mathematical_fallbacks": 0,
        },
        "sources": {
            "checkpoint": str(checkpoint.resolve()),
            "candidate_library": str(candidate_library.resolve()),
            "retained_library": str(retained_library.resolve()),
            "candidate_build_receipt": str(candidate_receipt_path.resolve()),
            "candidate_build_receipt_sha256": _sha256_file(candidate_receipt_path),
        },
        "source_integrity": _source_integrity(candidate_receipt),
        "progress": [],
        "status": "RUNNING",
    }
    invalid_attempt_path = output_path.with_name(
        "h014-027w-invalid-harness-handle-order.json"
    )
    if invalid_attempt_path.is_file():
        receipt["invalidated_attempt"] = {
            "path": str(invalid_attempt_path.resolve()),
            "sha256": _sha256_file(invalid_attempt_path),
            "result": "INVALID_FIXTURE_NOT_CANDIDATE_EVIDENCE",
            "cause": (
                "Harness uploaded handles as gate/down/up instead of the certified "
                "gate/up/down ABI; native dimension validation rejected batch 1."
            ),
            "post_cleanup_nvidia_smi": _health_snapshot(device),
        }

    def retain(phase: str, **evidence: Any) -> None:
        receipt["progress"].append({"phase": phase, **evidence})
        _atomic_json(output_path, receipt)

    candidate_runtime: _CudaRuntime | None = None
    retained_runtime: _CudaRuntime | None = None
    try:
        if not receipt["source_integrity"]["all_match"]:
            raise ValueError("candidate build/source integrity mismatch")
        if _sha256_file(candidate_library) != candidate_receipt["sources"][
            "candidate_binary"
        ]["sha256"]:
            raise ValueError("candidate library does not match quarantined build receipt")
        if _sha256_file(retained_library) != candidate_receipt["sources"][
            "deployment_binary_unchanged"
        ]["sha256"]:
            raise ValueError("retained library does not match quarantined build receipt")

        pre_health = _health_snapshot(device)
        receipt["gpu_health_before"] = pre_health
        receipt["device_identity_before"] = _device_identity(device)
        retain("gpu_identity_and_nvidia_smi_before", status=pre_health["status"])
        if pre_health["status"] != "MEASURED":
            raise RuntimeError("nvidia-smi did not report a healthy GPU before CUDA init")

        real_expert = _load_real_expert(checkpoint, layer, expert_id)
        expert_source_bytes = sum(
            tensor.packed.nbytes + tensor.scales.nbytes
            for tensor in (real_expert.gate, real_expert.down, real_expert.up)
        )
        rng = np.random.default_rng(140270803)
        rows = np.ascontiguousarray(
            rng.normal(0.0, 0.25, size=(4, 3584)), dtype=np.float32
        )
        receipt["fixture"] = {
            "real_checkpoint_weights": True,
            "expert_source_shards": real_expert.source_shards,
            "expert_source_bytes": expert_source_bytes,
            "activation_fingerprint": _array_fingerprint(rows),
            "activation_shape": list(rows.shape),
        }

        candidate_runtime = _CudaRuntime(candidate_library, device)
        candidate_handles = _upload_expert(candidate_runtime, real_expert)
        receipt["candidate"] = {
            "binary_sha256": candidate_runtime.sha256,
            "max_certified_batch": candidate_runtime.expert_max_certified_batch,
            "capability_source": candidate_runtime.expert_batch_capability_source,
            "minimum_compute_capability": candidate_runtime.binary_min_compute_capability,
            "forward_ptx": candidate_runtime.binary_has_forward_ptx,
            "capability_negotiation": candidate_runtime.capability_negotiation,
        }
        retain(
            "candidate_cuda_initialization",
            max_certified_batch=candidate_runtime.expert_max_certified_batch,
            capability_source=candidate_runtime.expert_batch_capability_source,
        )
        if (
            candidate_runtime.expert_max_certified_batch != 2
            or candidate_runtime.expert_batch_capability_source != "native_export"
        ):
            raise RuntimeError("candidate did not expose native max_certified_batch=2")

        candidate_modes: dict[str, tuple[dict[str, Any], np.ndarray]] = {}
        candidate_modes["batch1_pre"] = _run_mode(
            candidate_runtime,
            candidate_handles,
            rows[0:1],
            warmup=warmup,
            iterations=iterations,
        )
        receipt["candidate"]["modes"] = _serializable_modes(candidate_modes)
        retain(
            "candidate_real_expert_batch1",
            output_fingerprint=candidate_modes["batch1_pre"][0]["output_fingerprint"],
        )

        candidate_modes["batch2"] = _run_mode(
            candidate_runtime,
            candidate_handles,
            rows[0:2],
            warmup=warmup,
            iterations=iterations,
        )
        receipt["candidate"]["modes"] = _serializable_modes(candidate_modes)
        retain(
            "candidate_real_expert_batch2",
            output_fingerprint=candidate_modes["batch2"][0]["output_fingerprint"],
        )

        rejection = _guarded_batch4_rejection(
            candidate_runtime, candidate_handles, rows
        )
        receipt["candidate"]["batch4_rejection"] = rejection
        retain(
            "candidate_batch4_rejected_before_native_boundary",
            passed=rejection["pass"],
            native_call_attempts=rejection["native_call_attempts"],
        )
        if not rejection["pass"]:
            raise RuntimeError("batch-4 fail-closed gate did not pass")

        candidate_modes["batch1_post"] = _run_mode(
            candidate_runtime,
            candidate_handles,
            rows[1:2],
            warmup=warmup,
            iterations=iterations,
        )
        receipt["candidate"]["modes"] = _serializable_modes(candidate_modes)
        candidate_runtime.synchronize()
        post_rejection_health = _health_snapshot(device)
        receipt["gpu_health_after_rejection_and_batch1"] = post_rejection_health
        retain(
            "candidate_post_rejection_batch1_and_health",
            output_fingerprint=candidate_modes["batch1_post"][0]["output_fingerprint"],
            nvidia_smi=post_rejection_health["status"],
        )
        candidate_runtime.close()
        candidate_runtime = None

        retained_runtime = _CudaRuntime(retained_library, device)
        retained_handles = _upload_expert(retained_runtime, real_expert)
        retained_modes = {
            "batch1_pre": _run_mode(
                retained_runtime,
                retained_handles,
                rows[0:1],
                warmup=warmup,
                iterations=iterations,
            ),
            "batch2": _run_mode(
                retained_runtime,
                retained_handles,
                rows[0:2],
                warmup=warmup,
                iterations=iterations,
            ),
            "batch1_post": _run_mode(
                retained_runtime,
                retained_handles,
                rows[1:2],
                warmup=warmup,
                iterations=iterations,
            ),
        }
        receipt["retained_binary_control"] = {
            "binary_sha256": retained_runtime.sha256,
            "max_certified_batch": retained_runtime.expert_max_certified_batch,
            "capability_source": retained_runtime.expert_batch_capability_source,
            "modes": _serializable_modes(retained_modes),
        }
        comparison = _comparison(candidate_modes, retained_modes)
        receipt["comparison"] = comparison
        retain("retained_binary_batch1_batch2_comparison", passed=comparison["pass"])
        retained_runtime.close()
        retained_runtime = None

        final_health = _health_snapshot(device)
        receipt["gpu_health_final"] = final_health
        uuids = {
            snapshot.get("uuid")
            for snapshot in (pre_health, post_rejection_health, final_health)
        }
        health_pass = (
            all(
                snapshot["status"] == "MEASURED"
                for snapshot in (pre_health, post_rejection_health, final_health)
            )
            and len(uuids) == 1
        )
        receipt["health_gate"] = {
            "same_gpu_uuid": len(uuids) == 1,
            "observed_uuids": sorted(str(value) for value in uuids),
            "pass": health_pass,
        }
        receipt["gates"] = {
            "source_integrity": receipt["source_integrity"]["all_match"],
            "native_capability_query": True,
            "real_batch1": candidate_modes["batch1_pre"][0]["output_finite"],
            "real_batch2": candidate_modes["batch2"][0]["output_finite"],
            "batch4_prelaunch_rejection": rejection["pass"],
            "post_rejection_real_batch1": candidate_modes["batch1_post"][0][
                "output_finite"
            ],
            "retained_binary_equivalence": comparison["pass"],
            "gpu_health": health_pass,
        }
        all_pass = all(receipt["gates"].values())
        receipt["status"] = "PASS" if all_pass else "FAIL"
        receipt["hypothesis_supported"] = all_pass
        receipt["inspection"] = (
            "The bounded candidate path remained within batch 1/2, the armed sentinel "
            "observed zero native batch-4 calls, and the post-rejection real batch-1 "
            "fixture plus nvidia-smi determine whether CUDA remained healthy."
        )
        receipt["bottleneck"] = (
            "Fail-closed safety is resolved; existing batch execution still lacks "
            "cross-row weight reuse."
            if all_pass
            else "One or more H014-027w safety/preservation gates failed."
        )
        receipt["decision"] = (
            "PROMOTE_CANDIDATE_PENDING_P0_P1_NATIVE_REGRESSIONS"
            if all_pass
            else "KEEP_CANDIDATE_QUARANTINED"
        )
        receipt["redesign"] = (
            "After exact-binary P0/P1 regression certification, H014-028 may test a "
            "new row-cooperative kernel incrementally; do not reopen the old sweep."
        )
        retain("final_h014_027w_gate", status=receipt["status"])
    except Exception as exc:
        receipt["status"] = "FAIL"
        receipt["hypothesis_supported"] = False
        receipt["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        receipt["decision"] = "KEEP_CANDIDATE_QUARANTINED"
        receipt["redesign"] = "Inspect the retained failure before any further GPU work."
        try:
            receipt["gpu_health_on_failure"] = _health_snapshot(device)
        except Exception as health_exc:  # pragma: no cover - diagnostic best effort
            receipt["gpu_health_on_failure"] = {
                "status": "QUERY_FAILED",
                "error": str(health_exc),
            }
        _atomic_json(output_path, receipt)
    finally:
        for runtime in (retained_runtime, candidate_runtime):
            if runtime is not None:
                with suppress(Exception):  # pragma: no cover - diagnostic best effort
                    runtime.close()

    result = dict(receipt)
    result["output_path"] = str(output_path.resolve())
    result["output_sha256"] = _sha256_file(output_path)
    return result
