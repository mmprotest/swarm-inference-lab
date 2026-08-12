"""One-size-at-a-time Kimi expert batch certification for H014-028."""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import traceback
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.experiments.experiment_014.batch_safety import (
    MEMORY_STABILITY_TOLERANCE_BYTES,
    _health_snapshot,
    _run_mode,
    _serializable_modes,
    _upload_expert,
)
from swarm_inference.experiments.experiment_014.cuda import (
    KimiCudaError,
    _array_fingerprint,
    _CudaRuntime,
    _device_identity,
    _load_real_expert,
    _numerical_metrics,
    _sha256_file,
)

SCHEMA_VERSION = "experiment-014-k3-incremental-row-cooperative-batch-v1"
PRODUCTION_BATCHES = (1, 2, 4, 8, 16)
MAXIMUM_PRIOR_SIZE_REGRESSION_RATIO = 1.05


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _expected_sizes(target_batch: int) -> tuple[int, ...]:
    if target_batch not in PRODUCTION_BATCHES[2:]:
        raise ValueError("target batch must be one of 4, 8, or 16")
    return tuple(batch for batch in PRODUCTION_BATCHES if batch <= target_batch)


def _binary_inspection(
    binary: Path, cuobjdump: Path, target_batch: int
) -> dict[str, Any]:
    commands = {
        "elf": ("--list-elf",),
        "ptx": ("--list-ptx",),
        "sass": ("--dump-sass",),
        "ptx_text": ("--dump-ptx",),
    }
    outputs: dict[str, str] = {}
    returncodes: dict[str, int] = {}
    for name, arguments in commands.items():
        completed = subprocess.run(
            [str(cuobjdump), *arguments, str(binary)],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        outputs[name] = completed.stdout + completed.stderr
        returncodes[name] = completed.returncode
    template_marker = f"ILi{target_batch}E"
    target_pair_sass = (
        "mxfp4_matmul_pair_rows" in outputs["sass"]
        and template_marker in outputs["sass"]
    )
    target_single_sass = (
        "mxfp4_matmul_rows" in outputs["sass"]
        and template_marker in outputs["sass"]
    )
    target_pair_ptx = (
        "mxfp4_matmul_pair_rows" in outputs["ptx_text"]
        and template_marker in outputs["ptx_text"]
    )
    target_single_ptx = (
        "mxfp4_matmul_rows" in outputs["ptx_text"]
        and template_marker in outputs["ptx_text"]
    )
    result = {
        "cuobjdump": str(cuobjdump.resolve()),
        "returncodes": returncodes,
        "sm86_cubin": "sm_86.cubin" in outputs["elf"],
        "compute86_ptx": "sm_86.ptx" in outputs["ptx"],
        "target_template_marker": template_marker,
        "target_pair": {
            "sm86_sass": target_pair_sass,
            "compute86_ptx": target_pair_ptx,
        },
        "target_single": {
            "sm86_sass": target_single_sass,
            "compute86_ptx": target_single_ptx,
        },
    }
    result["pass"] = (
        all(code == 0 for code in returncodes.values())
        and result["sm86_cubin"]
        and result["compute86_ptx"]
        and target_pair_sass
        and target_single_sass
        and target_pair_ptx
        and target_single_ptx
    )
    return result


def _guarded_unsupported_size(
    runtime: _CudaRuntime,
    handles: tuple[Any, Any, Any],
    requested_batch: int,
) -> dict[str, Any]:
    native_name = "coli_cuda_kimi_expert_mlp_dev"
    native_function = getattr(runtime._library, native_name)
    native_attempts = 0

    def sentinel(*_args: Any) -> int:
        nonlocal native_attempts
        native_attempts += 1
        raise AssertionError("unsupported batch reached the native CUDA boundary")

    error: str | None = None
    try:
        setattr(runtime._library, native_name, sentinel)
        runtime.execute_resident(
            handles,
            ctypes.c_void_p(),
            ctypes.c_void_p(),
            requested_batch,
        )
    except KimiCudaError as exc:
        error = str(exc)
    finally:
        setattr(runtime._library, native_name, native_function)
    passed = (
        native_attempts == 0
        and error is not None
        and f"requested={requested_batch}" in error
        and f"certified_max={runtime.expert_max_certified_batch}" in error
        and str(runtime.expert_supported_batches) in error
    )
    return {
        "requested_batch": requested_batch,
        "error": error,
        "native_call_attempts": native_attempts,
        "native_sentinel_armed": True,
        "pass": passed,
    }


def _memory_stable(modes: dict[str, tuple[dict[str, Any], np.ndarray]]) -> bool:
    return all(
        int(record["memory"]["persistent_growth_bytes"])
        <= MEMORY_STABILITY_TOLERANCE_BYTES
        for record, _output in modes.values()
    )


def benchmark_incremental_row_cooperative_batch(
    checkpoint: Path,
    candidate_library: Path,
    prior_library: Path,
    prior_receipt_path: Path,
    cuda_source: Path,
    cuda_header: Path,
    cuobjdump: Path,
    output_path: Path,
    *,
    target_batch: int,
    minimum_target_speedup: float,
    cycle_id: str,
    device: int = 0,
    layer: int = 89,
    expert_id: int = 803,
    warmup: int = 50,
    iterations: int = 300,
) -> dict[str, Any]:
    """Attempt one new batch size only after all smaller retained sizes pass."""
    expected_sizes = _expected_sizes(target_batch)
    prior_sizes = expected_sizes[:-1]
    if minimum_target_speedup <= 1.0:
        raise ValueError("minimum target speedup must be >1")
    if warmup < 20 or iterations < 100:
        raise ValueError("incremental certification requires >=20/100 warm/retained")
    paths = {
        "checkpoint": checkpoint.resolve(),
        "candidate_library": candidate_library.resolve(),
        "prior_library": prior_library.resolve(),
        "prior_receipt": prior_receipt_path.resolve(),
        "cuda_source": cuda_source.resolve(),
        "cuda_header": cuda_header.resolve(),
        "cuobjdump": cuobjdump.resolve(),
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
                f"The exact row-cooperative batch-{target_batch} path yields at "
                f"least {minimum_target_speedup:.2f}x aggregate throughput versus "
                f"{target_batch} batch-1 calls, remains bit exact, preserves every "
                "previously certified size within 5%, and leaves CUDA, VRAM, and "
                "the physical GPU healthy for a known-safe batch-1 fixture."
            ),
            "minimum_target_speedup": minimum_target_speedup,
            "maximum_prior_size_regression_ratio": (
                MAXIMUM_PRIOR_SIZE_REGRESSION_RATIO
            ),
        },
        "implementation": {
            "target_template_rows": target_batch,
            "expected_exact_supported_sizes": list(expected_sizes),
            "newly_attempted_batch": target_batch,
            "larger_batch_executed": False,
            "deployed_binary_mutated": False,
            "one_size_per_cycle": True,
        },
        "configuration": {
            "device": device,
            "layer": layer,
            "expert_id": expert_id,
            "warmup_calls": warmup,
            "retained_calls": iterations,
            "candidate_batch_order": list(expected_sizes),
            "post_target_safe_fixture": 1,
            "timed_h2d_bytes": 0,
            "timed_d2h_bytes": 0,
            "cpu_mathematical_fallbacks": 0,
        },
        "sources": {
            name: {
                "path": str(path),
                "sha256": _sha256_file(path) if path.is_file() else None,
            }
            for name, path in paths.items()
        },
        "progress": [],
    }

    def retain(phase: str, **evidence: Any) -> None:
        receipt["progress"].append({"phase": phase, **evidence})
        _atomic_json(output_path, receipt)

    candidate_runtime: _CudaRuntime | None = None
    prior_runtime: _CudaRuntime | None = None
    try:
        prior_receipt = json.loads(
            paths["prior_receipt"].read_text(encoding="utf-8")
        )
        receipt["prior_evidence"] = {
            "cycle_id": prior_receipt.get("cycle_id"),
            "status": prior_receipt.get("status"),
            "hypothesis_supported": prior_receipt.get("hypothesis_supported"),
        }
        if prior_receipt.get("status") != "PASS" or not prior_receipt.get(
            "hypothesis_supported"
        ):
            raise ValueError("prior incremental batch evidence is not passing")

        receipt["artifact_inspection"] = _binary_inspection(
            paths["candidate_library"], paths["cuobjdump"], target_batch
        )
        retain(
            "candidate_binary_inspection",
            passed=receipt["artifact_inspection"]["pass"],
        )
        if not receipt["artifact_inspection"]["pass"]:
            raise RuntimeError("target template missing from sm86 SASS or compute86 PTX")

        health_before = _health_snapshot(device)
        receipt["gpu_health_before"] = health_before
        receipt["device_identity"] = _device_identity(device)
        retain("gpu_health_before", status=health_before["status"])
        if health_before["status"] != "MEASURED":
            raise RuntimeError("nvidia-smi unavailable before CUDA initialization")

        real_expert = _load_real_expert(checkpoint, layer, expert_id)
        rng = np.random.default_rng(140280803 + target_batch)
        rows = np.ascontiguousarray(
            rng.normal(0.0, 0.25, size=(target_batch, 3584)), dtype=np.float32
        )
        receipt["fixture"] = {
            "real_checkpoint_weights": True,
            "expert_source_shards": real_expert.source_shards,
            "expert_source_bytes": sum(
                tensor.packed.nbytes + tensor.scales.nbytes
                for tensor in (real_expert.gate, real_expert.up, real_expert.down)
            ),
            "activation_shape": list(rows.shape),
            "activation_fingerprint": _array_fingerprint(rows),
        }

        candidate_runtime = _CudaRuntime(paths["candidate_library"], device)
        candidate_runtime.set_telemetry("minimal")
        actual_sizes = candidate_runtime.expert_supported_batches
        receipt["candidate"] = {
            "binary_sha256": candidate_runtime.sha256,
            "max_certified_batch": candidate_runtime.expert_max_certified_batch,
            "supported_batches": list(actual_sizes),
            "batch_set_capability_source": (
                candidate_runtime.expert_batch_set_capability_source
            ),
            "cuda_error_state_query": (
                candidate_runtime.cuda_error_state_query is not None
            ),
            "modes": {},
        }
        retain("candidate_cuda_initialization", supported_batches=list(actual_sizes))
        if actual_sizes != expected_sizes:
            raise RuntimeError(
                f"candidate exposes {actual_sizes}, expected exact {expected_sizes}"
            )
        if candidate_runtime.cuda_error_state_query is None:
            raise RuntimeError("candidate lacks explicit CUDA error-state query")

        handles = _upload_expert(candidate_runtime, real_expert)
        candidate_modes: dict[str, tuple[dict[str, Any], np.ndarray]] = {}
        for batch in prior_sizes:
            name = f"batch{batch}"
            candidate_modes[name] = _run_mode(
                candidate_runtime,
                handles,
                rows[:batch],
                warmup=warmup,
                iterations=iterations,
            )
            receipt["candidate"]["modes"] = _serializable_modes(candidate_modes)
            retain(f"candidate_previously_certified_batch{batch}")

        target_name = f"batch{target_batch}"
        retain(
            f"armed_new_batch{target_batch}",
            prior_sizes_completed=list(prior_sizes),
            gpu_health=_health_snapshot(device)["status"],
        )
        try:
            candidate_modes[target_name] = _run_mode(
                candidate_runtime,
                handles,
                rows,
                warmup=warmup,
                iterations=iterations,
            )
        except Exception:
            receipt["target_failure_recovery"] = {
                "synchronize": "NOT_ATTEMPTED",
                "error_state_ok": None,
                "known_safe_fixture": "SKIPPED_ON_DEGRADATION",
            }
            with suppress(Exception):
                candidate_runtime.synchronize()
                receipt["target_failure_recovery"]["synchronize"] = "PASS"
            with suppress(Exception):
                receipt["target_failure_recovery"]["error_state_ok"] = (
                    candidate_runtime.error_state_ok()
                )
            receipt["target_failure_recovery"]["nvidia_smi"] = _health_snapshot(
                device
            )
            raise
        receipt["candidate"]["modes"] = _serializable_modes(candidate_modes)
        candidate_runtime.synchronize()
        error_state_after_target = candidate_runtime.error_state_ok()
        health_after_target = _health_snapshot(device)
        receipt["post_target_checks"] = {
            "cuda_synchronize": "PASS",
            "cuda_error_state_ok": error_state_after_target,
            "nvidia_smi": health_after_target,
            "free_vram_bytes": candidate_runtime.mem_info()["free_bytes"],
        }
        retain(
            f"new_batch{target_batch}_completed_and_checked",
            error_state_ok=error_state_after_target,
            nvidia_smi=health_after_target["status"],
        )
        if not error_state_after_target or health_after_target["status"] != "MEASURED":
            raise RuntimeError("CUDA or physical GPU degraded after new batch")

        candidate_modes["batch1_post_target"] = _run_mode(
            candidate_runtime,
            handles,
            rows[:1],
            warmup=warmup,
            iterations=iterations,
        )
        candidate_runtime.synchronize()
        post_safe_error_state = candidate_runtime.error_state_ok()
        health_after_safe = _health_snapshot(device)
        receipt["candidate"]["modes"] = _serializable_modes(candidate_modes)
        receipt["post_target_safe_fixture"] = {
            "batch": 1,
            "cuda_synchronize": "PASS",
            "cuda_error_state_ok": post_safe_error_state,
            "nvidia_smi": health_after_safe,
            "free_vram_bytes": candidate_runtime.mem_info()["free_bytes"],
        }
        retain(
            "known_safe_batch1_after_target",
            error_state_ok=post_safe_error_state,
            nvidia_smi=health_after_safe["status"],
        )

        for row_index in range(target_batch):
            name = f"serial_row{row_index}"
            if row_index == 0:
                candidate_modes[name] = candidate_modes["batch1"]
            else:
                candidate_modes[name] = _run_mode(
                    candidate_runtime,
                    handles,
                    rows[row_index : row_index + 1],
                    warmup=warmup,
                    iterations=iterations,
                )
        receipt["candidate"]["modes"] = _serializable_modes(candidate_modes)

        unsupported_checks = [
            _guarded_unsupported_size(candidate_runtime, handles, 3),
            _guarded_unsupported_size(
                candidate_runtime,
                handles,
                next(batch for batch in PRODUCTION_BATCHES if batch > target_batch)
                if target_batch < 16
                else 32,
            ),
        ]
        receipt["unsupported_size_preflight"] = unsupported_checks
        retain(
            "unsupported_sizes_rejected_before_native_boundary",
            passed=all(row["pass"] for row in unsupported_checks),
        )
        candidate_runtime.close()
        candidate_runtime = None

        prior_runtime = _CudaRuntime(paths["prior_library"], device)
        prior_runtime.set_telemetry("minimal")
        prior_handles = _upload_expert(prior_runtime, real_expert)
        prior_control: dict[str, tuple[dict[str, Any], np.ndarray]] = {}
        for batch in prior_sizes:
            prior_control[f"batch{batch}"] = _run_mode(
                prior_runtime,
                prior_handles,
                rows[:batch],
                warmup=warmup,
                iterations=iterations,
            )
        receipt["prior_binary_control"] = {
            "binary_sha256": prior_runtime.sha256,
            "supported_batches": list(prior_runtime.expert_supported_batches),
            "modes": _serializable_modes(prior_control),
        }
        prior_runtime.close()
        prior_runtime = None
        retain("prior_binary_safe_size_controls")

        serial_output = np.concatenate(
            [
                candidate_modes[f"serial_row{index}"][1]
                for index in range(target_batch)
            ],
            axis=0,
        )
        target_output = candidate_modes[target_name][1]
        target_metrics = _numerical_metrics(target_output, serial_output)
        serial_p50_ms = sum(
            float(candidate_modes[f"serial_row{index}"][0]["device"]["p50_ms"])
            for index in range(target_batch)
        )
        target_p50_ms = float(
            candidate_modes[target_name][0]["device"]["p50_ms"]
        )
        target_speedup = serial_p50_ms / target_p50_ms
        prior_comparisons: dict[str, Any] = {}
        for batch in prior_sizes:
            name = f"batch{batch}"
            candidate_p50 = float(candidate_modes[name][0]["device"]["p50_ms"])
            prior_p50 = float(prior_control[name][0]["device"]["p50_ms"])
            prior_comparisons[name] = {
                "candidate_device_p50_ms": candidate_p50,
                "prior_device_p50_ms": prior_p50,
                "ratio": candidate_p50 / prior_p50,
                "bit_exact": bool(
                    np.array_equal(candidate_modes[name][1], prior_control[name][1])
                ),
            }
        target_stats = candidate_modes[target_name][0]
        receipt["comparison"] = {
            "target_vs_serial": {
                **target_metrics,
                "bit_exact": bool(np.array_equal(target_output, serial_output)),
            },
            "post_target_batch1_bit_exact_to_pre": bool(
                np.array_equal(
                    candidate_modes["batch1_post_target"][1],
                    candidate_modes["batch1"][1],
                )
            ),
            "prior_size_controls": prior_comparisons,
            "performance": {
                "serial_device_p50_sum_ms": serial_p50_ms,
                "target_device_p50_ms": target_p50_ms,
                "target_device_p95_ms": target_stats["device"]["p95_ms"],
                "target_device_p99_ms": target_stats["device"]["p99_ms"],
                "target_wall_p50_ms": target_stats["wall"]["p50_ms"],
                "target_wall_p95_ms": target_stats["wall"]["p95_ms"],
                "target_wall_p99_ms": target_stats["wall"]["p99_ms"],
                "aggregate_rows_per_second_p50": 1000.0 * target_batch / target_p50_ms,
                "per_row_service_ms_p50": target_p50_ms / target_batch,
                "target_pair_throughput_speedup": target_speedup,
                "minimum_preregistered_speedup": minimum_target_speedup,
            },
        }

        health_final = _health_snapshot(device)
        receipt["gpu_health_final"] = health_final
        uuids = {
            snapshot.get("uuid")
            for snapshot in (
                health_before,
                health_after_target,
                health_after_safe,
                health_final,
            )
        }
        health_pass = (
            all(
                snapshot["status"] == "MEASURED"
                for snapshot in (
                    health_before,
                    health_after_target,
                    health_after_safe,
                    health_final,
                )
            )
            and len(uuids) == 1
        )
        gates = {
            "exact_supported_batch_set": tuple(
                receipt["candidate"]["supported_batches"]
            )
            == expected_sizes,
            "target_bit_exact_to_serial": bool(
                np.array_equal(target_output, serial_output)
            ),
            "post_target_safe_fixture_bit_exact": receipt["comparison"][
                "post_target_batch1_bit_exact_to_pre"
            ],
            "prior_sizes_bit_exact": all(
                row["bit_exact"] for row in prior_comparisons.values()
            ),
            "prior_sizes_within_5_percent": all(
                row["ratio"] <= MAXIMUM_PRIOR_SIZE_REGRESSION_RATIO
                for row in prior_comparisons.values()
            ),
            "target_speedup_gate": target_speedup >= minimum_target_speedup,
            "candidate_memory_stable": _memory_stable(candidate_modes),
            "prior_memory_stable": _memory_stable(prior_control),
            "post_target_cuda_error_state": error_state_after_target,
            "post_safe_cuda_error_state": post_safe_error_state,
            "unsupported_sizes_preflight": all(
                row["pass"] for row in unsupported_checks
            ),
            "gpu_health": health_pass,
            "artifact_contract": receipt["artifact_inspection"]["pass"],
        }
        receipt["gates"] = gates
        safety_gate_names = (
            "exact_supported_batch_set",
            "target_bit_exact_to_serial",
            "post_target_safe_fixture_bit_exact",
            "prior_sizes_bit_exact",
            "candidate_memory_stable",
            "prior_memory_stable",
            "post_target_cuda_error_state",
            "post_safe_cuda_error_state",
            "unsupported_sizes_preflight",
            "gpu_health",
            "artifact_contract",
        )
        benchmark_valid = all(gates[name] for name in safety_gate_names)
        hypothesis_supported = benchmark_valid and all(gates.values())
        receipt["status"] = "PASS" if benchmark_valid else "FAIL"
        receipt["hypothesis_supported"] = hypothesis_supported
        receipt["hypothesis_result"] = (
            "SUPPORTED" if hypothesis_supported else "FALSIFIED"
        )
        receipt["inspection"] = (
            f"Batch {target_batch} delivered {target_speedup:.6f}x aggregate "
            f"throughput at device p50 {target_p50_ms:.6f} ms; p95/p99, exact "
            "outputs, prior-size drift, sticky error state, safe recovery, memory, "
            "and nvidia-smi were retained independently."
        )
        receipt["bottleneck"] = (
            "The target-size row-cooperative projection still retains useful "
            "cross-row weight reuse."
            if hypothesis_supported
            else (
                "The new size is numerically characterized, but its aggregate "
                "speedup or prior-size preservation is below the preregistered gate."
            )
        )
        receipt["decision"] = (
            f"RETAIN_AND_BUILD_SEPARATE_BATCH{target_batch * 2}_CANDIDATE"
            if hypothesis_supported and target_batch < 16
            else (
                "RETAIN_BATCH16_AS_LARGEST_LOCAL_CANDIDATE"
                if hypothesis_supported
                else "DO_NOT_EXPOSE_THE_NEXT_BATCH_SIZE"
            )
        )
        receipt["redesign"] = (
            f"Instantiate exactly batch {target_batch * 2} and repeat the same "
            "one-size health protocol."
            if hypothesis_supported and target_batch < 16
            else (
                "Move to complete-stage batching and retain this ceiling."
                if hypothesis_supported
                else "Inspect kernel resources and tail behavior before redesign."
            )
        )
        receipt["effective_weight_reuse"] = {
            "architectural_rows_per_weight_traversal": target_batch,
            "measured_dram_bytes": None,
            "qualification": "source-level reuse; profiler counters remain separate",
        }
        retain("final_incremental_batch_gate", hypothesis=receipt["hypothesis_result"])
    except Exception as exc:
        receipt["status"] = "FAIL"
        receipt["hypothesis_supported"] = False
        receipt["hypothesis_result"] = "INVALID_OR_UNSAFE_BENCHMARK"
        receipt["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        receipt["decision"] = "STOP_INCREMENTAL_BATCH_CERTIFICATION"
        receipt["redesign"] = "Inspect retained failure and physical GPU health."
        with suppress(Exception):
            receipt["gpu_health_on_failure"] = _health_snapshot(device)
        _atomic_json(output_path, receipt)
    finally:
        for runtime in (prior_runtime, candidate_runtime):
            if runtime is not None:
                with suppress(Exception):
                    runtime.close()

    result = dict(receipt)
    result["output_path"] = str(output_path.resolve())
    result["output_sha256"] = _sha256_file(output_path)
    return result
