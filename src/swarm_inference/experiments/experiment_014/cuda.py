"""Real-weight CUDA certification fixtures for Experiment 014."""

from __future__ import annotations

import ctypes
import hashlib
import json
import struct
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.execution.kimi_cuda_runtime import (
    SCHEMA_VERSION,
    KimiCudaError,
    _array_fingerprint,
    _array_fingerprint_streaming,
    _CudaRuntime,
    _dequantize_grouped_int4,
    _grouped_int4_matvec,
    _GroupedInt4Tensor,
    _kda_core_reference,
    _load_real_expert,
    _load_safetensor_f32,
    _load_safetensor_raw_bf16,
    _mla_stage_reference,
    _numerical_metrics,
    _percentiles,
    _quantize_bf16_rows_int8,
    _quantize_grouped_int4,
    _QuantizedInt8Tensor,
    _rmsnorm_reference,
    _sha256_file,
)
from swarm_inference.experiments.experiment_010.kimi import NativeMXFP4Runtime


def _device_identity(device: int) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        f"--id={device}",
        "--query-gpu=name,uuid,compute_cap,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=20)
    if completed.returncode:
        return {"query": command, "status": "UNAVAILABLE", "stderr": completed.stderr.strip()}
    fields = [field.strip() for field in completed.stdout.strip().split(",")]
    if len(fields) != 5:
        return {"query": command, "status": "UNPARSEABLE", "stdout": completed.stdout.strip()}
    return {
        "status": "MEASURED",
        "name": fields[0],
        "uuid": fields[1],
        "compute_capability": fields[2],
        "driver_version": fields[3],
        "memory_total_mib": float(fields[4]),
    }


def benchmark_real_mxfp4_expert(
    checkpoint: Path,
    cuda_library: Path,
    reference_library: Path,
    output_path: Path,
    *,
    device: int = 0,
    layer: int = 1,
    expert: int = 0,
    warmup: int = 5,
    iterations: int = 30,
    batch: int = 1,
    seed: int = 14025,
    cuda_architecture: str = "sm_120",
    up_first: bool = False,
    cycle_id: str = "H014-025b",
    fuse_gate_up: bool = False,
) -> dict[str, Any]:
    """Benchmark one immutable real Kimi expert against the proven CPU arithmetic."""

    if warmup < 1 or iterations < 5 or batch < 1:
        raise KimiCudaError("warmup >=1, iterations >=5, and batch >=1 are required")
    checkpoint_path = checkpoint.expanduser().resolve()
    cuda_path = cuda_library.expanduser().resolve()
    reference_path = reference_library.expanduser().resolve()
    real = _load_real_expert(checkpoint_path, layer, expert)
    generator = np.random.default_rng(seed)
    activation = np.ascontiguousarray(
        generator.normal(0.0, 0.05, (batch, real.gate.input_dimension)), dtype=np.float32
    )

    reference_runtime = NativeMXFP4Runtime(reference_path)
    reference_start = time.perf_counter_ns()
    gate = reference_runtime.matmul(activation, real.gate)
    up = reference_runtime.matmul(activation, real.up)
    hidden = reference_runtime.situ_glu(gate, up)
    reference = reference_runtime.matmul(hidden, real.down)
    reference_wall_ms = (time.perf_counter_ns() - reference_start) / 1e6

    runtime = _CudaRuntime(cuda_path, device)
    runtime.set_up_first(up_first)
    runtime.set_fused_gate_up(fuse_gate_up)
    memory_before_upload = runtime.mem_info()
    handles = (runtime.upload(real.gate), runtime.upload(real.up), runtime.upload(real.down))
    memory_after_upload = runtime.mem_info()
    resident_bytes = sum(runtime.tensor_bytes(handle) for handle in handles)
    expected_resident_bytes = real.gate.byte_size + real.up.byte_size + real.down.byte_size
    if resident_bytes != expected_resident_bytes:
        runtime.close()
        raise KimiCudaError(
            f"resident tensor accounting mismatch: {resident_bytes} != {expected_resident_bytes}"
        )

    modes: dict[str, Any] = {}
    outputs: dict[str, np.ndarray] = {}
    memory_after_warmup: dict[str, int] | None = None
    try:
        for mode in ("minimal", "production", "detailed"):
            runtime.set_telemetry(mode)
            for _ in range(warmup):
                runtime.execute(handles, activation)
            if memory_after_warmup is None:
                memory_after_warmup = runtime.mem_info()
            runtime.reset_stats()
            wall: list[float] = []
            host: list[float] = []
            observed: np.ndarray | None = None
            for _ in range(iterations):
                wall_start = time.perf_counter_ns()
                host_start = time.process_time_ns()
                observed = runtime.execute(handles, activation)
                host.append((time.process_time_ns() - host_start) / 1e6)
                wall.append((time.perf_counter_ns() - wall_start) / 1e6)
            assert observed is not None
            outputs[mode] = observed
            stats = runtime.stats()
            calls = int(stats["calls"])
            modes[mode] = {
                "wall": _percentiles(wall),
                "host_cpu": _percentiles(host),
                "telemetry": stats,
                "telemetry_per_call": {
                    "h2d_ms": float(stats["h2d_ms"]) / calls if calls else 0.0,
                    "kernel_ms": float(stats["kernel_ms"]) / calls if calls else 0.0,
                    "d2h_ms": float(stats["d2h_ms"]) / calls if calls else 0.0,
                    "gate_ms": float(stats["gate_ms"]) / calls if calls else 0.0,
                    "up_ms": float(stats["up_ms"]) / calls if calls else 0.0,
                    "situ_ms": float(stats["situ_ms"]) / calls if calls else 0.0,
                    "down_ms": float(stats["down_ms"]) / calls if calls else 0.0,
                    "gate_up_pair_ms": float(stats["gate_up_pair_ms"]) / calls
                    if calls
                    else 0.0,
                },
                "output_fingerprint": _array_fingerprint(observed),
            }

        runtime.set_telemetry("minimal")
        resident_input = runtime.allocate(activation.nbytes)
        resident_output = runtime.allocate(reference.nbytes)
        try:
            runtime.upload_activation(resident_input, activation)
            for _ in range(warmup):
                runtime.execute_resident(handles, resident_output, resident_input, batch)
            runtime.synchronize()
            resident_observed = runtime.download_activation(resident_output, reference.shape)
            resident_correctness = _numerical_metrics(reference, resident_observed)

            synchronized_wall: list[float] = []
            for _ in range(iterations):
                start = time.perf_counter_ns()
                runtime.execute_resident(handles, resident_output, resident_input, batch)
                runtime.synchronize()
                synchronized_wall.append((time.perf_counter_ns() - start) / 1e6)

            queued_start = time.perf_counter_ns()
            for _ in range(iterations):
                runtime.execute_resident(handles, resident_output, resident_input, batch)
            runtime.synchronize()
            queued_per_call_ms = (time.perf_counter_ns() - queued_start) / 1e6 / iterations
            resident = {
                "backend_identity": "nvidia_cuda_device_resident",
                "cpu_fallback_allowed": False,
                "warmup_calls": warmup,
                "retained_calls": iterations,
                "h2d_bytes_per_call": 0,
                "d2h_bytes_per_call": 0,
                "one_time_activation_upload_bytes": int(activation.nbytes),
                "one_time_validation_download_bytes": int(reference.nbytes),
                "synchronized_generation": _percentiles(synchronized_wall),
                "queued_service_ms_per_call": queued_per_call_ms,
                "correctness": resident_correctness,
                "output_fingerprint": _array_fingerprint(resident_observed),
            }
        finally:
            runtime.free(resident_output)
            runtime.free(resident_input)
    finally:
        runtime.close()

    correctness = _numerical_metrics(reference, outputs["minimal"])
    mode_equality = {
        mode: _numerical_metrics(outputs["minimal"], value)
        for mode, value in outputs.items()
    }
    minimal_p50 = float(modes["minimal"]["wall"]["p50_ms"])
    resident_p50 = float(resident["synchronized_generation"]["p50_ms"])
    resident_hypothesis = {
        "minimum_synchronized_p50_improvement_percent": 15.0,
        "minimum_queued_service_improvement_percent": 25.0,
        "synchronized_p50_improvement_percent": (1.0 - resident_p50 / minimal_p50) * 100.0,
        "queued_service_improvement_percent": (
            1.0 - float(resident["queued_service_ms_per_call"]) / minimal_p50
        )
        * 100.0,
    }
    resident_hypothesis["supported"] = (
        resident_hypothesis["synchronized_p50_improvement_percent"] >= 15.0
        and resident_hypothesis["queued_service_improvement_percent"] >= 25.0
    )
    overhead = {
        mode: {
            "p50_ms": float(row["wall"]["p50_ms"]),
            "relative_to_minimal_percent": (
                (float(row["wall"]["p50_ms"]) / minimal_p50 - 1.0) * 100.0
                if minimal_p50
                else 0.0
            ),
        }
        for mode, row in modes.items()
    }
    detailed_kernel_ms = float(modes["detailed"]["telemetry_per_call"]["kernel_ms"])
    exposed_sync_ms = max(
        0.0,
        float(modes["detailed"]["wall"]["p50_ms"])
        - sum(
            float(modes["detailed"]["telemetry_per_call"][key])
            for key in ("h2d_ms", "kernel_ms", "d2h_ms")
        ),
    )
    threshold = {
        "maximum_relative_l2_error": 3e-4,
        "minimum_cosine_similarity": 0.999999,
    }
    telemetry_contract = (
        int(modes["minimal"]["telemetry"]["calls"]) == 0
        and int(modes["production"]["telemetry"]["calls"]) == iterations
        and float(modes["production"]["telemetry"]["kernel_ms"]) == 0.0
        and int(modes["detailed"]["telemetry"]["calls"]) == iterations
        and float(modes["detailed"]["telemetry"]["kernel_ms"]) > 0.0
    )
    passed = (
        bool(correctness["reference_finite"])
        and bool(correctness["actual_finite"])
        and float(correctness["relative_l2_error"]) <= threshold["maximum_relative_l2_error"]
        and float(correctness["cosine_similarity"]) >= threshold["minimum_cosine_similarity"]
        and all(float(row["relative_l2_error"]) == 0.0 for row in mode_equality.values())
        and telemetry_contract
        and float(resident["correctness"]["relative_l2_error"])
        <= threshold["maximum_relative_l2_error"]
    )
    tensor_rows = {}
    for role, tensor in (("gate", real.gate), ("up", real.up), ("down", real.down)):
        tensor_rows[role] = {
            "dimensions": [tensor.output_dimension, tensor.input_dimension],
            "packed_shape": list(tensor.packed.shape),
            "scale_shape": list(tensor.scales.shape),
            "packed_dtype": str(tensor.packed.dtype),
            "scale_dtype": str(tensor.scales.dtype),
            "packed_fingerprint": _array_fingerprint(tensor.packed),
            "scale_fingerprint": _array_fingerprint(tensor.scales),
            "resident_bytes": tensor.byte_size,
            "tensor_names": real.names[role],
        }
    index_path = checkpoint_path / "model.safetensors.index.json"
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "cycle_id": cycle_id,
        "status": "PASS" if passed else "FAIL",
        "hypothesis": (
            "The existing resident CUDA expert path can consume byte-exact Kimi E2M1/UE8M0 "
            "group-32 tensors with a SiTU epilogue, match the serial FP32-activation oracle "
            "within 3e-4 relative L2 error, and execute without CPU fallback or layout conversion."
        ),
        "backend": {
            "identity": "nvidia_cuda_direct_dll",
            "cpu_fallback_allowed": False,
            "cuda_library": str(cuda_path),
            "cuda_library_sha256": runtime.sha256,
            "compiled_architecture": cuda_architecture,
            "binary_min_compute_capability": runtime.binary_min_compute_capability,
            "binary_has_forward_ptx": runtime.binary_has_forward_ptx,
            "capability_negotiation": runtime.capability_negotiation,
            "projection_order": "up_then_gate" if up_first else "gate_then_up",
            "gate_up_fused": fuse_gate_up,
            "device": _device_identity(device),
            "reference_identity": NativeMXFP4Runtime.ABI,
            "reference_library": str(reference_path),
            "reference_library_sha256": reference_runtime.sha256,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "index_sha256": _sha256_file(index_path),
            "source_shards": real.source_shards,
            "layer": layer,
            "expert_id": expert,
        },
        "fixture": {
            "activation_source": "deterministic_fp32_latent_fixture_with_real_checkpoint_weights",
            "seed": seed,
            "batch": batch,
            "activation_dimensions": list(activation.shape),
            "activation_dtype": str(activation.dtype),
            "activation_fingerprint": _array_fingerprint(activation),
            "activation_minimum": float(np.min(activation)),
            "activation_maximum": float(np.max(activation)),
            "tensors": tensor_rows,
            "reference_output_fingerprint": _array_fingerprint(reference),
        },
        "correctness": correctness,
        "correctness_threshold": threshold,
        "telemetry_mode_output_equality": mode_equality,
        "benchmark": {
            "warmup_iterations_per_mode": warmup,
            "retained_iterations_per_mode": iterations,
            "reference_wall_ms": reference_wall_ms,
            "modes": modes,
            "resident": resident,
            "resident_hypothesis": resident_hypothesis,
            "telemetry_overhead": overhead,
            "telemetry_contract_passed": telemetry_contract,
            "phase_hypothesis": {
                "prediction": "MXFP4 GEMMs consume more than 95% of aggregate kernel time",
                "gemm_share_percent": (
                    100.0
                    * sum(
                        float(modes["detailed"]["telemetry_per_call"][key])
                        for key in ("gate_ms", "up_ms", "gate_up_pair_ms", "down_ms")
                    )
                    / detailed_kernel_ms
                    if detailed_kernel_ms
                    else None
                ),
            },
            "detailed_exposed_host_or_sync_ms": exposed_sync_ms,
            "kernel_count_per_call": 4,
            "memory_copies_per_call": 2,
            "h2d_bytes_per_call": int(activation.nbytes),
            "d2h_bytes_per_call": int(reference.nbytes),
            "d2d_bytes_per_call": 0,
            "effective_weight_bandwidth_gbps": (
                expected_resident_bytes / (detailed_kernel_ms / 1000.0) / 1e9
                if detailed_kernel_ms
                else None
            ),
            "gpu_utilization": None,
            "compute_utilization": None,
            "utilization_note": (
                "not retained for this sub-millisecond component fixture; a kernel profiler is "
                "required before bottleneck classification"
            ),
        },
        "memory": {
            "before_upload": memory_before_upload,
            "after_upload": memory_after_upload,
            "after_warmup": memory_after_warmup,
            "tensor_resident_bytes": resident_bytes,
            "expected_tensor_resident_bytes": expected_resident_bytes,
            "observed_upload_vram_delta_bytes": max(
                0, memory_before_upload["free_bytes"] - memory_after_upload["free_bytes"]
            ),
            "observed_workspace_vram_delta_bytes": max(
                0,
                memory_after_upload["free_bytes"]
                - (memory_after_warmup or memory_after_upload)["free_bytes"],
            ),
        },
        "inspection": {
            "layout_conversion_kernel_count": 0,
            "persistent_dequantized_weight_bytes": 0,
            "routing_equality": "NOT_APPLICABLE_FIXED_EXPERT_ID",
            "selected_expert_ids": [expert],
            "decision": "RETAIN" if passed else "MODIFY",
            "bottleneck": "PENDING_KERNEL_PROFILE" if passed else "NUMERICAL_CORRECTNESS",
        },
    }
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": payload["status"],
        "output_path": str(destination),
        "output_sha256": _sha256_file(destination),
        "relative_l2_error": correctness["relative_l2_error"],
        "cosine_similarity": correctness["cosine_similarity"],
        "minimal_p50_ms": modes["minimal"]["wall"]["p50_ms"],
        "detailed_kernel_ms": detailed_kernel_ms,
        "production_telemetry_overhead_percent": overhead["production"][
            "relative_to_minimal_percent"
        ],
        "detailed_telemetry_overhead_percent": overhead["detailed"][
            "relative_to_minimal_percent"
        ],
    }


def benchmark_real_router(
    checkpoint: Path,
    cuda_library: Path,
    activation_path: Path,
    output_path: Path,
    *,
    device: int = 0,
    layer: int = 1,
    warmup: int = 50,
    iterations: int = 500,
    cuda_architecture: str = "sm_86+c86_ptx",
) -> dict[str, Any]:
    """Certify exact Kimi top-16 routing through the resident CUDA primitive."""

    if warmup < 1 or iterations < 20:
        raise KimiCudaError("router benchmark requires warmup >=1 and iterations >=20")
    checkpoint_path = checkpoint.expanduser().resolve()
    index_path = checkpoint_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    prefix = f"language_model.model.layers.{layer}.block_sparse_moe.gate"
    router, router_evidence = _load_safetensor_f32(
        checkpoint_path, index, f"{prefix}.weight"
    )
    bias, bias_evidence = _load_safetensor_f32(
        checkpoint_path, index, f"{prefix}.e_score_correction_bias"
    )
    activation_source = activation_path.expanduser().resolve()
    activation = np.fromfile(activation_source, dtype=np.float32)
    if router.shape != (896, 7168) or bias.shape != (896,) or activation.shape != (7168,):
        raise KimiCudaError(
            f"unexpected router fixture geometry W={router.shape}, b={bias.shape}, x={activation.shape}"
        )
    activation = np.ascontiguousarray(activation, dtype=np.float32)

    logits = np.asarray(router @ activation, dtype=np.float32)
    sigmoid = np.asarray(1.0 / (1.0 + np.exp(-logits)), dtype=np.float32)
    choice = np.asarray(sigmoid + bias, dtype=np.float32)
    reference_indices: list[int] = []
    for _ in range(16):
        masked = choice.copy()
        if reference_indices:
            masked[np.asarray(reference_indices, dtype=np.int64)] = -np.inf
        reference_indices.append(int(np.argmax(masked)))
    reference_index_array = np.asarray(reference_indices, dtype=np.int32)
    reference_weights = np.ascontiguousarray(sigmoid[reference_index_array], dtype=np.float32)
    reference_weights /= np.sum(reference_weights, dtype=np.float32) + np.float32(1e-20)
    unselected = np.ones(896, dtype=bool)
    unselected[reference_index_array] = False
    selection_margin = float(
        np.min(choice[reference_index_array]) - np.max(choice[unselected])
    )

    runtime = _CudaRuntime(cuda_library.expanduser().resolve(), device)
    runtime.set_telemetry("minimal")
    x_device = runtime.allocate(activation.nbytes)
    w_device = runtime.allocate(router.nbytes)
    b_device = runtime.allocate(bias.nbytes)
    try:
        runtime.upload_activation(x_device, activation)
        runtime.upload_activation(w_device, router)
        runtime.upload_activation(b_device, bias)
        for _ in range(warmup):
            runtime.route(
                x_device, w_device, b_device, hidden=7168, experts=896, topk=16
            )
        wall: list[float] = []
        host: list[float] = []
        observed_indices: np.ndarray | None = None
        observed_weights: np.ndarray | None = None
        observed_effective = 0
        for _ in range(iterations):
            wall_start = time.perf_counter_ns()
            host_start = time.process_time_ns()
            observed_indices, observed_weights, observed_effective = runtime.route(
                x_device, w_device, b_device, hidden=7168, experts=896, topk=16
            )
            host.append((time.process_time_ns() - host_start) / 1e6)
            wall.append((time.perf_counter_ns() - wall_start) / 1e6)
        runtime.set_telemetry("detailed")
        runtime.reset_router_stats()
        for _ in range(min(iterations,100)):
            runtime.route(x_device,w_device,b_device,hidden=7168,experts=896,topk=16)
        profile=runtime.router_stats()
    finally:
        runtime.free(b_device)
        runtime.free(w_device)
        runtime.free(x_device)
        runtime.close()
    assert observed_indices is not None and observed_weights is not None
    weight_metrics = _numerical_metrics(reference_weights, observed_weights)
    routing_equal = bool(np.array_equal(reference_index_array, observed_indices))
    passed = (
        routing_equal
        and observed_effective == 16
        and float(weight_metrics["relative_l2_error"]) <= 1e-4
        and float(weight_metrics["cosine_similarity"]) >= 0.999999
    )
    profiled_total=(
        float(profile["logits_ms_per_call"])
        +float(profile["selection_ms_per_call"])
        +float(profile["d2h_ms_per_call"])
    )
    selection_share=(
        100.0*float(profile["selection_ms_per_call"])/profiled_total
        if profiled_total else 0.0
    )
    payload = {
        "schema_version": "experiment-014-k3-cuda-router-v1",
        "cycle_id": "H014-025i",
        "status": "PASS" if passed else "FAIL",
        "hypothesis": (
            "The existing resident CUDA router exactly reproduces Kimi top-16 IDs and keff "
            "with selected-weight relative L2 <=1e-4 on real router weights and activation."
        ),
        "backend": {
            "identity": "nvidia_cuda_device_resident_router",
            "cpu_fallback_allowed": False,
            "cuda_library": str(cuda_library.expanduser().resolve()),
            "cuda_library_sha256": _sha256_file(cuda_library.expanduser().resolve()),
            "compiled_architecture": cuda_architecture,
            "binary_min_compute_capability": runtime.binary_min_compute_capability,
            "binary_has_forward_ptx": runtime.binary_has_forward_ptx,
            "capability_negotiation": runtime.capability_negotiation,
            "device": _device_identity(device),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "index_sha256": _sha256_file(index_path),
            "layer": layer,
            "router": router_evidence,
            "bias": bias_evidence,
        },
        "fixture": {
            "activation_source": str(activation_source),
            "activation_source_sha256": _sha256_file(activation_source),
            "activation_dimensions": list(activation.shape),
            "activation_dtype": str(activation.dtype),
            "activation_fingerprint": _array_fingerprint(activation),
        },
        "correctness": {
            "routing_equality": routing_equal,
            "reference_expert_ids": reference_index_array.tolist(),
            "cuda_expert_ids": observed_indices.tolist(),
            "reference_keff": 16,
            "cuda_keff": observed_effective,
            "selected_weight_metrics": weight_metrics,
            "selection_margin": selection_margin,
            "reference_weight_fingerprint": _array_fingerprint(reference_weights),
            "cuda_weight_fingerprint": _array_fingerprint(observed_weights),
        },
        "benchmark": {
            "warmup_iterations": warmup,
            "retained_iterations": iterations,
            "wall": _percentiles(wall),
            "host_cpu": _percentiles(host),
            "resident_weight_bytes": int(router.nbytes + bias.nbytes),
            "per_call_h2d_bytes": 0,
            "per_call_d2h_bytes": 16 * (4 + 4) + 4,
            "kernel_count_per_call": 2,
            "detailed_profile":profile,
            "selection_share_percent":selection_share,
        },
        "inspection": {
            "decision": "RETAIN" if passed else "MODIFY",
            "bottleneck": "PENDING_ROUTER_TIMING_INSPECTION" if passed else "ROUTING_DIVERGENCE",
        },
    }
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return {
        "schema_version": payload["schema_version"],
        "status": payload["status"],
        "output_path": str(destination),
        "output_sha256": _sha256_file(destination),
        "routing_equality": routing_equal,
        "selected_weight_relative_l2_error": weight_metrics["relative_l2_error"],
        "p50_ms": payload["benchmark"]["wall"]["p50_ms"],
        "p95_ms": payload["benchmark"]["wall"]["p95_ms"],
        "p99_ms": payload["benchmark"]["wall"]["p99_ms"],
    }


def benchmark_real_dense_projection(
    checkpoint: Path,
    cuda_library: Path,
    activation_path: Path,
    output_path: Path,
    *,
    device: int = 0,
    layer: int = 1,
    tensor_suffix: str = "self_attn.f_a_proj.weight",
    warmup: int = 50,
    iterations: int = 500,
    cuda_architecture: str = "sm_86+c86_ptx",
    cycle_id: str = "H014-025m",
) -> dict[str, Any]:
    """Test a production-format grouped-int4 Kimi projection on resident CUDA."""

    if warmup < 1 or iterations < 20:
        raise KimiCudaError("dense benchmark requires warmup >=1 and iterations >=20")
    checkpoint_path = checkpoint.expanduser().resolve()
    index_path = checkpoint_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    tensor_name = f"language_model.model.layers.{layer}.{tensor_suffix}"
    source_weights, source_evidence = _load_safetensor_f32(
        checkpoint_path, index, tensor_name
    )
    quantized = _quantize_grouped_int4(source_weights)
    activation_source = activation_path.expanduser().resolve()
    activation = np.ascontiguousarray(
        np.fromfile(activation_source, dtype=np.float32), dtype=np.float32
    )
    if activation.shape != (quantized.input_dimension,):
        raise KimiCudaError(
            f"activation {activation.shape} does not match projection input "
            f"{quantized.input_dimension}"
        )
    dequantized = _dequantize_grouped_int4(quantized)
    source_reference = np.ascontiguousarray(source_weights @ activation, dtype=np.float32)
    quantized_reference = np.ascontiguousarray(dequantized @ activation, dtype=np.float32)
    del dequantized

    runtime = _CudaRuntime(cuda_library.expanduser().resolve(), device)
    memory_before_upload = runtime.mem_info()
    handle = runtime.upload_grouped_int4(quantized)
    memory_after_upload = runtime.mem_info()
    input_device = runtime.allocate(activation.nbytes)
    output_bytes = quantized.output_dimension * np.dtype(np.float32).itemsize
    output_device = runtime.allocate(output_bytes)
    memory_after_buffers = runtime.mem_info()
    observed: np.ndarray | None = None
    try:
        runtime.upload_activation(input_device, activation)
        runtime.set_telemetry("minimal")
        for _ in range(warmup):
            runtime.execute_dense(handle, output_device, input_device, 1)
            runtime.synchronize()
        wall: list[float] = []
        host: list[float] = []
        for _ in range(iterations):
            wall_start = time.perf_counter_ns()
            host_start = time.process_time_ns()
            runtime.execute_dense(handle, output_device, input_device, 1)
            runtime.synchronize()
            host.append((time.process_time_ns() - host_start) / 1e6)
            wall.append((time.perf_counter_ns() - wall_start) / 1e6)
        queued_start = time.perf_counter_ns()
        for _ in range(iterations):
            runtime.execute_dense(handle, output_device, input_device, 1)
        runtime.synchronize()
        queued_service_ms = (time.perf_counter_ns() - queued_start) / 1e6 / iterations
        runtime.set_telemetry("detailed")
        runtime.reset_dense_stats()
        profiled_calls = min(iterations, 100)
        for _ in range(profiled_calls):
            runtime.execute_dense(handle, output_device, input_device, 1)
        profile = runtime.dense_stats()
        runtime.set_telemetry("minimal")
        observed = runtime.download_activation(
            output_device, (quantized.output_dimension,)
        )
    finally:
        runtime.free(output_device)
        runtime.free(input_device)
        runtime.close()
    assert observed is not None
    execution_metrics = _numerical_metrics(quantized_reference, observed)
    quantization_metrics = _numerical_metrics(source_reference, quantized_reference)
    passed = (
        float(execution_metrics["relative_l2_error"]) <= 1e-5
        and float(execution_metrics["cosine_similarity"]) >= 0.999999
        and bool(execution_metrics["actual_finite"])
    )
    wall_metrics = _percentiles(wall)
    kernel_ms = float(profile["kernel_ms_per_call"])
    kernel_share = 100.0 * kernel_ms / wall_metrics["p50_ms"] if wall_metrics["p50_ms"] else 0.0
    # The tensor has been released; its exact storage contract is deterministic.
    resident_bytes = int(quantized.packed.nbytes + quantized.scales.nbytes)
    payload = {
        "schema_version": "experiment-014-k3-cuda-real-dense-v1",
        "cycle_id": cycle_id,
        "status": "PASS" if passed else "FAIL",
        "hypothesis": (
            "The existing resident grouped-int4 CUDA GEMV consumes Kimi's exact "
            "load-time int4-g64 representation and matches an independent dequantized "
            "oracle within 1e-5 relative L2 without CPU fallback."
        ),
        "backend": {
            "identity": "nvidia_cuda_device_resident_grouped_int4",
            "cpu_fallback_allowed": False,
            "cuda_library": str(cuda_library.expanduser().resolve()),
            "cuda_library_sha256": _sha256_file(cuda_library.expanduser().resolve()),
            "compiled_architecture": cuda_architecture,
            "binary_min_compute_capability": runtime.binary_min_compute_capability,
            "binary_has_forward_ptx": runtime.binary_has_forward_ptx,
            "capability_negotiation": runtime.capability_negotiation,
            "device": _device_identity(device),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "index_sha256": _sha256_file(index_path),
            "layer": layer,
            "tensor": source_evidence,
        },
        "fixture": {
            "dimensions": [quantized.output_dimension, quantized.input_dimension],
            "source_dtype": source_evidence["source_dtype"],
            "runtime_dtype": "packed signed int4 with float32 group-64 scales",
            "packed_weight_fingerprint": _array_fingerprint(quantized.packed),
            "scale_fingerprint": _array_fingerprint(quantized.scales),
            "activation_source": str(activation_source),
            "activation_source_sha256": _sha256_file(activation_source),
            "activation_fingerprint": _array_fingerprint(activation),
            "reference_output_fingerprint": _array_fingerprint(quantized_reference),
            "cuda_output_fingerprint": _array_fingerprint(observed),
        },
        "correctness": {
            "cuda_vs_production_quantized_oracle": execution_metrics,
            "production_int4_quantization_vs_source_bf16": quantization_metrics,
            "threshold": {
                "relative_l2_error_maximum": 1e-5,
                "cosine_similarity_minimum": 0.999999,
            },
        },
        "benchmark": {
            "warmup_iterations": warmup,
            "retained_iterations": iterations,
            "wall": wall_metrics,
            "host_cpu": _percentiles(host),
            "queued_service_ms_per_call": queued_service_ms,
            "detailed_profile": profile,
            "kernel_share_of_synchronized_p50_percent": kernel_share,
            "kernel_count_per_call": 1,
            "per_call_h2d_bytes": 0,
            "per_call_d2h_bytes": 0,
            "one_time_h2d_bytes": int(activation.nbytes),
            "one_time_validation_d2h_bytes": output_bytes,
            "resident_weight_bytes": resident_bytes,
            "effective_weight_bandwidth_gbps": (
                resident_bytes / (kernel_ms / 1000.0) / 1e9 if kernel_ms else None
            ),
            "gpu_utilization": None,
            "compute_utilization": None,
            "utilization_note": "micro-kernel utilization requires a kernel profiler",
        },
        "memory": {
            "before_upload": memory_before_upload,
            "after_upload": memory_after_upload,
            "after_buffers": memory_after_buffers,
            "tensor_resident_bytes": resident_bytes,
            "activation_buffer_bytes": int(activation.nbytes),
            "output_buffer_bytes": output_bytes,
            "workspace_bytes": 0,
            "observed_upload_vram_delta_bytes": max(
                0, memory_before_upload["free_bytes"] - memory_after_upload["free_bytes"]
            ),
            "observed_buffer_vram_delta_bytes": max(
                0, memory_after_upload["free_bytes"] - memory_after_buffers["free_bytes"]
            ),
        },
        "inspection": {
            "layout_conversion_kernel_count": 1,
            "layout_conversion_scope": (
                "one-time in-place offset-binary-to-signed nibble conversion at upload"
            ),
            "persistent_dequantized_weight_bytes": 0,
            "decision": "RETAIN" if passed else "MODIFY",
            "bottleneck": (
                "SYNCHRONIZATION_OR_LAUNCH_BOUND"
                if kernel_share < 50.0
                else "MIXED_PENDING_KERNEL_PROFILE"
            ),
        },
    }
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return {
        "schema_version": payload["schema_version"],
        "status": payload["status"],
        "output_path": str(destination),
        "output_sha256": _sha256_file(destination),
        "relative_l2_error": execution_metrics["relative_l2_error"],
        "cosine_similarity": execution_metrics["cosine_similarity"],
        "p50_ms": wall_metrics["p50_ms"],
        "p95_ms": wall_metrics["p95_ms"],
        "p99_ms": wall_metrics["p99_ms"],
        "kernel_ms": kernel_ms,
    }


def benchmark_real_final_norm(
    checkpoint: Path,
    cuda_library: Path,
    activation_path: Path,
    output_path: Path,
    *,
    device: int = 0,
    warmup: int = 50,
    iterations: int = 500,
    cuda_architecture: str = "sm_86+c86_ptx",
    cycle_id: str = "H014-025o",
) -> dict[str, Any]:
    """Certify final Kimi RMSNorm using real scale weights and activation state."""

    if warmup < 1 or iterations < 20:
        raise KimiCudaError("final-norm benchmark requires warmup >=1 and iterations >=20")
    checkpoint_path = checkpoint.expanduser().resolve()
    index_path = checkpoint_path / "model.safetensors.index.json"
    config_path = checkpoint_path / "config.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    epsilon = float(config.get("text_config", {}).get("rms_norm_eps", 1e-5))
    weight, weight_evidence = _load_safetensor_f32(
        checkpoint_path, index, "language_model.model.norm.weight"
    )
    activation_source = activation_path.expanduser().resolve()
    activation = np.ascontiguousarray(
        np.fromfile(activation_source, dtype=np.float32), dtype=np.float32
    )
    if weight.shape != activation.shape or weight.ndim != 1:
        raise KimiCudaError(
            f"final norm geometry mismatch weight={weight.shape}, activation={activation.shape}"
        )
    dimension = int(weight.shape[0])
    mean_square = np.float32(
        np.sum(activation.astype(np.float64) ** 2, dtype=np.float64) / dimension
    )
    inverse_rms = np.float32(1.0) / np.sqrt(
        mean_square + np.float32(epsilon)
    )
    reference = np.ascontiguousarray(
        activation * inverse_rms * weight, dtype=np.float32
    )

    runtime = _CudaRuntime(cuda_library.expanduser().resolve(), device)
    memory_before = runtime.mem_info()
    input_device = runtime.allocate(activation.nbytes)
    weight_device = runtime.allocate(weight.nbytes)
    output_device = runtime.allocate(reference.nbytes)
    memory_after_buffers = runtime.mem_info()
    observed: np.ndarray | None = None
    try:
        runtime.upload_activation(input_device, activation)
        runtime.upload_activation(weight_device, weight)
        for _ in range(warmup):
            runtime.execute_rmsnorm(
                output_device,
                input_device,
                weight_device,
                batch=1,
                dimension=dimension,
                epsilon=epsilon,
            )
            runtime.synchronize()
        wall: list[float] = []
        host: list[float] = []
        for _ in range(iterations):
            wall_start = time.perf_counter_ns()
            host_start = time.process_time_ns()
            runtime.execute_rmsnorm(
                output_device,
                input_device,
                weight_device,
                batch=1,
                dimension=dimension,
                epsilon=epsilon,
            )
            runtime.synchronize()
            host.append((time.process_time_ns() - host_start) / 1e6)
            wall.append((time.perf_counter_ns() - wall_start) / 1e6)
        queued_start = time.perf_counter_ns()
        for _ in range(iterations):
            runtime.execute_rmsnorm(
                output_device,
                input_device,
                weight_device,
                batch=1,
                dimension=dimension,
                epsilon=epsilon,
            )
        runtime.synchronize()
        queued_service_ms = (time.perf_counter_ns() - queued_start) / 1e6 / iterations
        profile_calls = min(iterations, 100)
        kernel_samples: list[float] = []
        for _ in range(profile_calls):
            runtime.profile_begin()
            runtime.execute_rmsnorm(
                output_device,
                input_device,
                weight_device,
                batch=1,
                dimension=dimension,
                epsilon=epsilon,
            )
            kernel_samples.append(runtime.profile_end())
        runtime.profile_begin()
        for _ in range(iterations):
            runtime.execute_rmsnorm(
                output_device,
                input_device,
                weight_device,
                batch=1,
                dimension=dimension,
                epsilon=epsilon,
            )
        batched_profile_ms_per_call = runtime.profile_end() / iterations
        observed = runtime.download_activation(output_device, reference.shape)
    finally:
        runtime.free(output_device)
        runtime.free(weight_device)
        runtime.free(input_device)
        runtime.close()
    assert observed is not None
    correctness = _numerical_metrics(reference, observed)
    passed = (
        float(correctness["relative_l2_error"]) <= 2e-6
        and float(correctness["cosine_similarity"]) >= 0.999999
        and bool(correctness["actual_finite"])
    )
    wall_metrics = _percentiles(wall)
    kernel_metrics = _percentiles(kernel_samples)
    bytes_touched = int(activation.nbytes + weight.nbytes + reference.nbytes)
    kernel_share = (
        100.0 * batched_profile_ms_per_call / wall_metrics["p50_ms"]
        if wall_metrics["p50_ms"]
        else 0.0
    )
    per_call_event_overhead = (
        100.0
        * (kernel_metrics["p50_ms"] - batched_profile_ms_per_call)
        / batched_profile_ms_per_call
        if batched_profile_ms_per_call
        else 0.0
    )
    payload = {
        "schema_version": "experiment-014-k3-cuda-real-final-norm-v1",
        "cycle_id": cycle_id,
        "status": "PASS" if passed else "FAIL",
        "hypothesis": (
            "The resident generic CUDA RMSNorm reproduces the serial FP32-activation "
            "final normalization with real Kimi scale weights at <=2e-6 relative L2."
        ),
        "backend": {
            "identity": "nvidia_cuda_device_resident_rmsnorm",
            "cpu_fallback_allowed": False,
            "cuda_library": str(cuda_library.expanduser().resolve()),
            "cuda_library_sha256": _sha256_file(cuda_library.expanduser().resolve()),
            "compiled_architecture": cuda_architecture,
            "binary_min_compute_capability": runtime.binary_min_compute_capability,
            "binary_has_forward_ptx": runtime.binary_has_forward_ptx,
            "capability_negotiation": runtime.capability_negotiation,
            "device": _device_identity(device),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "index_sha256": _sha256_file(index_path),
            "config_sha256": _sha256_file(config_path),
            "weight": weight_evidence,
            "rms_norm_epsilon": epsilon,
        },
        "fixture": {
            "dimensions": [dimension],
            "activation_dtype": str(activation.dtype),
            "weight_dtype": str(weight.dtype),
            "output_dtype": str(observed.dtype),
            "activation_source": str(activation_source),
            "activation_source_sha256": _sha256_file(activation_source),
            "activation_fingerprint": _array_fingerprint(activation),
            "weight_fingerprint": _array_fingerprint(weight),
            "reference_output_fingerprint": _array_fingerprint(reference),
            "cuda_output_fingerprint": _array_fingerprint(observed),
        },
        "correctness": correctness,
        "correctness_threshold": {
            "relative_l2_error_maximum": 2e-6,
            "cosine_similarity_minimum": 0.999999,
        },
        "benchmark": {
            "warmup_iterations": warmup,
            "retained_iterations": iterations,
            "profiled_iterations": profile_calls,
            "wall": wall_metrics,
            "host_cpu": _percentiles(host),
            "kernel": kernel_metrics,
            "batched_event_service_ms_per_call": batched_profile_ms_per_call,
            "per_call_event_perturbation_percent": per_call_event_overhead,
            "retained_kernel_measurement": "batched_event_service_ms_per_call",
            "queued_service_ms_per_call": queued_service_ms,
            "batched_device_service_share_of_synchronized_p50_percent": kernel_share,
            "kernel_count_per_call": 1,
            "per_call_h2d_bytes": 0,
            "per_call_d2h_bytes": 0,
            "bytes_touched_per_call": bytes_touched,
            "effective_bytes_touched_service_bandwidth_gbps": (
                bytes_touched / (batched_profile_ms_per_call / 1000.0) / 1e9
                if batched_profile_ms_per_call
                else None
            ),
            "gpu_utilization": None,
            "compute_utilization": None,
            "utilization_note": "single-row micro-kernel utilization is not sampled reliably",
        },
        "memory": {
            "before_buffers": memory_before,
            "after_buffers": memory_after_buffers,
            "persistent_weight_bytes": int(weight.nbytes),
            "activation_buffer_bytes": int(activation.nbytes),
            "output_buffer_bytes": int(reference.nbytes),
            "workspace_bytes": 0,
        },
        "inspection": {
            "decision": "RETAIN" if passed else "MODIFY",
            "bottleneck": "MIXED_TINY_DEVICE_KERNEL_LAUNCH_AND_SYNCHRONIZATION",
            "timing_decision": (
                "retain batched event service; reject per-call event timing when "
                "instrumentation perturbation exceeds 10%"
            ),
        },
    }
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return {
        "schema_version": payload["schema_version"],
        "status": payload["status"],
        "output_path": str(destination),
        "output_sha256": _sha256_file(destination),
        "relative_l2_error": correctness["relative_l2_error"],
        "cosine_similarity": correctness["cosine_similarity"],
        "p50_ms": wall_metrics["p50_ms"],
        "p95_ms": wall_metrics["p95_ms"],
        "p99_ms": wall_metrics["p99_ms"],
        "kernel_p50_ms": kernel_metrics["p50_ms"],
        "retained_kernel_service_ms": batched_profile_ms_per_call,
        "per_call_event_perturbation_percent": per_call_event_overhead,
    }


def benchmark_real_embedding(
    checkpoint: Path,
    cuda_library: Path,
    output_path: Path,
    *,
    device: int = 0,
    warmup: int = 50,
    iterations: int = 500,
    cuda_architecture: str = "sm_86+c86_ptx",
    cycle_id: str = "H014-025q",
) -> dict[str, Any]:
    """Certify a fully resident real Kimi BF16 token-embedding gather."""

    if warmup < 1 or iterations < 20:
        raise KimiCudaError("embedding benchmark requires warmup >=1 and iterations >=20")
    checkpoint_path = checkpoint.expanduser().resolve()
    index_path = checkpoint_path / "model.safetensors.index.json"
    config_path = checkpoint_path / "config.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    table, table_evidence = _load_safetensor_raw_bf16(
        checkpoint_path, index, "language_model.model.embed_tokens.weight"
    )
    vocab_size = int(config.get("text_config", {}).get("vocab_size", table.shape[0]))
    hidden = int(table.shape[1])
    if table.shape != (vocab_size, hidden) or hidden != 7168:
        raise KimiCudaError(f"unexpected Kimi embedding geometry {table.shape}")
    text_config = config.get("text_config", {})
    token_ids = np.ascontiguousarray(
        [
            11,
            int(text_config["bos_token_id"]),
            int(text_config["eos_token_id"]),
            int(text_config["pad_token_id"]),
            0,
            vocab_size - 2,
        ],
        dtype=np.int32,
    )
    selected_bits = np.ascontiguousarray(table[token_ids], dtype=np.uint16)
    reference = np.ascontiguousarray(
        (selected_bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
    )

    runtime = _CudaRuntime(cuda_library.expanduser().resolve(), device)
    memory_before_upload = runtime.mem_info()
    upload_start = time.perf_counter_ns()
    embedding = runtime.upload_bf16_embedding(table)
    runtime.synchronize()
    upload_ms = (time.perf_counter_ns() - upload_start) / 1e6
    memory_after_upload = runtime.mem_info()
    resident_bytes = runtime.tensor_bytes(embedding)
    token_device = runtime.allocate(token_ids.nbytes)
    output_device = runtime.allocate(reference.nbytes)
    memory_after_buffers = runtime.mem_info()
    observed: np.ndarray | None = None
    try:
        runtime.upload_bytes(token_device, token_ids)
        for _ in range(warmup):
            runtime.execute_embedding(embedding, output_device, token_device, 1)
            runtime.synchronize()
        wall: list[float] = []
        host: list[float] = []
        for _ in range(iterations):
            wall_start = time.perf_counter_ns()
            host_start = time.process_time_ns()
            runtime.execute_embedding(embedding, output_device, token_device, 1)
            runtime.synchronize()
            host.append((time.process_time_ns() - host_start) / 1e6)
            wall.append((time.perf_counter_ns() - wall_start) / 1e6)
        queued_start = time.perf_counter_ns()
        for _ in range(iterations):
            runtime.execute_embedding(embedding, output_device, token_device, 1)
        runtime.synchronize()
        queued_service_ms = (time.perf_counter_ns() - queued_start) / 1e6 / iterations
        runtime.profile_begin()
        for _ in range(iterations):
            runtime.execute_embedding(embedding, output_device, token_device, 1)
        profiled_service_ms = runtime.profile_end() / iterations
        runtime.execute_embedding(
            embedding, output_device, token_device, int(token_ids.shape[0])
        )
        observed = runtime.download_activation(output_device, reference.shape)
    finally:
        runtime.free(output_device)
        runtime.free(token_device)
        runtime.close()
    assert observed is not None
    correctness = _numerical_metrics(reference, observed)
    exact = bool(np.array_equal(reference.view(np.uint32), observed.view(np.uint32)))
    passed = exact and bool(correctness["actual_finite"])
    wall_metrics = _percentiles(wall)
    row_bytes_touched = hidden * (2 + 4) + np.dtype(np.int32).itemsize
    payload = {
        "schema_version": "experiment-014-k3-cuda-real-embedding-v1",
        "cycle_id": cycle_id,
        "status": "PASS" if passed else "FAIL",
        "hypothesis": (
            "A fully resident BF16 Kimi embedding table can gather actual and boundary "
            "token IDs on CUDA with bit-exact FP32 output and no warm-path host transfer."
        ),
        "backend": {
            "identity": "nvidia_cuda_device_resident_bf16_embedding",
            "implementation_kind": "custom Kimi CUDA",
            "cpu_fallback_allowed": False,
            "cuda_library": str(cuda_library.expanduser().resolve()),
            "cuda_library_sha256": _sha256_file(cuda_library.expanduser().resolve()),
            "compiled_architecture": cuda_architecture,
            "binary_min_compute_capability": runtime.binary_min_compute_capability,
            "binary_has_forward_ptx": runtime.binary_has_forward_ptx,
            "capability_negotiation": runtime.capability_negotiation,
            "device": _device_identity(device),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "index_sha256": _sha256_file(index_path),
            "config_sha256": _sha256_file(config_path),
            "embedding": table_evidence,
        },
        "fixture": {
            "dimensions": [vocab_size, hidden],
            "token_ids": token_ids.tolist(),
            "actual_generated_token_id": 11,
            "token_id_fingerprint": _array_fingerprint(token_ids),
            "selected_bf16_fingerprint": _array_fingerprint(selected_bits),
            "reference_output_fingerprint": _array_fingerprint(reference),
            "cuda_output_fingerprint": _array_fingerprint(observed),
            "output_dtype": str(observed.dtype),
        },
        "correctness": {
            **correctness,
            "bit_exact": exact,
        },
        "benchmark": {
            "warmup_iterations": warmup,
            "retained_iterations": iterations,
            "timed_batch": 1,
            "wall": wall_metrics,
            "host_cpu": _percentiles(host),
            "queued_service_ms_per_call": queued_service_ms,
            "batched_event_service_ms_per_call": profiled_service_ms,
            "kernel_count_per_call": 1,
            "per_call_h2d_bytes": 0,
            "per_call_d2h_bytes": 0,
            "bytes_touched_per_call": row_bytes_touched,
            "effective_bytes_touched_service_bandwidth_gbps": (
                row_bytes_touched / (profiled_service_ms / 1000.0) / 1e9
                if profiled_service_ms
                else None
            ),
            "gpu_utilization": None,
            "compute_utilization": None,
            "utilization_note": "one-row gather is a launch-scale operation",
        },
        "memory": {
            "before_upload": memory_before_upload,
            "after_upload": memory_after_upload,
            "after_buffers": memory_after_buffers,
            "embedding_resident_bytes": resident_bytes,
            "expected_embedding_resident_bytes": int(table.nbytes),
            "upload_ms": upload_ms,
            "upload_effective_gbps": table.nbytes / (upload_ms / 1000.0) / 1e9,
            "persistent_dequantized_weight_bytes": 0,
            "token_buffer_bytes": int(token_ids.nbytes),
            "output_buffer_bytes": int(reference.nbytes),
            "workspace_bytes": 0,
            "observed_upload_vram_delta_bytes": max(
                0, memory_before_upload["free_bytes"] - memory_after_upload["free_bytes"]
            ),
        },
        "inspection": {
            "warm_path_weight_loading": 0,
            "warm_path_h2d_bytes": 0,
            "warm_path_d2h_bytes": 0,
            "invalid_token_behavior": "device writes NaN; caller must validate token range",
            "decision": "RETAIN" if passed else "MODIFY",
            "bottleneck": "ONE_TIME_TABLE_UPLOAD" if passed else "GATHER_CORRECTNESS",
        },
    }
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return {
        "schema_version": payload["schema_version"],
        "status": payload["status"],
        "output_path": str(destination),
        "output_sha256": _sha256_file(destination),
        "bit_exact": exact,
        "relative_l2_error": correctness["relative_l2_error"],
        "p50_ms": wall_metrics["p50_ms"],
        "p95_ms": wall_metrics["p95_ms"],
        "p99_ms": wall_metrics["p99_ms"],
        "resident_bytes": resident_bytes,
    }


def benchmark_real_lm_head(
    checkpoint: Path,
    cuda_library: Path,
    trace_path: Path,
    oracle_logits_path: Path,
    output_path: Path,
    *,
    trace_row: int = 187,
    logits_row: int = 1,
    device: int = 0,
    warmup: int = 20,
    iterations: int = 100,
    cuda_architecture: str = "sm_86+c86_ptx",
) -> dict[str, Any]:
    """Certify the production-int8 LM head against retained full-model logits."""

    if warmup < 1 or iterations < 20:
        raise KimiCudaError("LM-head benchmark requires warmup >=1 and iterations >=20")
    checkpoint_path = checkpoint.expanduser().resolve()
    index_path = checkpoint_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    raw_head, head_evidence = _load_safetensor_raw_bf16(
        checkpoint_path, index, "language_model.lm_head.weight"
    )
    if raw_head.shape != (163840, 7168):
        raise KimiCudaError(f"unexpected Kimi LM-head geometry {raw_head.shape}")
    trace_source = trace_path.expanduser().resolve()
    trace = np.memmap(trace_source, mode="r", dtype="<f4")
    if trace.size % 7168:
        raise KimiCudaError("retained Kimi trace has a partial hidden row")
    trace = trace.reshape(-1, 7168)
    if trace_row < 0 or trace_row >= trace.shape[0]:
        raise KimiCudaError(f"trace row {trace_row} outside {trace.shape[0]} rows")
    activation = np.ascontiguousarray(trace[trace_row], dtype=np.float32)
    logits_source = oracle_logits_path.expanduser().resolve()
    serial_logits = np.memmap(logits_source, mode="r", dtype="<f4")
    if serial_logits.size % 163840:
        raise KimiCudaError("retained Kimi logits have a partial vocabulary row")
    serial_logits = serial_logits.reshape(-1, 163840)
    if logits_row < 0 or logits_row >= serial_logits.shape[0]:
        raise KimiCudaError(f"logits row {logits_row} outside {serial_logits.shape[0]} rows")
    serial_reference = np.ascontiguousarray(serial_logits[logits_row], dtype=np.float32)

    quantize_start = time.perf_counter_ns()
    quantized = _quantize_bf16_rows_int8(raw_head)
    quantize_ms = (time.perf_counter_ns() - quantize_start) / 1e6
    quantized_reference = np.empty(quantized.output_dimension, dtype=np.float32)
    source_reference = np.empty(quantized.output_dimension, dtype=np.float32)
    reference_start = time.perf_counter_ns()
    for start in range(0, quantized.output_dimension, 1024):
        stop = min(quantized.output_dimension, start + 1024)
        q8 = quantized.weights[start:stop].astype(np.float32)
        quantized_reference[start:stop] = np.asarray(q8 @ activation, dtype=np.float32)
        quantized_reference[start:stop] *= quantized.scales[start:stop]
        bits = np.asarray(raw_head[start:stop], dtype=np.uint16)
        source = (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
        source_reference[start:stop] = np.asarray(source @ activation, dtype=np.float32)
    reference_ms = (time.perf_counter_ns() - reference_start) / 1e6

    runtime = _CudaRuntime(cuda_library.expanduser().resolve(), device)
    memory_before_upload = runtime.mem_info()
    upload_start = time.perf_counter_ns()
    head = runtime.upload_int8(quantized)
    runtime.synchronize()
    upload_ms = (time.perf_counter_ns() - upload_start) / 1e6
    memory_after_upload = runtime.mem_info()
    resident_bytes = runtime.tensor_bytes(head)
    input_device = runtime.allocate(activation.nbytes)
    output_device = runtime.allocate(serial_reference.nbytes)
    memory_after_buffers = runtime.mem_info()
    observed: np.ndarray | None = None
    try:
        runtime.upload_activation(input_device, activation)
        runtime.set_telemetry("minimal")
        for _ in range(warmup):
            runtime.execute_dense(head, output_device, input_device, 1)
            runtime.synchronize()
        wall: list[float] = []
        host: list[float] = []
        for _ in range(iterations):
            wall_start = time.perf_counter_ns()
            host_start = time.process_time_ns()
            runtime.execute_dense(head, output_device, input_device, 1)
            runtime.synchronize()
            host.append((time.process_time_ns() - host_start) / 1e6)
            wall.append((time.perf_counter_ns() - wall_start) / 1e6)
        queued_start = time.perf_counter_ns()
        for _ in range(iterations):
            runtime.execute_dense(head, output_device, input_device, 1)
        runtime.synchronize()
        queued_service_ms = (time.perf_counter_ns() - queued_start) / 1e6 / iterations
        runtime.profile_begin()
        for _ in range(iterations):
            runtime.execute_dense(head, output_device, input_device, 1)
        profiled_service_ms = runtime.profile_end() / iterations
        observed = runtime.download_activation(output_device, serial_reference.shape)
    finally:
        runtime.free(output_device)
        runtime.free(input_device)
        runtime.close()
    assert observed is not None
    cuda_vs_quantized = _numerical_metrics(quantized_reference, observed)
    cuda_vs_serial = _numerical_metrics(serial_reference, observed)
    quantized_vs_serial = _numerical_metrics(serial_reference, quantized_reference)
    quantization_effect = _numerical_metrics(source_reference, quantized_reference)
    serial_argmax = int(np.argmax(serial_reference))
    cuda_argmax = int(np.argmax(observed))
    passed = (
        float(cuda_vs_quantized["relative_l2_error"]) <= 1e-5
        and float(cuda_vs_serial["relative_l2_error"]) <= 3e-4
        and serial_argmax == cuda_argmax == 11
        and bool(cuda_vs_serial["actual_finite"])
    )
    wall_metrics = _percentiles(wall)
    payload = {
        "schema_version": "experiment-014-k3-cuda-real-lm-head-v1",
        "cycle_id": "H014-025s",
        "status": "PASS" if passed else "FAIL",
        "hypothesis": (
            "The existing resident int8 CUDA GEMV reproduces retained real Kimi "
            "163840-way serial logits within 3e-4 relative L2 and preserves argmax token 11."
        ),
        "backend": {
            "identity": "nvidia_cuda_device_resident_int8_lm_head",
            "implementation_kind": "generic existing CUDA",
            "cpu_fallback_allowed": False,
            "cuda_library": str(cuda_library.expanduser().resolve()),
            "cuda_library_sha256": _sha256_file(cuda_library.expanduser().resolve()),
            "compiled_architecture": cuda_architecture,
            "binary_min_compute_capability": runtime.binary_min_compute_capability,
            "binary_has_forward_ptx": runtime.binary_has_forward_ptx,
            "capability_negotiation": runtime.capability_negotiation,
            "device": _device_identity(device),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "index_sha256": _sha256_file(index_path),
            "lm_head": head_evidence,
        },
        "fixture": {
            "dimensions": [quantized.output_dimension, quantized.input_dimension],
            "runtime_dtype": "signed int8 with one float32 scale per output row",
            "trace_source": str(trace_source),
            "trace_source_sha256": _sha256_file(trace_source),
            "trace_row": trace_row,
            "activation_raw_sha256": hashlib.sha256(activation.tobytes()).hexdigest(),
            "activation_fingerprint": _array_fingerprint(activation),
            "oracle_logits_source": str(logits_source),
            "oracle_logits_source_sha256": _sha256_file(logits_source),
            "oracle_logits_row": logits_row,
            "serial_logits_fingerprint": _array_fingerprint(serial_reference),
            "quantized_weight_fingerprint": _array_fingerprint_streaming(
                quantized.weights
            ),
            "scale_fingerprint": _array_fingerprint(quantized.scales),
            "cuda_logits_fingerprint": _array_fingerprint(observed),
            "output_dtype": str(observed.dtype),
        },
        "correctness": {
            "cuda_vs_independent_quantized_oracle": cuda_vs_quantized,
            "cuda_vs_retained_serial_oracle": cuda_vs_serial,
            "independent_quantized_vs_retained_serial": quantized_vs_serial,
            "production_int8_quantization_vs_source_bf16": quantization_effect,
            "serial_argmax_token_id": serial_argmax,
            "cuda_argmax_token_id": cuda_argmax,
            "expected_generated_token_id": 11,
        },
        "benchmark": {
            "warmup_iterations": warmup,
            "retained_iterations": iterations,
            "wall": wall_metrics,
            "host_cpu": _percentiles(host),
            "queued_service_ms_per_call": queued_service_ms,
            "batched_event_service_ms_per_call": profiled_service_ms,
            "kernel_count_per_call": 1,
            "per_call_h2d_bytes": 0,
            "per_call_d2h_bytes": 0,
            "resident_weight_bytes": resident_bytes,
            "effective_weight_bandwidth_gbps": (
                resident_bytes / (profiled_service_ms / 1000.0) / 1e9
                if profiled_service_ms
                else None
            ),
            "quantization_ms": quantize_ms,
            "independent_reference_ms": reference_ms,
            "one_time_upload_ms": upload_ms,
            "gpu_utilization": None,
            "compute_utilization": None,
            "utilization_note": "decode GEMV is classified from size response and traffic",
        },
        "memory": {
            "before_upload": memory_before_upload,
            "after_upload": memory_after_upload,
            "after_buffers": memory_after_buffers,
            "resident_weight_bytes": resident_bytes,
            "expected_resident_weight_bytes": int(
                quantized.weights.nbytes + quantized.scales.nbytes
            ),
            "activation_buffer_bytes": int(activation.nbytes),
            "output_buffer_bytes": int(serial_reference.nbytes),
            "workspace_bytes": 0,
            "observed_upload_vram_delta_bytes": max(
                0, memory_before_upload["free_bytes"] - memory_after_upload["free_bytes"]
            ),
        },
        "inspection": {
            "warm_path_weight_loading": 0,
            "warm_path_h2d_bytes": 0,
            "warm_path_d2h_bytes": 0,
            "persistent_dequantized_weight_bytes": 0,
            "decision": "RETAIN" if passed else "MODIFY",
            "bottleneck": "WEIGHT_MEMORY_TRAFFIC" if passed else "LOGIT_DIVERGENCE",
        },
    }
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return {
        "schema_version": payload["schema_version"],
        "status": payload["status"],
        "output_path": str(destination),
        "output_sha256": _sha256_file(destination),
        "cuda_vs_serial_relative_l2_error": cuda_vs_serial["relative_l2_error"],
        "serial_argmax_token_id": serial_argmax,
        "cuda_argmax_token_id": cuda_argmax,
        "p50_ms": wall_metrics["p50_ms"],
        "p95_ms": wall_metrics["p95_ms"],
        "p99_ms": wall_metrics["p99_ms"],
        "resident_bytes": resident_bytes,
    }


def benchmark_real_shared_expert(
    checkpoint: Path,
    cuda_library: Path,
    activation_path: Path,
    output_path: Path,
    *,
    layer: int = 1,
    device: int = 0,
    warmup: int = 20,
    iterations: int = 100,
    cuda_architecture: str = "sm_86+c86_ptx",
) -> dict[str, Any]:
    """Certify a complete real grouped-int4 Kimi shared-expert SiTU MLP."""

    if warmup < 1 or iterations < 20:
        raise KimiCudaError("shared-expert benchmark requires warmup >=1 and iterations >=20")
    checkpoint_path = checkpoint.expanduser().resolve()
    index_path = checkpoint_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    prefix = f"language_model.model.layers.{layer}.block_sparse_moe.shared_experts"
    source_gate, gate_evidence = _load_safetensor_f32(
        checkpoint_path, index, f"{prefix}.gate_proj.weight"
    )
    source_up, up_evidence = _load_safetensor_f32(
        checkpoint_path, index, f"{prefix}.up_proj.weight"
    )
    source_down, down_evidence = _load_safetensor_f32(
        checkpoint_path, index, f"{prefix}.down_proj.weight"
    )
    if (
        source_gate.shape != (6144, 7168)
        or source_up.shape != (6144, 7168)
        or source_down.shape != (7168, 6144)
    ):
        raise KimiCudaError(
            f"unexpected shared-expert geometry {source_gate.shape}/{source_up.shape}/"
            f"{source_down.shape}"
        )
    activation_source = activation_path.expanduser().resolve()
    activation = np.ascontiguousarray(
        np.fromfile(activation_source, dtype=np.float32), dtype=np.float32
    )
    if activation.shape != (7168,):
        raise KimiCudaError(f"unexpected shared-expert activation {activation.shape}")
    quantize_start = time.perf_counter_ns()
    gate = _quantize_grouped_int4(source_gate)
    up = _quantize_grouped_int4(source_up)
    down = _quantize_grouped_int4(source_down)
    quantize_ms = (time.perf_counter_ns() - quantize_start) / 1e6
    reference_start = time.perf_counter_ns()
    gate_value = _grouped_int4_matvec(gate, activation)
    up_value = _grouped_int4_matvec(up, activation)
    sigmoid_gate = np.asarray(
        np.float32(1.0) / (np.float32(1.0) + np.exp(-gate_value)),
        dtype=np.float32,
    )
    situ = np.ascontiguousarray(
        np.float32(4.0)
        * np.tanh(gate_value / np.float32(4.0))
        * sigmoid_gate
        * np.float32(25.0)
        * np.tanh(up_value / np.float32(25.0)),
        dtype=np.float32,
    )
    reference = _grouped_int4_matvec(down, situ)
    quantized_reference_ms = (time.perf_counter_ns() - reference_start) / 1e6
    source_gate_value = np.asarray(source_gate @ activation, dtype=np.float32)
    source_up_value = np.asarray(source_up @ activation, dtype=np.float32)
    source_situ = np.ascontiguousarray(
        np.float32(4.0)
        * np.tanh(source_gate_value / np.float32(4.0))
        * (
            np.float32(1.0)
            / (np.float32(1.0) + np.exp(-source_gate_value))
        )
        * np.float32(25.0)
        * np.tanh(source_up_value / np.float32(25.0)),
        dtype=np.float32,
    )
    source_reference = np.asarray(source_down @ source_situ, dtype=np.float32)

    runtime = _CudaRuntime(cuda_library.expanduser().resolve(), device)
    memory_before_upload = runtime.mem_info()
    upload_start = time.perf_counter_ns()
    handles = (
        runtime.upload_grouped_int4(gate),
        runtime.upload_grouped_int4(up),
        runtime.upload_grouped_int4(down),
    )
    runtime.synchronize()
    upload_ms = (time.perf_counter_ns() - upload_start) / 1e6
    memory_after_upload = runtime.mem_info()
    resident_bytes = sum(runtime.tensor_bytes(handle) for handle in handles)
    input_device = runtime.allocate(activation.nbytes)
    output_device = runtime.allocate(activation.nbytes)
    observed: np.ndarray | None = None
    try:
        runtime.upload_activation(input_device, activation)
        runtime.set_telemetry("minimal")
        for _ in range(warmup):
            runtime.execute_resident(handles, output_device, input_device, 1)
            runtime.synchronize()
        memory_after_warmup = runtime.mem_info()
        wall: list[float] = []
        host: list[float] = []
        for _ in range(iterations):
            wall_start = time.perf_counter_ns()
            host_start = time.process_time_ns()
            runtime.execute_resident(handles, output_device, input_device, 1)
            runtime.synchronize()
            host.append((time.process_time_ns() - host_start) / 1e6)
            wall.append((time.perf_counter_ns() - wall_start) / 1e6)
        queued_start = time.perf_counter_ns()
        for _ in range(iterations):
            runtime.execute_resident(handles, output_device, input_device, 1)
        runtime.synchronize()
        queued_service_ms = (time.perf_counter_ns() - queued_start) / 1e6 / iterations
        runtime.profile_begin()
        for _ in range(iterations):
            runtime.execute_resident(handles, output_device, input_device, 1)
        profiled_service_ms = runtime.profile_end() / iterations
        observed = runtime.download_activation(output_device, reference.shape)
    finally:
        runtime.free(output_device)
        runtime.free(input_device)
        runtime.close()
    assert observed is not None
    execution_metrics = _numerical_metrics(reference, observed)
    quantization_metrics = _numerical_metrics(source_reference, reference)
    passed = (
        float(execution_metrics["relative_l2_error"]) <= 1e-5
        and float(execution_metrics["cosine_similarity"]) >= 0.999999
        and bool(execution_metrics["actual_finite"])
    )
    wall_metrics = _percentiles(wall)
    expected_resident_bytes = sum(
        tensor.packed.nbytes + tensor.scales.nbytes for tensor in (gate, up, down)
    )
    payload = {
        "schema_version": "experiment-014-k3-cuda-real-shared-expert-v1",
        "cycle_id": "H014-025t",
        "status": "PASS" if passed else "FAIL",
        "hypothesis": (
            "The resident Kimi SiTU expert path can reuse generic grouped-int4 CUDA "
            "GEMV for real shared-expert weights and match the production-quantized "
            "oracle within 1e-5 relative L2."
        ),
        "backend": {
            "identity": "nvidia_cuda_device_resident_grouped_int4_situ",
            "implementation_kind": "adapted CUDA",
            "cpu_fallback_allowed": False,
            "cuda_library": str(cuda_library.expanduser().resolve()),
            "cuda_library_sha256": _sha256_file(cuda_library.expanduser().resolve()),
            "compiled_architecture": cuda_architecture,
            "binary_min_compute_capability": runtime.binary_min_compute_capability,
            "binary_has_forward_ptx": runtime.binary_has_forward_ptx,
            "capability_negotiation": runtime.capability_negotiation,
            "device": _device_identity(device),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "index_sha256": _sha256_file(index_path),
            "layer": layer,
            "gate": gate_evidence,
            "up": up_evidence,
            "down": down_evidence,
        },
        "fixture": {
            "input_dimensions": [7168],
            "intermediate_dimensions": [6144],
            "output_dimensions": [7168],
            "runtime_dtype": "packed signed int4 with float32 group-64 scales",
            "activation_source": str(activation_source),
            "activation_source_sha256": _sha256_file(activation_source),
            "activation_fingerprint": _array_fingerprint(activation),
            "gate_packed_fingerprint": _array_fingerprint_streaming(gate.packed),
            "up_packed_fingerprint": _array_fingerprint_streaming(up.packed),
            "down_packed_fingerprint": _array_fingerprint_streaming(down.packed),
            "reference_output_fingerprint": _array_fingerprint(reference),
            "cuda_output_fingerprint": _array_fingerprint(observed),
            "situ_beta": 4.0,
            "situ_linear_beta": 25.0,
        },
        "correctness": {
            "cuda_vs_production_quantized_oracle": execution_metrics,
            "production_int4_quantization_vs_source_bf16": quantization_metrics,
        },
        "benchmark": {
            "warmup_iterations": warmup,
            "retained_iterations": iterations,
            "wall": wall_metrics,
            "host_cpu": _percentiles(host),
            "queued_service_ms_per_call": queued_service_ms,
            "batched_event_service_ms_per_call": profiled_service_ms,
            "kernel_count_per_call": 4,
            "per_call_h2d_bytes": 0,
            "per_call_d2h_bytes": 0,
            "resident_weight_bytes": resident_bytes,
            "effective_weight_bandwidth_gbps": (
                resident_bytes / (profiled_service_ms / 1000.0) / 1e9
                if profiled_service_ms
                else None
            ),
            "quantization_ms": quantize_ms,
            "independent_quantized_reference_ms": quantized_reference_ms,
            "one_time_upload_ms": upload_ms,
            "gpu_utilization": None,
            "compute_utilization": None,
        },
        "memory": {
            "before_upload": memory_before_upload,
            "after_upload": memory_after_upload,
            "after_warmup": memory_after_warmup,
            "resident_weight_bytes": resident_bytes,
            "expected_resident_weight_bytes": expected_resident_bytes,
            "persistent_scratch_bytes": 2 * 6144 * 4,
            "activation_buffer_bytes": int(activation.nbytes),
            "output_buffer_bytes": int(activation.nbytes),
            "persistent_dequantized_weight_bytes": 0,
            "observed_upload_vram_delta_bytes": max(
                0, memory_before_upload["free_bytes"] - memory_after_upload["free_bytes"]
            ),
        },
        "inspection": {
            "warm_path_weight_loading": 0,
            "warm_path_h2d_bytes": 0,
            "warm_path_d2h_bytes": 0,
            "decision": "RETAIN" if passed else "MODIFY",
            "bottleneck": "WEIGHT_MEMORY_TRAFFIC" if passed else "SITU_MLP_DIVERGENCE",
        },
    }
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return {
        "schema_version": payload["schema_version"],
        "status": payload["status"],
        "output_path": str(destination),
        "output_sha256": _sha256_file(destination),
        "relative_l2_error": execution_metrics["relative_l2_error"],
        "cosine_similarity": execution_metrics["cosine_similarity"],
        "p50_ms": wall_metrics["p50_ms"],
        "p95_ms": wall_metrics["p95_ms"],
        "p99_ms": wall_metrics["p99_ms"],
        "resident_bytes": resident_bytes,
    }


def benchmark_real_moe_reduction(
    checkpoint: Path,
    cuda_library: Path,
    reference_library: Path,
    activation_path: Path,
    output_path: Path,
    *,
    layer: int = 1,
    device: int = 0,
    warmup: int = 50,
    iterations: int = 500,
    cuda_architecture: str = "sm_86+c86_ptx",
) -> dict[str, Any]:
    """Reduce 16 actual routed-expert outputs with actual Kimi route weights."""

    if warmup < 1 or iterations < 20:
        raise KimiCudaError("MoE reduction benchmark requires warmup >=1 and iterations >=20")
    checkpoint_path = checkpoint.expanduser().resolve()
    index_path = checkpoint_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    prefix = f"language_model.model.layers.{layer}.block_sparse_moe"
    router, router_evidence = _load_safetensor_f32(
        checkpoint_path, index, f"{prefix}.gate.weight"
    )
    bias, bias_evidence = _load_safetensor_f32(
        checkpoint_path, index, f"{prefix}.gate.e_score_correction_bias"
    )
    latent_down_source, latent_down_evidence = _load_safetensor_f32(
        checkpoint_path, index, f"{prefix}.routed_expert_down_proj.weight"
    )
    activation_source = activation_path.expanduser().resolve()
    activation = np.ascontiguousarray(
        np.fromfile(activation_source, dtype=np.float32), dtype=np.float32
    )
    if (
        activation.shape != (7168,)
        or router.shape != (896, 7168)
        or bias.shape != (896,)
        or latent_down_source.shape != (3584, 7168)
    ):
        raise KimiCudaError(
            "unexpected MoE reduction fixture geometry: "
            f"x={activation.shape}, router={router.shape}, bias={bias.shape}, "
            f"latent_down={latent_down_source.shape}"
        )

    logits = np.asarray(router @ activation, dtype=np.float32)
    sigmoid = np.asarray(
        np.float32(1.0) / (np.float32(1.0) + np.exp(-logits)), dtype=np.float32
    )
    choice = np.asarray(sigmoid + bias, dtype=np.float32)
    selected: list[int] = []
    for _ in range(16):
        masked = choice.copy()
        if selected:
            masked[np.asarray(selected, dtype=np.int64)] = -np.inf
        selected.append(int(np.argmax(masked)))
    selected_ids = np.asarray(selected, dtype=np.int32)
    selected_weights = np.ascontiguousarray(sigmoid[selected_ids], dtype=np.float32)
    selected_weights /= (
        np.sum(selected_weights, dtype=np.float32) + np.float32(1e-20)
    )

    latent_down = _quantize_grouped_int4(latent_down_source)
    latent = np.ascontiguousarray(
        _grouped_int4_matvec(latent_down, activation)[None, :], dtype=np.float32
    )
    reference_runtime = NativeMXFP4Runtime(reference_library.expanduser().resolve())
    rows: list[np.ndarray] = []
    expert_weight_digest = hashlib.sha256()
    expert_source_shards: set[str] = set()
    expert_compute_start = time.perf_counter_ns()
    for expert_id in selected:
        expert = _load_real_expert(checkpoint_path, layer, expert_id)
        expert_weight_digest.update(struct.pack("<i", expert_id))
        for tensor in (expert.gate, expert.up, expert.down):
            expert_weight_digest.update(memoryview(tensor.packed).cast("B"))
            expert_weight_digest.update(memoryview(tensor.scales).cast("B"))
        expert_source_shards.update(expert.source_shards)
        gate = reference_runtime.matmul(latent, expert.gate)
        up = reference_runtime.matmul(latent, expert.up)
        hidden = reference_runtime.situ_glu(gate, up)
        rows.append(reference_runtime.matmul(hidden, expert.down)[0])
    expert_reference_ms = (time.perf_counter_ns() - expert_compute_start) / 1e6
    expert_rows = np.ascontiguousarray(np.stack(rows), dtype=np.float32)
    reference = np.zeros(3584, dtype=np.float32)
    for index in range(16):
        reference = np.asarray(
            reference + selected_weights[index] * expert_rows[index],
            dtype=np.float32,
        )

    runtime = _CudaRuntime(cuda_library.expanduser().resolve(), device)
    x_device = runtime.allocate(activation.nbytes)
    router_device = runtime.allocate(router.nbytes)
    bias_device = runtime.allocate(bias.nbytes)
    rows_device = runtime.allocate(expert_rows.nbytes)
    weights_device = runtime.allocate(selected_weights.nbytes)
    output_device = runtime.allocate(reference.nbytes)
    observed: np.ndarray | None = None
    repeated: np.ndarray | None = None
    try:
        runtime.upload_activation(x_device, activation)
        runtime.upload_activation(router_device, router)
        runtime.upload_activation(bias_device, bias)
        cuda_ids, cuda_weights, cuda_keff = runtime.route(
            x_device,
            router_device,
            bias_device,
            hidden=7168,
            experts=896,
            topk=16,
        )
        runtime.upload_activation(rows_device, expert_rows)
        runtime.upload_activation(weights_device, selected_weights)
        runtime.set_telemetry("minimal")
        for _ in range(warmup):
            runtime.execute_moe_reduction(
                output_device,
                rows_device,
                weights_device,
                count=16,
                dimension=3584,
            )
            runtime.synchronize()
        wall: list[float] = []
        host: list[float] = []
        for _ in range(iterations):
            wall_start = time.perf_counter_ns()
            host_start = time.process_time_ns()
            runtime.execute_moe_reduction(
                output_device,
                rows_device,
                weights_device,
                count=16,
                dimension=3584,
            )
            runtime.synchronize()
            host.append((time.process_time_ns() - host_start) / 1e6)
            wall.append((time.perf_counter_ns() - wall_start) / 1e6)
        queued_start = time.perf_counter_ns()
        for _ in range(iterations):
            runtime.execute_moe_reduction(
                output_device,
                rows_device,
                weights_device,
                count=16,
                dimension=3584,
            )
        runtime.synchronize()
        queued_service_ms = (time.perf_counter_ns() - queued_start) / 1e6 / iterations
        runtime.profile_begin()
        for _ in range(iterations):
            runtime.execute_moe_reduction(
                output_device,
                rows_device,
                weights_device,
                count=16,
                dimension=3584,
            )
        profiled_service_ms = runtime.profile_end() / iterations
        observed = runtime.download_activation(output_device, reference.shape)
        runtime.execute_moe_reduction(
            output_device,
            rows_device,
            weights_device,
            count=16,
            dimension=3584,
        )
        repeated = runtime.download_activation(output_device, reference.shape)
    finally:
        for pointer in (
            output_device,
            weights_device,
            rows_device,
            bias_device,
            router_device,
            x_device,
        ):
            runtime.free(pointer)
        runtime.close()
    assert observed is not None and repeated is not None
    correctness = _numerical_metrics(reference, observed)
    routing_weight_metrics = _numerical_metrics(selected_weights, cuda_weights)
    routing_equal = bool(np.array_equal(selected_ids, cuda_ids) and cuda_keff == 16)
    deterministic = bool(np.array_equal(observed, repeated))
    passed = (
        routing_equal
        and deterministic
        and float(correctness["relative_l2_error"]) <= 2e-6
        and float(correctness["cosine_similarity"]) >= 0.999999
        and bool(correctness["actual_finite"])
    )
    wall_metrics = _percentiles(wall)
    bytes_read = int(expert_rows.nbytes + selected_weights.nbytes)
    payload = {
        "schema_version": "experiment-014-k3-cuda-real-moe-reduction-v1",
        "cycle_id": "H014-025u",
        "status": "PASS" if passed else "FAIL",
        "hypothesis": (
            "The existing fixed-order resident CUDA reduction reproduces the "
            "weighted sum of 16 actual Kimi routed-expert outputs within 2e-6 "
            "relative L2 and is deterministic."
        ),
        "backend": {
            "identity": "nvidia_cuda_device_resident_fixed_order_moe_reduction",
            "implementation_kind": "adapted existing CUDA",
            "cpu_fallback_allowed": False,
            "cuda_library": str(cuda_library.expanduser().resolve()),
            "cuda_library_sha256": _sha256_file(cuda_library.expanduser().resolve()),
            "compiled_architecture": cuda_architecture,
            "binary_min_compute_capability": runtime.binary_min_compute_capability,
            "binary_has_forward_ptx": runtime.binary_has_forward_ptx,
            "capability_negotiation": runtime.capability_negotiation,
            "device": _device_identity(device),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "index_sha256": _sha256_file(index_path),
            "layer": layer,
            "router": router_evidence,
            "router_bias": bias_evidence,
            "latent_down": latent_down_evidence,
            "expert_source_shards": sorted(expert_source_shards),
            "selected_expert_weight_fingerprint": (
                "sha256:" + expert_weight_digest.hexdigest()
            ),
        },
        "fixture": {
            "activation_source": str(activation_source),
            "activation_source_sha256": _sha256_file(activation_source),
            "activation_fingerprint": _array_fingerprint(activation),
            "latent_fingerprint": _array_fingerprint(latent),
            "selected_expert_ids": selected_ids.tolist(),
            "selected_weights": selected_weights.tolist(),
            "expert_rows_dimensions": list(expert_rows.shape),
            "expert_rows_fingerprint": _array_fingerprint_streaming(expert_rows),
            "reference_output_fingerprint": _array_fingerprint(reference),
            "cuda_output_fingerprint": _array_fingerprint(observed),
            "dtype": "float32",
        },
        "correctness": {
            "routing_equality": routing_equal,
            "cuda_router_ids": cuda_ids.tolist(),
            "cuda_router_keff": cuda_keff,
            "cuda_router_weight_metrics": routing_weight_metrics,
            "reduction_metrics": correctness,
            "repeat_bit_exact": deterministic,
        },
        "benchmark": {
            "warmup_iterations": warmup,
            "retained_iterations": iterations,
            "wall": wall_metrics,
            "host_cpu": _percentiles(host),
            "queued_service_ms_per_call": queued_service_ms,
            "batched_event_service_ms_per_call": profiled_service_ms,
            "kernel_count_per_call": 1,
            "rows_per_call": 16,
            "dimension": 3584,
            "bytes_read_per_call": bytes_read,
            "bytes_written_per_call": int(reference.nbytes),
            "per_call_h2d_bytes": 0,
            "per_call_d2h_bytes": 0,
            "effective_io_bandwidth_gbps": (
                (bytes_read + reference.nbytes)
                / (profiled_service_ms / 1000.0)
                / 1e9
                if profiled_service_ms
                else None
            ),
            "one_time_expert_reference_ms": expert_reference_ms,
        },
        "inspection": {
            "warm_path_weight_loading": 0,
            "warm_path_h2d_bytes": 0,
            "warm_path_d2h_bytes": 0,
            "decision": "RETAIN" if passed else "MODIFY",
            "bottleneck": "PENDING_TIMING_INSPECTION" if passed else "REDUCTION_DIVERGENCE",
        },
    }
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return {
        "schema_version": payload["schema_version"],
        "status": payload["status"],
        "output_path": str(destination),
        "output_sha256": _sha256_file(destination),
        "routing_equality": routing_equal,
        "relative_l2_error": correctness["relative_l2_error"],
        "repeat_bit_exact": deterministic,
        "p50_ms": wall_metrics["p50_ms"],
        "p95_ms": wall_metrics["p95_ms"],
        "p99_ms": wall_metrics["p99_ms"],
    }


def benchmark_real_attnres(
    checkpoint: Path,
    cuda_library: Path,
    trace_path: Path,
    output_path: Path,
    *,
    device: int = 0,
    trace_step: int = 2,
    token_id: int = 11,
    warmup: int = 50,
    iterations: int = 500,
    cuda_architecture: str = "sm_86+c86_ptx",
    cycle_id: str = "H014-025v",
    baseline_path: Path | None = None,
) -> dict[str, Any]:
    """Certify final AttnRes mixing against a retained full-model decode state."""

    if warmup < 1 or iterations < 20:
        raise KimiCudaError("AttnRes benchmark requires warmup >=1 and iterations >=20")
    checkpoint_path = checkpoint.expanduser().resolve()
    index_path = checkpoint_path / "model.safetensors.index.json"
    config_path = checkpoint_path / "config.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config.get("text_config", {})
    dimension = int(text_config.get("hidden_size", 0))
    layer_count = int(text_config.get("num_hidden_layers", 0))
    block_size = int(text_config.get("attn_res_block_size", 0))
    epsilon = float(text_config.get("rms_norm_eps", 0.0))
    if (dimension, layer_count, block_size) != (7168, 93, 12) or epsilon <= 0:
        raise KimiCudaError(
            f"unexpected AttnRes configuration D={dimension}, L={layer_count}, "
            f"block={block_size}, eps={epsilon}"
        )
    table, table_evidence = _load_safetensor_raw_bf16(
        checkpoint_path, index, "language_model.model.embed_tokens.weight"
    )
    if token_id < 0 or token_id >= table.shape[0]:
        raise KimiCudaError(f"AttnRes token ID {token_id} is out of range")
    embedding_bits = np.ascontiguousarray(table[token_id], dtype=np.uint16)
    embedding = np.ascontiguousarray(
        (embedding_bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
    )
    trace_source = trace_path.expanduser().resolve()
    trace = np.memmap(trace_source, mode="r", dtype="<f4")
    rows_per_step = layer_count + 1
    if trace.size % dimension or trace.size // dimension < (trace_step + 1) * rows_per_step:
        raise KimiCudaError("retained trace does not contain the requested AttnRes step")
    trace = trace.reshape(-1, dimension)
    base = trace_step * rows_per_step
    prefix = np.ascontiguousarray(trace[base + layer_count - 1], dtype=np.float32)
    snapshot_layers = list(range(block_size - 1, layer_count - 1, block_size))
    block_residuals = np.ascontiguousarray(
        np.stack(
            [embedding]
            + [np.ascontiguousarray(trace[base + row]) for row in snapshot_layers]
        ),
        dtype=np.float32,
    )
    serial_final = np.ascontiguousarray(trace[base + layer_count], dtype=np.float32)
    attn_norm, attn_norm_evidence = _load_safetensor_f32(
        checkpoint_path, index, "language_model.model.output_attn_res_norm.weight"
    )
    attn_proj, attn_proj_evidence = _load_safetensor_f32(
        checkpoint_path, index, "language_model.model.output_attn_res_proj.weight"
    )
    final_norm, final_norm_evidence = _load_safetensor_f32(
        checkpoint_path, index, "language_model.model.norm.weight"
    )
    score_weight = np.ascontiguousarray(attn_norm * attn_proj, dtype=np.float32)
    vectors = np.ascontiguousarray(
        np.concatenate([block_residuals, prefix[None, :]], axis=0),
        dtype=np.float32,
    )
    scores = np.empty(vectors.shape[0], dtype=np.float32)
    score_weight_f64 = score_weight.astype(np.float64)
    for row, vector in enumerate(vectors):
        vector_f64 = vector.astype(np.float64)
        mean_square = np.sum(vector_f64 * vector_f64, dtype=np.float64) / dimension
        dot = np.sum(vector_f64 * score_weight_f64, dtype=np.float64)
        scores[row] = np.float32(dot / np.sqrt(mean_square + epsilon))
    exponentials = np.asarray(np.exp(scores - np.max(scores)), dtype=np.float32)
    probabilities = np.ascontiguousarray(
        exponentials / np.sum(exponentials, dtype=np.float32), dtype=np.float32
    )
    reference_mix = np.zeros(dimension, dtype=np.float32)
    for row in range(vectors.shape[0]):
        reference_mix = np.asarray(
            reference_mix + probabilities[row] * vectors[row], dtype=np.float32
        )
    reference_mean_square = np.sum(
        reference_mix.astype(np.float64) ** 2, dtype=np.float64
    ) / dimension
    inverse_rms = np.float32(1.0) / np.sqrt(
        np.float32(reference_mean_square) + np.float32(epsilon)
    )
    reference_final = np.ascontiguousarray(
        reference_mix * inverse_rms * final_norm, dtype=np.float32
    )
    reconstruction_metrics = _numerical_metrics(serial_final, reference_final)
    if float(reconstruction_metrics["relative_l2_error"]) > 2e-6:
        raise KimiCudaError(
            "retained AttnRes fixture reconstruction diverges from the serial trace"
        )

    runtime = _CudaRuntime(cuda_library.expanduser().resolve(), device)
    prefix_device = runtime.allocate(prefix.nbytes)
    residuals_device = runtime.allocate(block_residuals.nbytes)
    score_weight_device = runtime.allocate(score_weight.nbytes)
    mix_device = runtime.allocate(reference_mix.nbytes)
    final_norm_device = runtime.allocate(final_norm.nbytes)
    output_device = runtime.allocate(reference_final.nbytes)
    observed_mix: np.ndarray | None = None
    observed_final: np.ndarray | None = None
    repeated_final: np.ndarray | None = None
    try:
        runtime.upload_activation(prefix_device, prefix)
        runtime.upload_activation(residuals_device, block_residuals)
        runtime.upload_activation(score_weight_device, score_weight)
        runtime.upload_activation(final_norm_device, final_norm)

        def execute() -> None:
            runtime.execute_attnres_mix(
                mix_device,
                prefix_device,
                residuals_device,
                score_weight_device,
                block_count=block_residuals.shape[0],
                dimension=dimension,
                epsilon=epsilon,
            )
            runtime.execute_rmsnorm(
                output_device,
                mix_device,
                final_norm_device,
                batch=1,
                dimension=dimension,
                epsilon=epsilon,
            )

        for _ in range(warmup):
            execute()
            runtime.synchronize()
        wall: list[float] = []
        host: list[float] = []
        for _ in range(iterations):
            wall_start = time.perf_counter_ns()
            host_start = time.process_time_ns()
            execute()
            runtime.synchronize()
            host.append((time.process_time_ns() - host_start) / 1e6)
            wall.append((time.perf_counter_ns() - wall_start) / 1e6)
        queued_start = time.perf_counter_ns()
        for _ in range(iterations):
            execute()
        runtime.synchronize()
        queued_service_ms = (time.perf_counter_ns() - queued_start) / 1e6 / iterations
        runtime.profile_begin()
        for _ in range(iterations):
            runtime.execute_attnres_mix(
                mix_device,
                prefix_device,
                residuals_device,
                score_weight_device,
                block_count=block_residuals.shape[0],
                dimension=dimension,
                epsilon=epsilon,
            )
        mix_service_ms = runtime.profile_end() / iterations
        runtime.profile_begin()
        for _ in range(iterations):
            execute()
        combined_service_ms = runtime.profile_end() / iterations
        observed_mix = runtime.download_activation(mix_device, reference_mix.shape)
        observed_final = runtime.download_activation(output_device, reference_final.shape)
        execute()
        repeated_final = runtime.download_activation(output_device, reference_final.shape)
    finally:
        for pointer in (
            output_device,
            final_norm_device,
            mix_device,
            score_weight_device,
            residuals_device,
            prefix_device,
        ):
            runtime.free(pointer)
        runtime.close()
    assert observed_mix is not None and observed_final is not None and repeated_final is not None
    mix_metrics = _numerical_metrics(reference_mix, observed_mix)
    final_reference_metrics = _numerical_metrics(reference_final, observed_final)
    final_serial_metrics = _numerical_metrics(serial_final, observed_final)
    deterministic = bool(np.array_equal(observed_final, repeated_final))
    baseline_mix_ms: float | None = None
    mix_reduction_percent: float | None = None
    if baseline_path is not None:
        baseline_source = baseline_path.expanduser().resolve()
        baseline = json.loads(baseline_source.read_text(encoding="utf-8"))
        baseline_mix_ms = float(
            baseline["benchmark"]["attnres_batched_device_service_ms"]
        )
        mix_reduction_percent = 100.0 * (baseline_mix_ms - mix_service_ms) / baseline_mix_ms
    passed = (
        float(mix_metrics["relative_l2_error"]) <= 2e-6
        and float(final_reference_metrics["relative_l2_error"]) <= 2e-6
        and float(final_serial_metrics["relative_l2_error"]) <= 2e-6
        and float(final_serial_metrics["cosine_similarity"]) >= 0.999999
        and deterministic
        and bool(final_serial_metrics["actual_finite"])
        and (mix_reduction_percent is None or mix_reduction_percent >= 40.0)
    )
    wall_metrics = _percentiles(wall)
    payload = {
        "schema_version": "experiment-014-k3-cuda-real-attnres-v1",
        "cycle_id": cycle_id,
        "status": "PASS" if passed else "FAIL",
        "hypothesis": (
            (
                "One-warp-per-candidate AttnRes score reduction lowers mix device "
                "service at least 40% versus H014-025v while preserving the 2e-6 "
                "serial correctness and bit-repeatability gates."
            )
            if baseline_path is not None
            else (
                "A custom resident Kimi AttnRes score/softmax/mix kernel plus the "
                "existing RMSNorm primitive reproduces retained full-model decode "
                "trace row 281 within 2e-6 relative L2."
            )
        ),
        "backend": {
            "identity": "nvidia_cuda_device_resident_kimi_attnres_plus_rmsnorm",
            "implementation_kind": "custom Kimi CUDA plus generic existing CUDA",
            "cpu_fallback_allowed": False,
            "cuda_library": str(cuda_library.expanduser().resolve()),
            "cuda_library_sha256": _sha256_file(cuda_library.expanduser().resolve()),
            "compiled_architecture": cuda_architecture,
            "binary_min_compute_capability": runtime.binary_min_compute_capability,
            "binary_has_forward_ptx": runtime.binary_has_forward_ptx,
            "capability_negotiation": runtime.capability_negotiation,
            "device": _device_identity(device),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "index_sha256": _sha256_file(index_path),
            "config_sha256": _sha256_file(config_path),
            "embedding": table_evidence,
            "attn_res_norm": attn_norm_evidence,
            "attn_res_proj": attn_proj_evidence,
            "final_norm": final_norm_evidence,
        },
        "fixture": {
            "trace_source": str(trace_source),
            "trace_source_sha256": _sha256_file(trace_source),
            "trace_step": trace_step,
            "token_id": token_id,
            "trace_base_row": base,
            "prefix_row": base + layer_count - 1,
            "serial_final_row": base + layer_count,
            "snapshot_rows": [base + row for row in snapshot_layers],
            "dimensions": {
                "hidden": dimension,
                "block_residuals": list(block_residuals.shape),
            },
            "dtype": "float32",
            "prefix_fingerprint": _array_fingerprint(prefix),
            "block_residuals_fingerprint": _array_fingerprint(block_residuals),
            "score_weight_fingerprint": _array_fingerprint(score_weight),
            "reference_mix_fingerprint": _array_fingerprint(reference_mix),
            "cuda_mix_fingerprint": _array_fingerprint(observed_mix),
            "serial_final_fingerprint": _array_fingerprint(serial_final),
            "cuda_final_fingerprint": _array_fingerprint(observed_final),
            "scores": scores.tolist(),
            "probabilities": probabilities.tolist(),
        },
        "correctness": {
            "fixture_reconstruction_vs_serial": reconstruction_metrics,
            "cuda_mix_vs_independent_reference": mix_metrics,
            "cuda_final_vs_independent_reference": final_reference_metrics,
            "cuda_final_vs_retained_serial": final_serial_metrics,
            "repeat_bit_exact": deterministic,
        },
        "benchmark": {
            "warmup_iterations": warmup,
            "retained_iterations": iterations,
            "wall": wall_metrics,
            "host_cpu": _percentiles(host),
            "queued_service_ms_per_call": queued_service_ms,
            "attnres_batched_device_service_ms": mix_service_ms,
            "attnres_plus_norm_batched_device_service_ms": combined_service_ms,
            "baseline_attnres_batched_device_service_ms": baseline_mix_ms,
            "attnres_device_service_reduction_percent": mix_reduction_percent,
            "kernel_count_per_call": 2,
            "per_call_h2d_bytes": 0,
            "per_call_d2h_bytes": 0,
            "bytes_read_per_call": int(
                prefix.nbytes
                + block_residuals.nbytes
                + score_weight.nbytes
                + reference_mix.nbytes
                + final_norm.nbytes
            ),
            "bytes_written_per_call": int(reference_mix.nbytes + reference_final.nbytes),
        },
        "memory": {
            "persistent_block_residual_bytes": int(block_residuals.nbytes),
            "persistent_score_weight_bytes": int(score_weight.nbytes),
            "prefix_bytes": int(prefix.nbytes),
            "mix_bytes": int(reference_mix.nbytes),
            "final_norm_weight_bytes": int(final_norm.nbytes),
            "output_bytes": int(reference_final.nbytes),
            "workspace_bytes": 0,
        },
        "inspection": {
            "warm_path_state_reconstruction": 0,
            "warm_path_h2d_bytes": 0,
            "warm_path_d2h_bytes": 0,
            "decision": "RETAIN" if passed else "MODIFY",
            "bottleneck": (
                "PENDING_POST_PARALLEL_TIMING_INSPECTION"
                if passed and baseline_path is not None
                else "PENDING_TIMING_INSPECTION"
                if passed
                else "ATTNRES_CORRECTNESS_OR_SPEED_GATE"
            ),
        },
    }
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return {
        "schema_version": payload["schema_version"],
        "status": payload["status"],
        "output_path": str(destination),
        "output_sha256": _sha256_file(destination),
        "final_vs_serial_relative_l2_error": final_serial_metrics["relative_l2_error"],
        "repeat_bit_exact": deterministic,
        "p50_ms": wall_metrics["p50_ms"],
        "p95_ms": wall_metrics["p95_ms"],
        "p99_ms": wall_metrics["p99_ms"],
    }


def benchmark_real_kda_core(
    checkpoint: Path,
    cuda_library: Path,
    activation_path: Path,
    trace_path: Path,
    output_path: Path,
    *,
    layer: int = 1,
    device: int = 0,
    warmup: int = 20,
    iterations: int = 100,
    cuda_architecture: str = "sm_86+c86_ptx",
) -> dict[str, Any]:
    """Certify the missing stateful KDA core on real-weight-derived inputs."""

    if warmup < 1 or iterations < 20:
        raise KimiCudaError("KDA core benchmark requires warmup >=1 and iterations >=20")
    checkpoint_path = checkpoint.expanduser().resolve()
    index_path = checkpoint_path / "model.safetensors.index.json"
    config_path = checkpoint_path / "config.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config["text_config"]
    linear = text_config["linear_attn_config"]
    heads = int(linear["num_heads"])
    head_dimension = int(linear["head_dim"])
    projection = heads * head_dimension
    convolution_width = int(linear["short_conv_kernel_size"])
    gate_lower_bound = float(linear["gate_lower_bound"])
    epsilon = float(text_config["rms_norm_eps"])
    hidden = int(text_config["hidden_size"])
    if (heads, head_dimension, projection, convolution_width, hidden) != (
        96,
        128,
        12288,
        4,
        7168,
    ):
        raise KimiCudaError("unexpected Kimi KDA geometry")
    prefix = f"language_model.model.layers.{layer}"
    input_norm, input_norm_evidence = _load_safetensor_f32(
        checkpoint_path, index, f"{prefix}.input_layernorm.weight"
    )
    activation_source = activation_path.expanduser().resolve()
    x0 = np.ascontiguousarray(
        np.fromfile(activation_source, dtype=np.float32), dtype=np.float32
    )
    trace_source = trace_path.expanduser().resolve()
    trace = np.memmap(trace_source, mode="r", dtype="<f4")
    if x0.shape != (hidden,) or trace.size % hidden or trace.size // hidden < 190:
        raise KimiCudaError("invalid KDA activation/trace fixture")
    trace = trace.reshape(-1, hidden)
    raw_inputs = np.ascontiguousarray(
        np.stack([x0, trace[188], trace[189]]), dtype=np.float32
    )
    inputs = np.empty_like(raw_inputs)
    for row, value in enumerate(raw_inputs):
        mean_square = np.sum(value.astype(np.float64) ** 2, dtype=np.float64) / hidden
        inputs[row] = np.asarray(
            value
            * (np.float32(1.0) / np.sqrt(np.float32(mean_square) + np.float32(epsilon)))
            * input_norm,
            dtype=np.float32,
        )

    projected: dict[str, np.ndarray] = {}
    projection_evidence: dict[str, Any] = {}
    projection_quantize_ms: dict[str, float] = {}
    projection_reference_ms: dict[str, float] = {}
    for role in ("q", "k", "v", "g"):
        name = f"{prefix}.self_attn.{role}_proj.weight"
        source, evidence = _load_safetensor_f32(checkpoint_path, index, name)
        start = time.perf_counter_ns()
        quantized = _quantize_grouped_int4(source, retain_source=False)
        projection_quantize_ms[role] = (time.perf_counter_ns() - start) / 1e6
        del source
        start = time.perf_counter_ns()
        projected[role] = np.ascontiguousarray(
            np.stack([_grouped_int4_matvec(quantized, row) for row in inputs]),
            dtype=np.float32,
        )
        projection_reference_ms[role] = (time.perf_counter_ns() - start) / 1e6
        projection_evidence[role] = {
            "source": evidence,
            "runtime_format": "packed signed int4 with float32 group-64 scales",
            "packed_fingerprint": _array_fingerprint_streaming(quantized.packed),
            "scale_fingerprint": _array_fingerprint_streaming(quantized.scales),
        }
        del quantized

    fa, fa_evidence = _load_safetensor_f32(
        checkpoint_path, index, f"{prefix}.self_attn.f_a_proj.weight"
    )
    fb, fb_evidence = _load_safetensor_f32(
        checkpoint_path, index, f"{prefix}.self_attn.f_b_proj.weight"
    )
    bp, bp_evidence = _load_safetensor_f32(
        checkpoint_path, index, f"{prefix}.self_attn.b_proj.weight"
    )
    if fa.shape != (head_dimension, hidden) or fb.shape != (projection, head_dimension):
        raise KimiCudaError("unexpected KDA decay projection geometry")
    decay = np.ascontiguousarray((inputs @ fa.T) @ fb.T, dtype=np.float32)
    beta_raw = np.ascontiguousarray(inputs @ bp.T, dtype=np.float32)
    parameter_names = {
        "conv_q": "q_conv1d.weight",
        "conv_k": "k_conv1d.weight",
        "conv_v": "v_conv1d.weight",
        "dt": "dt_bias",
        "a_log": "A_log",
        "output_norm": "o_norm.weight",
    }
    parameters: dict[str, np.ndarray] = {}
    parameter_evidence: dict[str, Any] = {}
    for role, suffix in parameter_names.items():
        values, evidence = _load_safetensor_f32(
            checkpoint_path, index, f"{prefix}.self_attn.{suffix}"
        )
        parameters[role] = np.ascontiguousarray(values, dtype=np.float32)
        parameter_evidence[role] = evidence
    a = np.ascontiguousarray(np.exp(parameters["a_log"][:heads]), dtype=np.float32)
    reference_start = time.perf_counter_ns()
    reference_output, reference_state, reference_windows = _kda_core_reference(
        projected["q"],
        projected["k"],
        projected["v"],
        projected["g"],
        decay,
        beta_raw,
        parameters["conv_q"],
        parameters["conv_k"],
        parameters["conv_v"],
        parameters["dt"],
        a,
        parameters["output_norm"],
        gate_lower_bound=gate_lower_bound,
        epsilon=epsilon,
    )
    reference_ms = (time.perf_counter_ns() - reference_start) / 1e6

    runtime = _CudaRuntime(cuda_library.expanduser().resolve(), device)
    arrays = {
        "q": projected["q"],
        "k": projected["k"],
        "v": projected["v"],
        "gate": projected["g"],
        "decay": decay,
        "beta": beta_raw,
        "conv_q": parameters["conv_q"],
        "conv_k": parameters["conv_k"],
        "conv_v": parameters["conv_v"],
        "dt": parameters["dt"],
        "a": a,
        "output_norm": parameters["output_norm"],
    }
    pointers = {name: runtime.allocate(value.nbytes) for name, value in arrays.items()}
    state_zero = np.zeros(reference_state.shape, dtype=np.float32)
    window_zero = np.zeros((projection, convolution_width), dtype=np.float32)
    pointers["state"] = runtime.allocate(state_zero.nbytes)
    for role in ("window_q", "window_k", "window_v"):
        pointers[role] = runtime.allocate(window_zero.nbytes)
    pointers["output"] = runtime.allocate(projection * 4)
    memory_after_buffers = runtime.mem_info()
    for name, value in arrays.items():
        runtime.upload_activation(pointers[name], value)

    def row_pointer(pointer: ctypes.c_void_p, row: int, width: int) -> ctypes.c_void_p:
        if pointer.value is None:
            raise KimiCudaError("null KDA device pointer")
        return ctypes.c_void_p(int(pointer.value) + row * width * 4)

    def reset_state() -> None:
        runtime.upload_activation(pointers["state"], state_zero)
        for role in ("window_q", "window_k", "window_v"):
            runtime.upload_activation(pointers[role], window_zero)

    def execute(step: int) -> None:
        runtime.execute_kda_core(
            pointers["output"],
            row_pointer(pointers["q"], step, projection),
            row_pointer(pointers["k"], step, projection),
            row_pointer(pointers["v"], step, projection),
            row_pointer(pointers["gate"], step, projection),
            row_pointer(pointers["decay"], step, projection),
            row_pointer(pointers["beta"], step, heads),
            pointers["conv_q"],
            pointers["conv_k"],
            pointers["conv_v"],
            pointers["window_q"],
            pointers["window_k"],
            pointers["window_v"],
            pointers["state"],
            pointers["dt"],
            pointers["a"],
            pointers["output_norm"],
            heads=heads,
            head_dimension=head_dimension,
            convolution_width=convolution_width,
            gate_lower_bound=gate_lower_bound,
            epsilon=epsilon,
        )

    observed_rows: list[np.ndarray] = []
    repeated_rows: list[np.ndarray] = []
    observed_state: np.ndarray | None = None
    observed_windows: np.ndarray | None = None
    final_warm_state: np.ndarray | None = None
    try:
        reset_state()
        for step in range(3):
            execute(step)
            runtime.synchronize()
            observed_rows.append(
                runtime.download_activation(pointers["output"], (projection,))
            )
        observed_state = runtime.download_activation(
            pointers["state"], reference_state.shape
        )
        observed_windows = np.stack(
            [
                runtime.download_activation(
                    pointers[role], (projection, convolution_width)
                )
                for role in ("window_q", "window_k", "window_v")
            ]
        )
        reset_state()
        for step in range(3):
            execute(step)
            runtime.synchronize()
            repeated_rows.append(
                runtime.download_activation(pointers["output"], (projection,))
            )
        for call in range(warmup):
            execute(call % 3)
            runtime.synchronize()
        wall: list[float] = []
        host: list[float] = []
        for call in range(iterations):
            wall_start = time.perf_counter_ns()
            host_start = time.process_time_ns()
            execute(call % 3)
            runtime.synchronize()
            host.append((time.process_time_ns() - host_start) / 1e6)
            wall.append((time.perf_counter_ns() - wall_start) / 1e6)
        queued_start = time.perf_counter_ns()
        for call in range(iterations):
            execute(call % 3)
        runtime.synchronize()
        queued_service_ms = (time.perf_counter_ns() - queued_start) / 1e6 / iterations
        runtime.profile_begin()
        for call in range(iterations):
            execute(call % 3)
        device_service_ms = runtime.profile_end() / iterations
        final_warm_state = runtime.download_activation(
            pointers["state"], reference_state.shape
        )
    finally:
        for pointer in reversed(list(pointers.values())):
            runtime.free(pointer)
        runtime.close()
    assert observed_state is not None and observed_windows is not None
    assert final_warm_state is not None
    observed_output = np.ascontiguousarray(np.stack(observed_rows), dtype=np.float32)
    repeated_output = np.ascontiguousarray(np.stack(repeated_rows), dtype=np.float32)
    output_metrics = [
        _numerical_metrics(reference_output[row], observed_output[row])
        for row in range(3)
    ]
    state_metrics = _numerical_metrics(reference_state, observed_state)
    window_metrics = _numerical_metrics(reference_windows, observed_windows)
    repeat_bit_exact = bool(np.array_equal(observed_output, repeated_output))
    maximum_output_relative_l2 = max(
        float(metric["relative_l2_error"]) for metric in output_metrics
    )
    state_advanced = bool(np.any(observed_state != 0.0))
    final_state_finite = bool(np.all(np.isfinite(final_warm_state)))
    passed = (
        maximum_output_relative_l2 <= 2e-5
        and float(state_metrics["relative_l2_error"]) <= 2e-5
        and float(window_metrics["relative_l2_error"]) <= 2e-6
        and repeat_bit_exact
        and state_advanced
        and final_state_finite
    )
    wall_metrics = _percentiles(wall)
    state_bytes = int(reference_state.nbytes)
    payload = {
        "schema_version": "experiment-014-k3-cuda-real-kda-core-v1",
        "cycle_id": "H014-025y",
        "status": "PASS" if passed else "FAIL",
        "hypothesis": (
            "One custom resident CUDA kernel can cover only Kimi's missing "
            "depthwise-convolution/recurrent KDA core and match a three-step "
            "independent production-quantized oracle within 2e-5 relative L2."
        ),
        "backend": {
            "identity": "nvidia_cuda_device_resident_kimi_kda_core",
            "implementation_kind": "custom Kimi CUDA core; projections excluded",
            "cpu_fallback_allowed": False,
            "cuda_library": str(cuda_library.expanduser().resolve()),
            "cuda_library_sha256": _sha256_file(cuda_library.expanduser().resolve()),
            "compiled_architecture": cuda_architecture,
            "binary_min_compute_capability": runtime.binary_min_compute_capability,
            "binary_has_forward_ptx": runtime.binary_has_forward_ptx,
            "capability_negotiation": runtime.capability_negotiation,
            "device": _device_identity(device),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "index_sha256": _sha256_file(index_path),
            "config_sha256": _sha256_file(config_path),
            "layer": layer,
            "input_norm": input_norm_evidence,
            "quantized_input_projections": projection_evidence,
            "f_a": fa_evidence,
            "f_b": fb_evidence,
            "b": bp_evidence,
            "state_parameters": parameter_evidence,
        },
        "fixture": {
            "steps": 3,
            "raw_input_sources": [
                {"path": str(activation_source), "row": 0},
                {"path": str(trace_source), "row": 188},
                {"path": str(trace_source), "row": 189},
            ],
            "activation_source_sha256": _sha256_file(activation_source),
            "trace_source_sha256": _sha256_file(trace_source),
            "raw_input_fingerprint": _array_fingerprint(raw_inputs),
            "normalized_input_fingerprint": _array_fingerprint(inputs),
            "projected_input_fingerprints": {
                key: _array_fingerprint(value) for key, value in projected.items()
            },
            "decay_fingerprint": _array_fingerprint(decay),
            "beta_fingerprint": _array_fingerprint(beta_raw),
            "reference_output_fingerprints": [
                _array_fingerprint(row) for row in reference_output
            ],
            "cuda_output_fingerprints": [
                _array_fingerprint(row) for row in observed_output
            ],
            "reference_state_fingerprint": _array_fingerprint_streaming(reference_state),
            "cuda_state_fingerprint": _array_fingerprint_streaming(observed_state),
            "dtype": "float32",
        },
        "correctness": {
            "per_step_output_metrics": output_metrics,
            "maximum_output_relative_l2_error": maximum_output_relative_l2,
            "state_metrics": state_metrics,
            "convolution_window_metrics": window_metrics,
            "repeat_bit_exact": repeat_bit_exact,
            "state_advanced_from_zero": state_advanced,
            "post_benchmark_state_finite": final_state_finite,
        },
        "benchmark": {
            "warmup_iterations": warmup,
            "retained_iterations": iterations,
            "wall": wall_metrics,
            "host_cpu": _percentiles(host),
            "queued_service_ms_per_call": queued_service_ms,
            "batched_device_service_ms_per_call": device_service_ms,
            "kernel_count_per_call": 1,
            "per_call_h2d_bytes": 0,
            "per_call_d2h_bytes": 0,
            "recurrent_state_bytes": state_bytes,
            "convolution_state_bytes": int(reference_windows.nbytes),
            "approximate_state_bytes_read_written_per_call": 4 * state_bytes,
            "approximate_state_bandwidth_gbps": (
                4 * state_bytes / (device_service_ms / 1000.0) / 1e9
                if device_service_ms
                else None
            ),
            "independent_reference_ms": reference_ms,
            "projection_quantize_ms": projection_quantize_ms,
            "projection_reference_ms": projection_reference_ms,
        },
        "memory": {
            "after_buffers": memory_after_buffers,
            "persistent_recurrent_state_bytes": state_bytes,
            "persistent_convolution_state_bytes": int(reference_windows.nbytes),
            "persistent_parameter_bytes": int(
                sum(value.nbytes for value in parameters.values()) + a.nbytes
            ),
            "projected_fixture_bytes": int(
                sum(value.nbytes for value in projected.values())
                + decay.nbytes
                + beta_raw.nbytes
            ),
            "output_bytes": projection * 4,
        },
        "inspection": {
            "warm_path_state_reset": 0,
            "warm_path_h2d_bytes": 0,
            "warm_path_d2h_bytes": 0,
            "decision": "RETAIN" if passed else "MODIFY",
            "bottleneck": "PENDING_TIMING_INSPECTION" if passed else "KDA_CORE_DIVERGENCE",
            "scope": "KDA core only; eight generic projection kernels remain to be wired",
        },
    }
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return {
        "schema_version": payload["schema_version"],
        "status": payload["status"],
        "output_path": str(destination),
        "output_sha256": _sha256_file(destination),
        "maximum_output_relative_l2_error": maximum_output_relative_l2,
        "state_relative_l2_error": state_metrics["relative_l2_error"],
        "repeat_bit_exact": repeat_bit_exact,
        "p50_ms": wall_metrics["p50_ms"],
        "p95_ms": wall_metrics["p95_ms"],
        "p99_ms": wall_metrics["p99_ms"],
    }


def benchmark_real_kda_stage(
    checkpoint: Path,
    cuda_library: Path,
    activation_path: Path,
    trace_path: Path,
    output_path: Path,
    *,
    layer: int = 1,
    device: int = 0,
    warmup: int = 20,
    iterations: int = 100,
    cuda_architecture: str = "sm_86+c86_ptx",
    cycle_id: str = "H014-025z",
) -> dict[str, Any]:
    """Certify a complete resident KDA attention stage including all projections."""

    if warmup < 1 or iterations < 20:
        raise KimiCudaError("KDA stage benchmark requires warmup >=1 and iterations >=20")
    checkpoint_path = checkpoint.expanduser().resolve()
    index_path = checkpoint_path / "model.safetensors.index.json"
    config_path = checkpoint_path / "config.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config["text_config"]
    linear = text_config["linear_attn_config"]
    heads = int(linear["num_heads"])
    head_dimension = int(linear["head_dim"])
    projection = heads * head_dimension
    convolution_width = int(linear["short_conv_kernel_size"])
    gate_lower_bound = float(linear["gate_lower_bound"])
    epsilon = float(text_config["rms_norm_eps"])
    hidden = int(text_config["hidden_size"])
    if (heads, head_dimension, projection, convolution_width, hidden) != (
        96,
        128,
        12288,
        4,
        7168,
    ):
        raise KimiCudaError("unexpected complete KDA geometry")
    prefix = f"language_model.model.layers.{layer}"
    input_norm, input_norm_evidence = _load_safetensor_f32(
        checkpoint_path, index, f"{prefix}.input_layernorm.weight"
    )
    activation_source = activation_path.expanduser().resolve()
    x0 = np.ascontiguousarray(
        np.fromfile(activation_source, dtype=np.float32), dtype=np.float32
    )
    trace_source = trace_path.expanduser().resolve()
    trace = np.memmap(trace_source, mode="r", dtype="<f4")
    if x0.shape != (hidden,) or trace.size % hidden or trace.size // hidden < 190:
        raise KimiCudaError("invalid complete KDA activation/trace fixture")
    trace = trace.reshape(-1, hidden)
    raw_inputs = np.ascontiguousarray(
        np.stack([x0, trace[188], trace[189]]), dtype=np.float32
    )
    inputs = np.empty_like(raw_inputs)
    for row, value in enumerate(raw_inputs):
        mean_square = np.sum(value.astype(np.float64) ** 2, dtype=np.float64) / hidden
        inputs[row] = np.asarray(
            value
            * (np.float32(1.0) / np.sqrt(np.float32(mean_square) + np.float32(epsilon)))
            * input_norm,
            dtype=np.float32,
        )

    quantized: dict[str, _GroupedInt4Tensor] = {}
    quantized_evidence: dict[str, Any] = {}
    quantize_ms: dict[str, float] = {}
    projection_names = {
        "q": f"{prefix}.self_attn.q_proj.weight",
        "k": f"{prefix}.self_attn.k_proj.weight",
        "v": f"{prefix}.self_attn.v_proj.weight",
        "gate": f"{prefix}.self_attn.g_proj.weight",
        "output": f"{prefix}.self_attn.o_proj.weight",
    }
    for role, name in projection_names.items():
        source, evidence = _load_safetensor_f32(checkpoint_path, index, name)
        start = time.perf_counter_ns()
        tensor = _quantize_grouped_int4(source, retain_source=False)
        quantize_ms[role] = (time.perf_counter_ns() - start) / 1e6
        del source
        quantized[role] = tensor
        quantized_evidence[role] = {
            "source": evidence,
            "runtime_format": "packed signed int4 with float32 group-64 scales",
            "packed_fingerprint": _array_fingerprint_streaming(tensor.packed),
            "scale_fingerprint": _array_fingerprint_streaming(tensor.scales),
        }
    projected = {
        role: np.ascontiguousarray(
            np.stack([_grouped_int4_matvec(quantized[role], row) for row in inputs]),
            dtype=np.float32,
        )
        for role in ("q", "k", "v", "gate")
    }
    fa, fa_evidence = _load_safetensor_f32(
        checkpoint_path, index, f"{prefix}.self_attn.f_a_proj.weight"
    )
    fb, fb_evidence = _load_safetensor_f32(
        checkpoint_path, index, f"{prefix}.self_attn.f_b_proj.weight"
    )
    bp, bp_evidence = _load_safetensor_f32(
        checkpoint_path, index, f"{prefix}.self_attn.b_proj.weight"
    )
    decay = np.ascontiguousarray((inputs @ fa.T) @ fb.T, dtype=np.float32)
    beta_raw = np.ascontiguousarray(inputs @ bp.T, dtype=np.float32)
    parameter_names = {
        "conv_q": "q_conv1d.weight",
        "conv_k": "k_conv1d.weight",
        "conv_v": "v_conv1d.weight",
        "dt": "dt_bias",
        "a_log": "A_log",
        "output_norm": "o_norm.weight",
    }
    parameters: dict[str, np.ndarray] = {}
    parameter_evidence: dict[str, Any] = {}
    for role, suffix in parameter_names.items():
        values, evidence = _load_safetensor_f32(
            checkpoint_path, index, f"{prefix}.self_attn.{suffix}"
        )
        parameters[role] = np.ascontiguousarray(values, dtype=np.float32)
        parameter_evidence[role] = evidence
    a = np.ascontiguousarray(np.exp(parameters["a_log"][:heads]), dtype=np.float32)
    reference_start = time.perf_counter_ns()
    reference_core, reference_state, reference_windows = _kda_core_reference(
        projected["q"],
        projected["k"],
        projected["v"],
        projected["gate"],
        decay,
        beta_raw,
        parameters["conv_q"],
        parameters["conv_k"],
        parameters["conv_v"],
        parameters["dt"],
        a,
        parameters["output_norm"],
        gate_lower_bound=gate_lower_bound,
        epsilon=epsilon,
    )
    reference_output = np.ascontiguousarray(
        np.stack(
            [
                _grouped_int4_matvec(quantized["output"], row)
                for row in reference_core
            ]
        ),
        dtype=np.float32,
    )
    reference_ms = (time.perf_counter_ns() - reference_start) / 1e6

    runtime = _CudaRuntime(cuda_library.expanduser().resolve(), device)
    runtime.set_telemetry("minimal")
    handles = {
        role: runtime.upload_grouped_int4(tensor)
        for role, tensor in quantized.items()
    }
    handles["fa"] = runtime.upload_float32(fa)
    handles["fb"] = runtime.upload_float32(fb)
    handles["bp"] = runtime.upload_float32(bp)
    resident_weight_bytes = sum(runtime.tensor_bytes(handle) for handle in handles.values())
    arrays = {
        "inputs": inputs,
        "conv_q": parameters["conv_q"],
        "conv_k": parameters["conv_k"],
        "conv_v": parameters["conv_v"],
        "dt": parameters["dt"],
        "a": a,
        "output_norm": parameters["output_norm"],
    }
    pointers = {name: runtime.allocate(value.nbytes) for name, value in arrays.items()}
    for name, value in arrays.items():
        runtime.upload_activation(pointers[name], value)
    buffer_dimensions = {
        "q": projection,
        "k": projection,
        "v": projection,
        "gate": projection,
        "t1": head_dimension,
        "decay": projection,
        "beta": heads,
        "core": projection,
        "output": hidden,
    }
    for name, width in buffer_dimensions.items():
        pointers[name] = runtime.allocate(width * 4)
    state_zero = np.zeros(reference_state.shape, dtype=np.float32)
    window_zero = np.zeros((projection, convolution_width), dtype=np.float32)
    pointers["state"] = runtime.allocate(state_zero.nbytes)
    for role in ("window_q", "window_k", "window_v"):
        pointers[role] = runtime.allocate(window_zero.nbytes)
    memory_after_load = runtime.mem_info()

    def row_pointer(pointer: ctypes.c_void_p, row: int, width: int) -> ctypes.c_void_p:
        if pointer.value is None:
            raise KimiCudaError("null complete KDA pointer")
        return ctypes.c_void_p(int(pointer.value) + row * width * 4)

    def reset_state() -> None:
        runtime.upload_activation(pointers["state"], state_zero)
        for role in ("window_q", "window_k", "window_v"):
            runtime.upload_activation(pointers[role], window_zero)

    def execute_input_projections(step: int) -> None:
        source = row_pointer(pointers["inputs"], step, hidden)
        for role in ("q", "k", "v", "gate"):
            runtime.execute_dense(handles[role], pointers[role], source, 1)
        runtime.execute_dense(handles["fa"], pointers["t1"], source, 1)
        runtime.execute_dense(handles["fb"], pointers["decay"], pointers["t1"], 1)
        runtime.execute_dense(handles["bp"], pointers["beta"], source, 1)

    def execute_core() -> None:
        runtime.execute_kda_core(
            pointers["core"],
            pointers["q"],
            pointers["k"],
            pointers["v"],
            pointers["gate"],
            pointers["decay"],
            pointers["beta"],
            pointers["conv_q"],
            pointers["conv_k"],
            pointers["conv_v"],
            pointers["window_q"],
            pointers["window_k"],
            pointers["window_v"],
            pointers["state"],
            pointers["dt"],
            pointers["a"],
            pointers["output_norm"],
            heads=heads,
            head_dimension=head_dimension,
            convolution_width=convolution_width,
            gate_lower_bound=gate_lower_bound,
            epsilon=epsilon,
        )

    def execute(step: int) -> None:
        execute_input_projections(step)
        execute_core()
        runtime.execute_dense(
            handles["output"], pointers["output"], pointers["core"], 1
        )

    observed_rows: list[np.ndarray] = []
    repeated_rows: list[np.ndarray] = []
    observed_state: np.ndarray | None = None
    final_state: np.ndarray | None = None
    try:
        reset_state()
        for step in range(3):
            execute(step)
            runtime.synchronize()
            observed_rows.append(runtime.download_activation(pointers["output"], (hidden,)))
        observed_state = runtime.download_activation(pointers["state"], reference_state.shape)
        reset_state()
        for step in range(3):
            execute(step)
            runtime.synchronize()
            repeated_rows.append(runtime.download_activation(pointers["output"], (hidden,)))
        for call in range(warmup):
            execute(call % 3)
            runtime.synchronize()
        wall: list[float] = []
        host: list[float] = []
        for call in range(iterations):
            wall_start = time.perf_counter_ns()
            host_start = time.process_time_ns()
            execute(call % 3)
            runtime.synchronize()
            host.append((time.process_time_ns() - host_start) / 1e6)
            wall.append((time.perf_counter_ns() - wall_start) / 1e6)
        queued_start = time.perf_counter_ns()
        for call in range(iterations):
            execute(call % 3)
        runtime.synchronize()
        queued_service_ms = (time.perf_counter_ns() - queued_start) / 1e6 / iterations
        runtime.profile_begin()
        for call in range(iterations):
            execute(call % 3)
        full_device_ms = runtime.profile_end() / iterations
        runtime.profile_begin()
        for call in range(iterations):
            execute_input_projections(call % 3)
        input_projection_ms = runtime.profile_end() / iterations
        runtime.profile_begin()
        for _ in range(iterations):
            execute_core()
        core_ms = runtime.profile_end() / iterations
        runtime.profile_begin()
        for _ in range(iterations):
            runtime.execute_dense(
                handles["output"], pointers["output"], pointers["core"], 1
            )
        output_projection_ms = runtime.profile_end() / iterations
        individual_projection_ms: dict[str, float] = {}
        profile_source = row_pointer(pointers["inputs"], 0, hidden)
        for role in ("q", "k", "v", "gate"):
            runtime.profile_begin()
            for _ in range(iterations):
                runtime.execute_dense(handles[role], pointers[role], profile_source, 1)
            individual_projection_ms[role] = runtime.profile_end() / iterations
        runtime.profile_begin()
        for _ in range(iterations):
            runtime.execute_dense(handles["fa"], pointers["t1"], profile_source, 1)
        individual_projection_ms["f_a"] = runtime.profile_end() / iterations
        runtime.profile_begin()
        for _ in range(iterations):
            runtime.execute_dense(handles["fb"], pointers["decay"], pointers["t1"], 1)
        individual_projection_ms["f_b"] = runtime.profile_end() / iterations
        runtime.profile_begin()
        for _ in range(iterations):
            runtime.execute_dense(handles["bp"], pointers["beta"], profile_source, 1)
        individual_projection_ms["b"] = runtime.profile_end() / iterations
        individual_projection_ms["output"] = output_projection_ms
        final_state = runtime.download_activation(pointers["state"], reference_state.shape)
    finally:
        for pointer in reversed(list(pointers.values())):
            runtime.free(pointer)
        runtime.close()
    assert observed_state is not None and final_state is not None
    observed_output = np.ascontiguousarray(np.stack(observed_rows), dtype=np.float32)
    repeated_output = np.ascontiguousarray(np.stack(repeated_rows), dtype=np.float32)
    output_metrics = [
        _numerical_metrics(reference_output[row], observed_output[row])
        for row in range(3)
    ]
    state_metrics = _numerical_metrics(reference_state, observed_state)
    maximum_output_relative_l2 = max(
        float(metric["relative_l2_error"]) for metric in output_metrics
    )
    repeat_bit_exact = bool(np.array_equal(observed_output, repeated_output))
    final_state_finite = bool(np.all(np.isfinite(final_state)))
    individual_attributed_ms = sum(individual_projection_ms.values()) + core_ms
    attribution_percent = 100.0 * individual_attributed_ms / full_device_ms
    passed = (
        maximum_output_relative_l2 <= 3e-5
        and float(state_metrics["relative_l2_error"]) <= 3e-5
        and repeat_bit_exact
        and final_state_finite
        and (cycle_id != "H014-025aa" or attribution_percent >= 90.0)
    )
    wall_metrics = _percentiles(wall)
    phase_sum_ms = input_projection_ms + core_ms + output_projection_ms
    payload = {
        "schema_version": "experiment-014-k3-cuda-real-kda-stage-v1",
        "cycle_id": cycle_id,
        "status": "PASS" if passed else "FAIL",
        "hypothesis": (
            (
                "Individual unchanged projection profiles plus the KDA core "
                "attribute at least 90% of complete KDA device service."
            )
            if cycle_id == "H014-025aa"
            else (
                "The eight existing resident CUDA GEMV transforms plus the retained "
                "KDA state core reproduce a complete real Kimi KDA attention stage "
                "over three stateful steps within 3e-5 relative L2."
            )
        ),
        "backend": {
            "identity": "nvidia_cuda_device_resident_complete_kimi_kda",
            "implementation_kind": "generic existing CUDA projections plus custom Kimi core",
            "cpu_fallback_allowed": False,
            "cuda_library": str(cuda_library.expanduser().resolve()),
            "cuda_library_sha256": _sha256_file(cuda_library.expanduser().resolve()),
            "compiled_architecture": cuda_architecture,
            "binary_min_compute_capability": runtime.binary_min_compute_capability,
            "binary_has_forward_ptx": runtime.binary_has_forward_ptx,
            "capability_negotiation": runtime.capability_negotiation,
            "device": _device_identity(device),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "index_sha256": _sha256_file(index_path),
            "config_sha256": _sha256_file(config_path),
            "layer": layer,
            "input_norm": input_norm_evidence,
            "quantized_projections": quantized_evidence,
            "f_a": fa_evidence,
            "f_b": fb_evidence,
            "b": bp_evidence,
            "state_parameters": parameter_evidence,
        },
        "fixture": {
            "steps": 3,
            "input_fingerprint": _array_fingerprint(inputs),
            "reference_core_fingerprint": _array_fingerprint(reference_core),
            "reference_output_fingerprints": [
                _array_fingerprint(row) for row in reference_output
            ],
            "cuda_output_fingerprints": [
                _array_fingerprint(row) for row in observed_output
            ],
            "reference_state_fingerprint": _array_fingerprint_streaming(reference_state),
            "cuda_state_fingerprint": _array_fingerprint_streaming(observed_state),
            "dtype": "float32 activations; int4-g64 and FP32 weights",
        },
        "correctness": {
            "per_step_output_metrics": output_metrics,
            "maximum_output_relative_l2_error": maximum_output_relative_l2,
            "state_metrics": state_metrics,
            "repeat_bit_exact": repeat_bit_exact,
            "post_benchmark_state_finite": final_state_finite,
        },
        "benchmark": {
            "warmup_iterations": warmup,
            "retained_iterations": iterations,
            "wall": wall_metrics,
            "host_cpu": _percentiles(host),
            "queued_service_ms_per_call": queued_service_ms,
            "batched_device_service_ms_per_call": full_device_ms,
            "phase_device_ms_per_call": {
                "seven_input_projections": input_projection_ms,
                "kda_state_core": core_ms,
                "output_projection": output_projection_ms,
                "phase_sum": phase_sum_ms,
            },
            "individual_projection_device_ms_per_call": individual_projection_ms,
            "individual_projection_plus_core_attributed_ms": individual_attributed_ms,
            "individual_profile_attribution_percent": attribution_percent,
            "kernel_count_per_call": 9,
            "per_call_h2d_bytes": 0,
            "per_call_d2h_bytes": 0,
            "resident_projection_weight_bytes": resident_weight_bytes,
            "effective_projection_weight_bandwidth_gbps": (
                resident_weight_bytes / (full_device_ms / 1000.0) / 1e9
                if full_device_ms
                else None
            ),
            "independent_reference_ms": reference_ms,
            "quantization_ms": quantize_ms,
        },
        "memory": {
            "after_load": memory_after_load,
            "resident_projection_weight_bytes": resident_weight_bytes,
            "persistent_recurrent_state_bytes": int(reference_state.nbytes),
            "persistent_convolution_state_bytes": int(reference_windows.nbytes),
            "persistent_parameter_bytes": int(
                sum(value.nbytes for value in parameters.values()) + a.nbytes
            ),
            "persistent_activation_buffer_bytes": int(
                sum(width * 4 for width in buffer_dimensions.values())
            ),
        },
        "inspection": {
            "warm_path_weight_loading": 0,
            "warm_path_state_reset": 0,
            "warm_path_h2d_bytes": 0,
            "warm_path_d2h_bytes": 0,
            "decision": "RETAIN" if passed else "MODIFY",
            "bottleneck": "PENDING_PHASE_INSPECTION" if passed else "COMPLETE_KDA_DIVERGENCE",
        },
    }
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return {
        "schema_version": payload["schema_version"],
        "status": payload["status"],
        "output_path": str(destination),
        "output_sha256": _sha256_file(destination),
        "maximum_output_relative_l2_error": maximum_output_relative_l2,
        "state_relative_l2_error": state_metrics["relative_l2_error"],
        "repeat_bit_exact": repeat_bit_exact,
        "p50_ms": wall_metrics["p50_ms"],
        "p95_ms": wall_metrics["p95_ms"],
        "p99_ms": wall_metrics["p99_ms"],
    }


def benchmark_real_mla_stage(
    checkpoint: Path,
    cuda_library: Path,
    trace_path: Path,
    output_path: Path,
    *,
    layer: int = 3,
    device: int = 0,
    warmup: int = 20,
    iterations: int = 100,
    cuda_architecture: str = "sm_86+c86_ptx",
    cycle_id: str = "H014-025ac",
) -> dict[str, Any]:
    """Certify a complete resident, stateful Gated MLA attention stage."""

    if warmup < 1 or iterations < 20:
        raise KimiCudaError("MLA stage benchmark requires warmup >=1 and iterations >=20")
    checkpoint_path = checkpoint.expanduser().resolve()
    index_path = checkpoint_path / "model.safetensors.index.json"
    config_path = checkpoint_path / "config.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config["text_config"]
    hidden = int(text_config["hidden_size"])
    heads = int(text_config["num_attention_heads"])
    query_lora = int(text_config["q_lora_rank"])
    kv_lora = int(text_config["kv_lora_rank"])
    query_nope = int(text_config["qk_nope_head_dim"])
    query_rope = int(text_config["qk_rope_head_dim"])
    value_dimension = int(text_config["v_head_dim"])
    layer_count = int(text_config["num_hidden_layers"])
    epsilon = float(text_config["rms_norm_eps"])
    attention_scale = float(1.0 / np.sqrt(np.float32(query_nope + query_rope)))
    full_attention = {
        int(value) - 1
        for value in text_config["linear_attn_config"]["full_attn_layers"]
    }
    if layer not in full_attention:
        raise KimiCudaError(f"layer {layer} is not a Gated MLA layer")
    if (
        hidden,
        heads,
        query_lora,
        kv_lora,
        query_nope,
        query_rope,
        value_dimension,
    ) != (7168, 96, 1536, 512, 128, 64, 128):
        raise KimiCudaError("unexpected Kimi K3 Gated MLA geometry")

    prefix = f"language_model.model.layers.{layer}"
    input_norm, input_norm_evidence = _load_safetensor_f32(
        checkpoint_path, index, f"{prefix}.input_layernorm.weight"
    )
    query_norm, query_norm_evidence = _load_safetensor_f32(
        checkpoint_path, index, f"{prefix}.self_attn.q_a_layernorm.weight"
    )
    kv_norm, kv_norm_evidence = _load_safetensor_f32(
        checkpoint_path, index, f"{prefix}.self_attn.kv_a_layernorm.weight"
    )
    trace_source = trace_path.expanduser().resolve()
    trace = np.memmap(trace_source, mode="r", dtype="<f4")
    rows_per_token = layer_count + 1
    required_rows = 3 * rows_per_token
    if trace.size % hidden or trace.size // hidden < required_rows or layer < 1:
        raise KimiCudaError("full-oracle trace does not contain three complete token rows")
    trace = trace.reshape(-1, hidden)
    trace_rows = [token * rows_per_token + layer - 1 for token in range(3)]
    raw_inputs = np.ascontiguousarray(trace[trace_rows], dtype=np.float32)
    inputs = np.ascontiguousarray(
        np.stack(
            [_rmsnorm_reference(row, input_norm, epsilon) for row in raw_inputs]
        ),
        dtype=np.float32,
    )

    names = {
        "query_a": f"{prefix}.self_attn.q_a_proj.weight",
        "query_b": f"{prefix}.self_attn.q_b_proj.weight",
        "kv_a": f"{prefix}.self_attn.kv_a_proj_with_mqa.weight",
        "kv_b": f"{prefix}.self_attn.kv_b_proj.weight",
        "gate": f"{prefix}.self_attn.g_proj.weight",
        "output": f"{prefix}.self_attn.o_proj.weight",
    }
    tensors: dict[str, _QuantizedInt8Tensor] = {}
    tensor_evidence: dict[str, Any] = {}
    quantize_ms: dict[str, float] = {}
    for role, name in names.items():
        source, evidence = _load_safetensor_raw_bf16(checkpoint_path, index, name)
        start = time.perf_counter_ns()
        tensor = _quantize_bf16_rows_int8(source)
        quantize_ms[role] = (time.perf_counter_ns() - start) / 1e6
        tensors[role] = tensor
        tensor_evidence[role] = {
            "source": evidence,
            "runtime_format": "signed int8 with float32 per-row scale",
            "quantized_weight_fingerprint": _array_fingerprint_streaming(tensor.weights),
            "quantized_scale_fingerprint": _array_fingerprint_streaming(tensor.scales),
        }
        del source

    reference_start = time.perf_counter_ns()
    reference_output, reference_latent, reference_rope = _mla_stage_reference(
        inputs,
        tensors,
        query_norm,
        kv_norm,
        heads=heads,
        query_nope=query_nope,
        query_rope=query_rope,
        value_dimension=value_dimension,
        kv_lora=kv_lora,
        attention_scale=attention_scale,
        epsilon=epsilon,
    )
    reference_ms = (time.perf_counter_ns() - reference_start) / 1e6

    runtime = _CudaRuntime(cuda_library.expanduser().resolve(), device)
    runtime.set_telemetry("minimal")
    handles = {role: runtime.upload_int8(tensor) for role, tensor in tensors.items()}
    resident_weight_bytes = sum(runtime.tensor_bytes(handle) for handle in handles.values())
    arrays = {
        "inputs": inputs,
        "query_norm": query_norm,
        "kv_norm": kv_norm,
    }
    pointers = {name: runtime.allocate(value.nbytes) for name, value in arrays.items()}
    for name, value in arrays.items():
        runtime.upload_activation(pointers[name], value)
    widths = {
        "query_low_rank": query_lora,
        "query": heads * (query_nope + query_rope),
        "compressed_kv": kv_lora + query_rope,
        "gate": heads * value_dimension,
        "context": heads * value_dimension,
        "output": hidden,
    }
    for name, width in widths.items():
        pointers[name] = runtime.allocate(width * 4)
    latent_zero = np.zeros((3, kv_lora), dtype=np.float32)
    rope_zero = np.zeros((3, query_rope), dtype=np.float32)
    pointers["latent_cache"] = runtime.allocate(latent_zero.nbytes)
    pointers["rope_cache"] = runtime.allocate(rope_zero.nbytes)
    memory_after_load = runtime.mem_info()

    def row_pointer(pointer: ctypes.c_void_p, row: int, width: int) -> ctypes.c_void_p:
        if pointer.value is None:
            raise KimiCudaError("null complete MLA pointer")
        return ctypes.c_void_p(int(pointer.value) + row * width * 4)

    def reset_cache() -> None:
        runtime.upload_activation(pointers["latent_cache"], latent_zero)
        runtime.upload_activation(pointers["rope_cache"], rope_zero)

    def execute_projections(step: int) -> None:
        source = row_pointer(pointers["inputs"], step, hidden)
        runtime.execute_dense(
            handles["query_a"], pointers["query_low_rank"], source, 1
        )
        runtime.execute_rmsnorm(
            pointers["query_low_rank"],
            pointers["query_low_rank"],
            pointers["query_norm"],
            batch=1,
            dimension=query_lora,
            epsilon=epsilon,
        )
        runtime.execute_dense(
            handles["query_b"], pointers["query"], pointers["query_low_rank"], 1
        )
        runtime.execute_dense(
            handles["kv_a"], pointers["compressed_kv"], source, 1
        )
        runtime.execute_dense(handles["gate"], pointers["gate"], source, 1)

    def execute_cache_append(step: int) -> None:
        runtime.execute_mla_cache_append(
            row_pointer(pointers["latent_cache"], step, kv_lora),
            row_pointer(pointers["rope_cache"], step, query_rope),
            pointers["compressed_kv"],
            pointers["kv_norm"],
            kv_lora=kv_lora,
            rope_dimension=query_rope,
            epsilon=epsilon,
        )

    def execute_attention(step: int) -> None:
        runtime.execute_mla_absorb(
            handles["kv_b"],
            pointers["context"],
            pointers["query"],
            pointers["latent_cache"],
            pointers["rope_cache"],
            heads=heads,
            query_nope=query_nope,
            query_rope=query_rope,
            value_dimension=value_dimension,
            kv_lora=kv_lora,
            context_length=step + 1,
            attention_scale=attention_scale,
        )
        runtime.execute_mla_gate(
            pointers["context"], pointers["gate"], heads * value_dimension
        )

    def execute_output() -> None:
        runtime.execute_dense(
            handles["output"], pointers["output"], pointers["context"], 1
        )

    def execute(step: int) -> None:
        execute_projections(step)
        execute_cache_append(step)
        execute_attention(step)
        execute_output()

    observed_rows: list[np.ndarray] = []
    repeated_rows: list[np.ndarray] = []
    cuda_compressed_kv_rows: list[np.ndarray] = []
    observed_latent: np.ndarray | None = None
    observed_rope: np.ndarray | None = None
    try:
        reset_cache()
        for step in range(3):
            execute(step)
            runtime.synchronize()
            observed_rows.append(runtime.download_activation(pointers["output"], (hidden,)))
            cuda_compressed_kv_rows.append(
                runtime.download_activation(
                    pointers["compressed_kv"], (kv_lora + query_rope,)
                )
            )
        observed_latent = runtime.download_activation(
            pointers["latent_cache"], reference_latent.shape
        )
        observed_rope = runtime.download_activation(
            pointers["rope_cache"], reference_rope.shape
        )
        reset_cache()
        for step in range(3):
            execute(step)
            runtime.synchronize()
            repeated_rows.append(runtime.download_activation(pointers["output"], (hidden,)))
        for call in range(warmup):
            execute(call % 3)
            runtime.synchronize()
        wall: list[float] = []
        host: list[float] = []
        for call in range(iterations):
            wall_start = time.perf_counter_ns()
            host_start = time.process_time_ns()
            execute(call % 3)
            runtime.synchronize()
            host.append((time.process_time_ns() - host_start) / 1e6)
            wall.append((time.perf_counter_ns() - wall_start) / 1e6)
        queued_start = time.perf_counter_ns()
        for call in range(iterations):
            execute(call % 3)
        runtime.synchronize()
        queued_service_ms = (time.perf_counter_ns() - queued_start) / 1e6 / iterations
        runtime.profile_begin()
        for call in range(iterations):
            execute(call % 3)
        full_device_ms = runtime.profile_end() / iterations
        runtime.profile_begin()
        for call in range(iterations):
            execute_projections(call % 3)
        projection_ms = runtime.profile_end() / iterations
        runtime.profile_begin()
        for call in range(iterations):
            execute_cache_append(call % 3)
        cache_ms = runtime.profile_end() / iterations
        runtime.profile_begin()
        for call in range(iterations):
            execute_attention(call % 3)
        attention_gate_ms = runtime.profile_end() / iterations
        runtime.profile_begin()
        for _ in range(iterations):
            execute_output()
        output_ms = runtime.profile_end() / iterations
    finally:
        for pointer in reversed(list(pointers.values())):
            runtime.free(pointer)
        runtime.close()
    assert observed_latent is not None and observed_rope is not None
    observed_output = np.ascontiguousarray(np.stack(observed_rows), dtype=np.float32)
    repeated_output = np.ascontiguousarray(np.stack(repeated_rows), dtype=np.float32)
    cuda_projected_rope = np.ascontiguousarray(
        np.stack(cuda_compressed_kv_rows)[:, kv_lora:], dtype=np.float32
    )
    output_metrics = [
        _numerical_metrics(reference_output[row], observed_output[row])
        for row in range(3)
    ]
    latent_metrics = _numerical_metrics(reference_latent, observed_latent)
    rope_metrics = _numerical_metrics(reference_rope, observed_rope)
    maximum_output_relative_l2 = max(
        float(metric["relative_l2_error"]) for metric in output_metrics
    )
    repeat_bit_exact = bool(np.array_equal(observed_output, repeated_output))
    rope_copy_bit_exact = bool(np.array_equal(cuda_projected_rope, observed_rope))
    passed = (
        maximum_output_relative_l2 <= 3e-5
        and float(latent_metrics["relative_l2_error"]) <= 3e-5
        and float(rope_metrics["relative_l2_error"]) <= 3e-5
        and rope_copy_bit_exact
        and repeat_bit_exact
    )
    wall_metrics = _percentiles(wall)
    phase_sum_ms = projection_ms + cache_ms + attention_gate_ms + output_ms
    payload = {
        "schema_version": "experiment-014-k3-cuda-real-mla-stage-v1",
        "cycle_id": cycle_id,
        "status": "PASS" if passed else "FAIL",
        "hypothesis": (
            "The retained separate-gate MLA path keeps the NoPE cache bit-exact to its "
            "CUDA projection while the complete stage and both caches remain within "
            "3e-5 relative L2 of the independent real-weight oracle."
        ),
        "backend": {
            "identity": "nvidia_cuda_device_resident_complete_kimi_gated_mla",
            "implementation_kind": (
                "generic existing CUDA projections/norm/attention plus adapted cache "
                "wiring and a separately testable sigmoid elementwise gate"
            ),
            "cpu_fallback_allowed": False,
            "cuda_library": str(cuda_library.expanduser().resolve()),
            "cuda_library_sha256": _sha256_file(cuda_library.expanduser().resolve()),
            "compiled_architecture": cuda_architecture,
            "binary_min_compute_capability": runtime.binary_min_compute_capability,
            "binary_has_forward_ptx": runtime.binary_has_forward_ptx,
            "capability_negotiation": runtime.capability_negotiation,
            "device": _device_identity(device),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "index_sha256": _sha256_file(index_path),
            "config_sha256": _sha256_file(config_path),
            "layer": layer,
            "input_norm": input_norm_evidence,
            "query_norm": query_norm_evidence,
            "kv_norm": kv_norm_evidence,
            "quantized_projections": tensor_evidence,
        },
        "dimensions": {
            "hidden": hidden,
            "heads": heads,
            "query_lora": query_lora,
            "kv_lora": kv_lora,
            "query_nope_per_head": query_nope,
            "query_rope_per_head": query_rope,
            "value_per_head": value_dimension,
            "attention_scale": attention_scale,
        },
        "fixture": {
            "steps": 3,
            "trace_path": str(trace_source),
            "trace_sha256": _sha256_file(trace_source),
            "trace_rows": trace_rows,
            "trace_row_semantics": "real serial layer-2 hidden outputs for three tokens",
            "raw_input_fingerprint": _array_fingerprint(raw_inputs),
            "normalized_input_fingerprint": _array_fingerprint(inputs),
            "reference_output_fingerprints": [
                _array_fingerprint(row) for row in reference_output
            ],
            "cuda_output_fingerprints": [
                _array_fingerprint(row) for row in observed_output
            ],
            "reference_latent_cache_fingerprint": _array_fingerprint(reference_latent),
            "cuda_latent_cache_fingerprint": _array_fingerprint(observed_latent),
            "reference_rope_cache_fingerprint": _array_fingerprint(reference_rope),
            "cuda_rope_cache_fingerprint": _array_fingerprint(observed_rope),
            "cuda_projected_rope_fingerprint": _array_fingerprint(cuda_projected_rope),
            "dtype": "float32 activations/state; per-row int8 projection weights",
        },
        "correctness": {
            "per_step_output_metrics": output_metrics,
            "maximum_output_relative_l2_error": maximum_output_relative_l2,
            "latent_cache_metrics": latent_metrics,
            "rope_cache_metrics": rope_metrics,
            "rope_cache_copy_from_cuda_projection_bit_exact": rope_copy_bit_exact,
            "cpu_projection_to_cuda_cache_bit_exact_expected": False,
            "repeat_output_bit_exact": repeat_bit_exact,
        },
        "benchmark": {
            "warmup_iterations": warmup,
            "retained_iterations": iterations,
            "wall": wall_metrics,
            "host_cpu": _percentiles(host),
            "queued_service_ms_per_call": queued_service_ms,
            "batched_device_service_ms_per_call": full_device_ms,
            "phase_device_ms_per_call": {
                "five_projection_and_query_norm_kernels": projection_ms,
                "latent_norm_and_nope_cache_append": cache_ms,
                "absorb_attention_and_sigmoid_gate": attention_gate_ms,
                "output_projection": output_ms,
                "phase_sum": phase_sum_ms,
            },
            "kernel_count_per_call": 9,
            "device_copy_count_per_call": 1,
            "per_call_h2d_bytes": 0,
            "per_call_d2h_bytes": 0,
            "resident_projection_weight_bytes": resident_weight_bytes,
            "effective_projection_weight_bandwidth_gbps": (
                resident_weight_bytes / (full_device_ms / 1000.0) / 1e9
                if full_device_ms
                else None
            ),
            "independent_reference_ms": reference_ms,
            "quantization_ms": quantize_ms,
        },
        "memory": {
            "after_load": memory_after_load,
            "resident_projection_weight_bytes": resident_weight_bytes,
            "persistent_cache_bytes_per_token_per_layer_per_request": (
                kv_lora + query_rope
            )
            * 4,
            "fixture_persistent_cache_bytes": latent_zero.nbytes + rope_zero.nbytes,
            "persistent_norm_parameter_bytes": query_norm.nbytes + kv_norm.nbytes,
            "persistent_activation_buffer_bytes": sum(widths.values()) * 4,
        },
        "inspection": {
            "warm_path_weight_loading": 0,
            "warm_path_state_reset": 0,
            "warm_path_h2d_bytes": 0,
            "warm_path_d2h_bytes": 0,
            "decision": "RETAIN" if passed else "MODIFY",
            "bottleneck": "PENDING_PHASE_INSPECTION" if passed else "MLA_NUMERICAL_DIVERGENCE",
        },
    }
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return {
        "schema_version": payload["schema_version"],
        "status": payload["status"],
        "output_path": str(destination),
        "output_sha256": _sha256_file(destination),
        "maximum_output_relative_l2_error": maximum_output_relative_l2,
        "latent_cache_relative_l2_error": latent_metrics["relative_l2_error"],
        "rope_cache_copy_bit_exact": rope_copy_bit_exact,
        "repeat_output_bit_exact": repeat_bit_exact,
        "p50_ms": wall_metrics["p50_ms"],
        "p95_ms": wall_metrics["p95_ms"],
        "p99_ms": wall_metrics["p99_ms"],
    }


__all__ = [
    "KimiCudaError",
    "benchmark_real_attnres",
    "benchmark_real_dense_projection",
    "benchmark_real_embedding",
    "benchmark_real_final_norm",
    "benchmark_real_kda_core",
    "benchmark_real_kda_stage",
    "benchmark_real_lm_head",
    "benchmark_real_mla_stage",
    "benchmark_real_moe_reduction",
    "benchmark_real_mxfp4_expert",
    "benchmark_real_router",
    "benchmark_real_shared_expert",
]
