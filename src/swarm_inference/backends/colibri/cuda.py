"""Fail-closed ctypes proof for Colibri's native CUDA expert kernel."""

from __future__ import annotations

import argparse
import contextlib
import csv
import ctypes
import hashlib
import io
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.experiments.experiment_010.colibri_expert_bank import (
    scan_safetensors,
)


class ColibriCudaError(RuntimeError):
    """The requested Colibri CUDA path did not execute correctly."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gpu_snapshot(device: int) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(device),
                "--query-gpu=index,name,uuid,compute_cap,memory.used,memory.total,"
                "utilization.gpu,utilization.memory,power.draw,power.limit,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ColibriCudaError(f"NVIDIA device query failed: {error}") from error
    if result.returncode:
        raise ColibriCudaError(f"NVIDIA device query failed: {result.stderr.strip()}")
    rows = list(csv.reader(io.StringIO(result.stdout)))
    if not rows or len(rows[0]) != 11:
        raise ColibriCudaError("NVIDIA device query returned an invalid row")
    row = [item.strip() for item in rows[0]]
    return {
        "index": int(row[0]),
        "name": row[1],
        "uuid": row[2],
        "compute_capability": row[3],
        "memory_used_bytes": int(float(row[4]) * 1024 * 1024),
        "memory_total_bytes": int(float(row[5]) * 1024 * 1024),
        "gpu_utilization_percent": float(row[6]),
        "memory_utilization_percent": float(row[7]),
        "power_watts": float(row[8]),
        "power_limit_watts": float(row[9]),
        "temperature_celsius": float(row[10]),
    }


def _configure_library(library: Any) -> None:
    pointer = ctypes.c_void_p
    library.coli_cuda_init.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    library.coli_cuda_init.restype = ctypes.c_int
    library.coli_cuda_shutdown.argtypes = []
    library.coli_cuda_shutdown.restype = None
    library.coli_cuda_device_count.argtypes = []
    library.coli_cuda_device_count.restype = ctypes.c_int
    library.coli_cuda_mem_info.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    library.coli_cuda_mem_info.restype = ctypes.c_int
    library.coli_cuda_tensor_upload.argtypes = [
        ctypes.POINTER(pointer),
        pointer,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    library.coli_cuda_tensor_upload.restype = ctypes.c_int
    library.coli_cuda_expert_mlp.argtypes = [
        pointer,
        pointer,
        pointer,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    library.coli_cuda_expert_mlp.restype = ctypes.c_int
    library.coli_cuda_tensor_free.argtypes = [pointer]
    library.coli_cuda_tensor_free.restype = None
    library.coli_cuda_stats.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    library.coli_cuda_stats.restype = None
    library.coli_cuda_group_stats.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    library.coli_cuda_group_stats.restype = None


def _read_tensor_payload(location: Any) -> bytes:
    with location.path.open("rb") as handle:
        handle.seek(location.absolute_offset)
        payload = bytes(handle.read(location.nbytes))
    if len(payload) != location.nbytes:
        raise ColibriCudaError(f"short native tensor read for {location.name}")
    return payload


def _load_cuda_library(path: Path, device: int) -> tuple[Any, list[Any]]:
    handles: list[Any] = []
    candidates = [path.parent]
    cuda_root = os.environ.get("CUDA_PATH")
    if cuda_root:
        candidates.append(Path(cuda_root) / "bin")
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        for directory in candidates:
            if directory.is_dir():
                handles.append(os.add_dll_directory(str(directory)))
    loader = ctypes.WinDLL if os.name == "nt" else ctypes.CDLL
    try:
        library = loader(str(path))
    except OSError as error:
        for handle in handles:
            with contextlib.suppress(Exception):
                handle.close()
        raise ColibriCudaError(f"Colibri CUDA DLL load failed: {error}") from error
    _configure_library(library)
    devices = (ctypes.c_int * 1)(device)
    if library.coli_cuda_init(devices, 1) != 1 or library.coli_cuda_device_count() < 1:
        for handle in handles:
            with contextlib.suppress(Exception):
                handle.close()
        raise ColibriCudaError("Colibri CUDA initialization failed")
    return library, handles


def _cpu_expert(
    activation: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
) -> np.ndarray:
    gate_projection = activation @ gate.T
    up_projection = activation @ up.T
    sigmoid = 1.0 / (1.0 + np.exp(-gate_projection))
    return np.asarray((gate_projection * sigmoid * up_projection) @ down.T, dtype=np.float32)


def run_colibri_cuda_kernel_proof(
    cuda_dll: str | Path,
    *,
    device: int = 0,
    output_path: str | Path | None = None,
    seed: int = 1010,
    latent_dimension: int = 64,
    intermediate_dimension: int = 96,
    batch_rows: int = 3,
    maximum_absolute_error: float = 2e-4,
    maximum_relative_l2_error: float = 2e-4,
) -> dict[str, Any]:
    """Load, execute, validate, and fingerprint the real Colibri CUDA DLL."""

    path = Path(cuda_dll).expanduser().resolve()
    proof: dict[str, Any] = {
        "schema_version": "experiment-010-colibri-cuda-proof-v1",
        "timestamp_ns": time.time_ns(),
        "cuda_dll": str(path),
        "cuda_dll_sha256": _sha256(path) if path.is_file() else None,
        "requested_device": device,
        "dll_loaded": False,
        "device_detected": False,
        "kernel_executed": False,
        "correctness_passed": False,
        "failure": None,
    }
    library: Any | None = None
    tensors = [ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()]
    dll_handles: list[Any] = []
    initialized = False
    try:
        if not path.is_file():
            raise ColibriCudaError(f"requested Colibri CUDA DLL is missing: {path}")
        gpu_before = _gpu_snapshot(device)
        proof["gpu_before"] = gpu_before
        proof["device_detected"] = True
        candidates = [path.parent]
        cuda_root = os.environ.get("CUDA_PATH")
        if cuda_root:
            candidates.append(Path(cuda_root) / "bin")
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            for directory in candidates:
                if directory.is_dir():
                    dll_handles.append(os.add_dll_directory(str(directory)))
        loader = ctypes.WinDLL if os.name == "nt" else ctypes.CDLL
        try:
            library = loader(str(path))
        except OSError as error:
            raise ColibriCudaError(f"Colibri CUDA DLL load failed: {error}") from error
        proof["dll_loaded"] = True
        _configure_library(library)
        devices = (ctypes.c_int * 1)(device)
        if library.coli_cuda_init(devices, 1) != 1:
            raise ColibriCudaError("Colibri CUDA initialization failed")
        initialized = True
        if library.coli_cuda_device_count() < 1:
            raise ColibriCudaError("Colibri CUDA initialized without a device")
        generator = np.random.default_rng(seed)
        activation = np.ascontiguousarray(
            generator.normal(0, 0.05, (batch_rows, latent_dimension)), dtype=np.float32
        )
        gate = np.ascontiguousarray(
            generator.normal(0, 0.05, (intermediate_dimension, latent_dimension)),
            dtype=np.float32,
        )
        up = np.ascontiguousarray(
            generator.normal(0, 0.05, (intermediate_dimension, latent_dimension)),
            dtype=np.float32,
        )
        down = np.ascontiguousarray(
            generator.normal(0, 0.05, (latent_dimension, intermediate_dimension)),
            dtype=np.float32,
        )
        definitions = (
            (tensors[0], gate, latent_dimension, intermediate_dimension),
            (tensors[1], up, latent_dimension, intermediate_dimension),
            (tensors[2], down, intermediate_dimension, latent_dimension),
        )
        upload_started = time.perf_counter_ns()
        for tensor, weights, inputs, outputs in definitions:
            result = library.coli_cuda_tensor_upload(
                ctypes.byref(tensor),
                weights.ctypes.data_as(ctypes.c_void_p),
                None,
                0,
                inputs,
                outputs,
                device,
            )
            if result != 1 or not tensor.value:
                raise ColibriCudaError("Colibri CUDA FP32 tensor upload failed")
        proof["weight_upload_ns"] = time.perf_counter_ns() - upload_started
        tensor_count, tensor_bytes = ctypes.c_size_t(), ctypes.c_size_t()
        library.coli_cuda_stats(device, ctypes.byref(tensor_count), ctypes.byref(tensor_bytes))
        proof["resident_tensor_count"] = tensor_count.value
        proof["resident_tensor_bytes"] = tensor_bytes.value
        proof["gpu_expert_resident_bytes"] = tensor_bytes.value
        output = np.empty((batch_rows, latent_dimension), dtype=np.float32)
        kernel_started = time.perf_counter_ns()
        executed = library.coli_cuda_expert_mlp(
            tensors[0],
            tensors[1],
            tensors[2],
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            activation.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            batch_rows,
        )
        proof["wall_kernel_ns"] = time.perf_counter_ns() - kernel_started
        if executed != 1:
            raise ColibriCudaError(
                "Colibri CUDA expert kernel launch or synchronization failed "
                f"(GPU compute capability {gpu_before['compute_capability']})"
            )
        proof["kernel_executed"] = True
        calls, experts, rows = ctypes.c_uint64(), ctypes.c_uint64(), ctypes.c_uint64()
        h2d, kernel, d2h = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
        library.coli_cuda_group_stats(
            ctypes.byref(calls),
            ctypes.byref(experts),
            ctypes.byref(rows),
            ctypes.byref(h2d),
            ctypes.byref(kernel),
            ctypes.byref(d2h),
        )
        reference = _cpu_expert(activation, gate, up, down)
        difference = np.asarray(output, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
        max_error = float(np.max(np.abs(difference)))
        relative_l2 = float(
            np.linalg.norm(difference.ravel())
            / max(float(np.linalg.norm(reference.astype(np.float64).ravel())), 1e-30)
        )
        proof.update(
            {
                "correctness_passed": max_error <= maximum_absolute_error
                and relative_l2 <= maximum_relative_l2_error,
                "maximum_absolute_error": max_error,
                "mean_absolute_error": float(np.mean(np.abs(difference))),
                "relative_l2_error": relative_l2,
                "cpu_output_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
                "cuda_output_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
                "input_bytes": int(activation.nbytes),
                "output_bytes": int(output.nbytes),
                "host_to_device_bytes": int(
                    activation.nbytes + gate.nbytes + up.nbytes + down.nbytes
                ),
                "device_to_host_bytes": int(output.nbytes),
                "group_statistics": {
                    "calls": calls.value,
                    "experts": experts.value,
                    "rows": rows.value,
                    "h2d_ms": h2d.value,
                    "kernel_ms": kernel.value,
                    "d2h_ms": d2h.value,
                },
                "tensor_shapes": {
                    "activation": list(activation.shape),
                    "gate": list(gate.shape),
                    "up": list(up.shape),
                    "down": list(down.shape),
                    "output": list(output.shape),
                },
            }
        )
        if not proof["correctness_passed"]:
            raise ColibriCudaError(
                f"Colibri CUDA result failed CPU equivalence: max={max_error}, rel_l2={relative_l2}"
            )
        proof["gpu_after"] = _gpu_snapshot(device)
    except Exception as error:
        proof["failure"] = f"{type(error).__name__}: {error}"
        if output_path is not None:
            destination = Path(output_path).expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
        if isinstance(error, ColibriCudaError):
            raise
        raise ColibriCudaError(str(error)) from error
    finally:
        if library is not None:
            for tensor in tensors:
                if tensor.value:
                    with contextlib.suppress(Exception):
                        library.coli_cuda_tensor_free(tensor)
            if initialized:
                with contextlib.suppress(Exception):
                    library.coli_cuda_shutdown()
        for handle in dll_handles:
            with contextlib.suppress(Exception):
                handle.close()
    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    return proof


def run_colibri_real_olmoe_cuda_expert(
    cuda_dll: str | Path,
    model_path: str | Path,
    *,
    layer_id: int,
    expert_id: int,
    device: int = 0,
    output_path: str | Path | None = None,
    seed: int = 1010,
    batch_rows: int = 1,
    maximum_absolute_error: float = 5e-3,
    maximum_relative_l2_error: float = 5e-4,
) -> dict[str, Any]:
    """Execute exact native Colibri OLMoE int8 bytes on the CUDA ABI."""

    dll = Path(cuda_dll).expanduser().resolve()
    model = Path(model_path).expanduser().resolve()
    proof: dict[str, Any] = {
        "schema_version": "experiment-010-real-olmoe-cuda-expert-v1",
        "timestamp_ns": time.time_ns(),
        "cuda_dll": str(dll),
        "cuda_dll_sha256": _sha256(dll) if dll.is_file() else None,
        "model_path": str(model),
        "layer_id": layer_id,
        "expert_id": expert_id,
        "requested_device": device,
        "quantization": "native_colibri_merged_int8_f32_row_scales",
        "kernel_executed": False,
        "native_tensor_bytes_used": False,
        "nonzero_vram_residency": False,
        "no_silent_cpu_fallback": False,
        "correctness_passed": False,
        "failure": None,
    }
    library: Any | None = None
    handles: list[Any] = []
    cuda_tensors = [ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()]
    initialized = False
    try:
        if not dll.is_file():
            raise ColibriCudaError(f"requested Colibri CUDA DLL is missing: {dll}")
        config_path = model / "config.json"
        if not config_path.is_file():
            raise ColibriCudaError(f"missing Colibri model config: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        hidden = int(config["hidden_size"])
        intermediate = int(config["intermediate_size"])
        if not 0 <= layer_id < int(config["num_hidden_layers"]):
            raise ColibriCudaError("real CUDA expert layer is outside model dimensions")
        if not 0 <= expert_id < int(config["num_experts"]):
            raise ColibriCudaError("real CUDA expert ID is outside model dimensions")
        tensors = scan_safetensors(model)
        prefix = f"model.layers.{layer_id}.mlp.experts.{expert_id}"
        weight_location = tensors.get(f"{prefix}.merged_weight")
        scale_location = tensors.get(f"{prefix}.qs")
        expected_weight_bytes = 3 * hidden * intermediate
        expected_scale_count = 2 * intermediate + hidden
        if (
            weight_location is None
            or weight_location.dtype != "I8"
            or weight_location.nbytes != expected_weight_bytes
            or scale_location is None
            or scale_location.dtype != "F32"
            or scale_location.nbytes != expected_scale_count * 4
        ):
            raise ColibriCudaError("real OLMoE expert tensor shape/dtype contract failed")
        weight_bytes = _read_tensor_payload(weight_location)
        scale_bytes = _read_tensor_payload(scale_location)
        merged = np.frombuffer(weight_bytes, dtype=np.int8)
        scales = np.frombuffer(scale_bytes, dtype="<f4")
        matrix = hidden * intermediate
        gate_q = np.ascontiguousarray(merged[:matrix].reshape(intermediate, hidden))
        up_q = np.ascontiguousarray(merged[matrix : 2 * matrix].reshape(intermediate, hidden))
        down_q = np.ascontiguousarray(merged[2 * matrix :].reshape(hidden, intermediate))
        gate_scale = np.ascontiguousarray(scales[:intermediate], dtype=np.float32)
        up_scale = np.ascontiguousarray(scales[intermediate : 2 * intermediate], dtype=np.float32)
        down_scale = np.ascontiguousarray(scales[2 * intermediate :], dtype=np.float32)
        proof.update(
            {
                "hidden_dimension": hidden,
                "intermediate_dimension": intermediate,
                "batch_rows": batch_rows,
                "merged_weight_sha256": hashlib.sha256(weight_bytes).hexdigest(),
                "row_scales_sha256": hashlib.sha256(scale_bytes).hexdigest(),
                "merged_weight_source_file": weight_location.path.name,
                "merged_weight_source_offset": weight_location.absolute_offset,
                "row_scales_source_file": scale_location.path.name,
                "row_scales_source_offset": scale_location.absolute_offset,
                "native_weight_bytes": len(weight_bytes),
                "native_scale_bytes": len(scale_bytes),
                "native_tensor_bytes_used": True,
            }
        )
        gpu_before = _gpu_snapshot(device)
        proof["gpu_before"] = gpu_before
        library, handles = _load_cuda_library(dll, device)
        initialized = True
        free_before, total_before = ctypes.c_size_t(), ctypes.c_size_t()
        if (
            library.coli_cuda_mem_info(
                device, ctypes.byref(free_before), ctypes.byref(total_before)
            )
            != 1
        ):
            raise ColibriCudaError("CUDA memory query failed before native upload")
        definitions = (
            (cuda_tensors[0], gate_q, gate_scale, hidden, intermediate),
            (cuda_tensors[1], up_q, up_scale, hidden, intermediate),
            (cuda_tensors[2], down_q, down_scale, intermediate, hidden),
        )
        upload_started = time.perf_counter_ns()
        for tensor, weights, row_scales, inputs, outputs in definitions:
            if (
                library.coli_cuda_tensor_upload(
                    ctypes.byref(tensor),
                    weights.ctypes.data_as(ctypes.c_void_p),
                    row_scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    1,
                    inputs,
                    outputs,
                    device,
                )
                != 1
            ):
                raise ColibriCudaError("native int8 Colibri expert upload failed")
        proof["weight_upload_ns"] = time.perf_counter_ns() - upload_started
        resident_count, resident_bytes = ctypes.c_size_t(), ctypes.c_size_t()
        library.coli_cuda_stats(device, ctypes.byref(resident_count), ctypes.byref(resident_bytes))
        free_after, total_after = ctypes.c_size_t(), ctypes.c_size_t()
        library.coli_cuda_mem_info(device, ctypes.byref(free_after), ctypes.byref(total_after))
        proof.update(
            {
                "resident_tensor_count": resident_count.value,
                "resident_tensor_bytes": resident_bytes.value,
                "gpu_expert_resident_bytes": resident_bytes.value,
                "cuda_free_bytes_before_upload": free_before.value,
                "cuda_free_bytes_after_upload": free_after.value,
                "cuda_total_bytes": total_after.value,
                "nonzero_vram_residency": resident_count.value == 3
                and resident_bytes.value > 0
                and free_after.value < free_before.value,
            }
        )
        generator = np.random.default_rng(seed)
        activation = np.ascontiguousarray(
            generator.normal(0.0, 0.05, (batch_rows, hidden)), dtype=np.float32
        )
        output = np.empty((batch_rows, hidden), dtype=np.float32)
        kernel_started = time.perf_counter_ns()
        executed = library.coli_cuda_expert_mlp(
            cuda_tensors[0],
            cuda_tensors[1],
            cuda_tensors[2],
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            activation.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            batch_rows,
        )
        proof["wall_kernel_ns"] = time.perf_counter_ns() - kernel_started
        if executed != 1:
            raise ColibriCudaError("real native int8 CUDA expert execution failed")
        proof["kernel_executed"] = True
        calls, experts, rows = ctypes.c_uint64(), ctypes.c_uint64(), ctypes.c_uint64()
        h2d, kernel, d2h = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
        library.coli_cuda_group_stats(
            ctypes.byref(calls),
            ctypes.byref(experts),
            ctypes.byref(rows),
            ctypes.byref(h2d),
            ctypes.byref(kernel),
            ctypes.byref(d2h),
        )
        cpu_started = time.perf_counter_ns()
        gate = gate_q.astype(np.float32) * gate_scale[:, None]
        up = up_q.astype(np.float32) * up_scale[:, None]
        down = down_q.astype(np.float32) * down_scale[:, None]
        reference = _cpu_expert(activation, gate, up, down)
        proof["cpu_reference_ns"] = time.perf_counter_ns() - cpu_started
        difference = output.astype(np.float64) - reference.astype(np.float64)
        max_error = float(np.max(np.abs(difference)))
        relative_l2 = float(
            np.linalg.norm(difference.ravel())
            / max(float(np.linalg.norm(reference.astype(np.float64).ravel())), 1e-30)
        )
        proof.update(
            {
                "maximum_absolute_error": max_error,
                "mean_absolute_error": float(np.mean(np.abs(difference))),
                "relative_l2_error": relative_l2,
                "cpu_output_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
                "cuda_output_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
                "host_to_device_bytes": int(
                    len(weight_bytes) + len(scale_bytes) + activation.nbytes
                ),
                "device_to_host_bytes": int(output.nbytes),
                "group_statistics": {
                    "calls": calls.value,
                    "experts": experts.value,
                    "rows": rows.value,
                    "h2d_ms": h2d.value,
                    "kernel_ms": kernel.value,
                    "d2h_ms": d2h.value,
                },
                "no_silent_cpu_fallback": calls.value >= 1
                and experts.value >= 1
                and resident_bytes.value > 0,
                "correctness_passed": max_error <= maximum_absolute_error
                and relative_l2 <= maximum_relative_l2_error,
                "gpu_after": _gpu_snapshot(device),
            }
        )
        if not proof["nonzero_vram_residency"]:
            raise ColibriCudaError("native expert did not establish measured VRAM residency")
        if not proof["no_silent_cpu_fallback"]:
            raise ColibriCudaError("native expert execution did not emit a GPU kernel event")
        if not proof["correctness_passed"]:
            raise ColibriCudaError(
                f"real CUDA expert failed CPU comparison: max={max_error}, rel_l2={relative_l2}"
            )
    except Exception as error:
        proof["failure"] = f"{type(error).__name__}: {error}"
        if output_path is not None:
            destination = Path(output_path).expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
        if isinstance(error, ColibriCudaError):
            raise
        raise ColibriCudaError(str(error)) from error
    finally:
        if library is not None:
            for tensor in cuda_tensors:
                if tensor.value:
                    with contextlib.suppress(Exception):
                        library.coli_cuda_tensor_free(tensor)
            if initialized:
                with contextlib.suppress(Exception):
                    library.coli_cuda_shutdown()
        for handle in handles:
            with contextlib.suppress(Exception):
                handle.close()
    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    return proof


def consolidate_real_model_cuda_results(
    *,
    expert_proof_path: str | Path,
    generation_result_path: str | Path,
    local_stdout_path: str | Path,
    distributed_stdout_path: str | Path,
    local_route_path: str | Path,
    distributed_route_path: str | Path,
    output_csv_path: str | Path,
    output_json_path: str | Path,
) -> dict[str, Any]:
    """Reconcile operator, GPU-worker, token, route, and timing evidence."""

    import re

    expert_path = Path(expert_proof_path).expanduser().resolve()
    generation_path = Path(generation_result_path).expanduser().resolve()
    local_stdout = Path(local_stdout_path).expanduser().resolve()
    distributed_stdout = Path(distributed_stdout_path).expanduser().resolve()
    local_route = Path(local_route_path).expanduser().resolve()
    distributed_route = Path(distributed_route_path).expanduser().resolve()
    expert = json.loads(expert_path.read_text(encoding="utf-8"))
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    accounting = generation.get("worker_process_accounting", [])
    cuda_workers = [row for row in accounting if row.get("cuda_requested")]
    if len(cuda_workers) != 1:
        raise ColibriCudaError("real CUDA generation must identify exactly one CUDA worker")
    worker = cuda_workers[0]
    if int(expert.get("layer_id", -1)) != int(worker.get("cuda_target_layer", -2)) or int(
        expert.get("expert_id", -1)
    ) != int(worker.get("cuda_target_expert", -2)):
        raise ColibriCudaError("operator proof expert identity differs from generation target")

    def timing(path: Path) -> dict[str, float]:
        text = path.read_text(encoding="utf-8")
        speed = re.search(r"Speed:\s*([0-9.]+) tok/s", text)
        detail = re.search(
            r"TTFT:\s*([0-9.]+)s \| Prefill:\s*([0-9.]+)s "
            r"\| Decode after first token:\s*([0-9.]+)s",
            text,
        )
        if not speed or not detail:
            raise ColibriCudaError(f"missing Colibri timing record in {path}")
        return {
            "throughput_tokens_per_second": float(speed.group(1)),
            "ttft_seconds": float(detail.group(1)),
            "prefill_seconds": float(detail.group(2)),
            "decode_after_first_token_seconds": float(detail.group(3)),
        }

    local_timing = timing(local_stdout)
    distributed_timing = timing(distributed_stdout)
    route_identity = local_route.read_bytes() == distributed_route.read_bytes()
    hidden = int(expert["hidden_dimension"])
    executions = int(worker["cuda_execution_count"])
    activation_bytes = executions * hidden * 4
    output_bytes = executions * hidden * 4
    row = {
        "evidence_category": "REAL_MODEL_MEASURED",
        "model_path": expert["model_path"],
        "model_revision": "pinned-b085b48888a88d9a1c00b151a9979774b72cdbfd",
        "model_fingerprint": generation["model_fingerprint"],
        "cuda_dll_sha256": expert["cuda_dll_sha256"],
        "layer_id": int(worker["cuda_target_layer"]),
        "expert_id": int(worker["cuda_target_expert"]),
        "quantization": expert["quantization"],
        "merged_weight_sha256": expert["merged_weight_sha256"],
        "row_scales_sha256": expert["row_scales_sha256"],
        "native_weight_bytes": expert["native_weight_bytes"],
        "native_scale_bytes": expert["native_scale_bytes"],
        "token_count": generation["expected_tokens"],
        "matching_token_count": generation["matching_tokens"],
        "exact_token_identity": bool(generation["exact_token_identity"]),
        "router_trace_identity": route_identity,
        "remote_results_consumed": generation["remote_results_consumed"],
        "forbidden_local_loads": generation["forbidden_local_loads"],
        "cuda_execution_count": executions,
        "cuda_fallback_count": int(worker["cuda_fallback_count"]),
        "gpu_resident_tensor_count": int(worker["cuda_resident_tensor_count"]),
        "gpu_resident_bytes": int(worker["cuda_resident_tensor_bytes"]),
        "weight_upload_ns": int(worker["cuda_weight_upload_ns"]),
        "gpu_h2d_ns": int(worker["cuda_h2d_ns"]),
        "gpu_kernel_ns": int(worker["cuda_kernel_ns"]),
        "gpu_d2h_ns": int(worker["cuda_d2h_ns"]),
        "gpu_execution_wall_ns": int(worker["cuda_execution_wall_ns"]),
        "generation_activation_h2d_bytes": activation_bytes,
        "generation_output_d2h_bytes": output_bytes,
        "operator_maximum_absolute_error": expert["maximum_absolute_error"],
        "operator_relative_l2_error": expert["relative_l2_error"],
        "operator_cpu_reference_ns": expert["cpu_reference_ns"],
        "operator_gpu_kernel_ns": int(expert["group_statistics"]["kernel_ms"] * 1e6),
        "local_throughput_tokens_per_second": local_timing["throughput_tokens_per_second"],
        "distributed_throughput_tokens_per_second": distributed_timing[
            "throughput_tokens_per_second"
        ],
        "local_ttft_seconds": local_timing["ttft_seconds"],
        "distributed_ttft_seconds": distributed_timing["ttft_seconds"],
        "local_prefill_seconds": local_timing["prefill_seconds"],
        "distributed_prefill_seconds": distributed_timing["prefill_seconds"],
        "local_decode_after_first_token_seconds": local_timing["decode_after_first_token_seconds"],
        "distributed_decode_after_first_token_seconds": distributed_timing[
            "decode_after_first_token_seconds"
        ],
        "pass": bool(
            generation["exact_token_identity"]
            and route_identity
            and generation["remote_results_consumed"] > 0
            and generation["forbidden_local_loads"] == 0
            and expert["correctness_passed"]
            and expert["nonzero_vram_residency"]
            and expert["no_silent_cpu_fallback"]
            and executions > 0
            and worker["cuda_fallback_count"] == 0
        ),
    }
    csv_path = Path(output_csv_path).expanduser().resolve()
    json_path = Path(output_json_path).expanduser().resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    document = {
        "schema_version": "experiment-010-real-model-cuda-results-v1",
        "complete": row["pass"],
        "result": row,
        "raw_evidence": {
            "expert_proof": str(expert_path),
            "generation_result": str(generation_path),
            "local_stdout": str(local_stdout),
            "distributed_stdout": str(distributed_stdout),
            "local_route": str(local_route),
            "distributed_route": str(distributed_route),
        },
        "limitations": [
            "This is single-machine process isolation and is not physical distributed inference.",
            "IDOT=0 is held identical across the CPU oracle and CUDA run because the CUDA ABI consumes FP32 activations; the default IDOT=1 result is retained as a negative numerical-contract diagnostic.",
        ],
    }
    json_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    arguments = parser.parse_args()
    if arguments.model:
        run_colibri_real_olmoe_cuda_expert(
            arguments.dll,
            arguments.model,
            layer_id=arguments.layer,
            expert_id=arguments.expert,
            device=arguments.device,
            output_path=arguments.output,
        )
    else:
        run_colibri_cuda_kernel_proof(
            arguments.dll, device=arguments.device, output_path=arguments.output
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
