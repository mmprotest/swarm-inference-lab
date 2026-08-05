"""Measured worker capability reports."""

from __future__ import annotations

import hashlib
import os
import platform
import socket
import time
from typing import Any
from uuid import uuid4

import numpy as np
import psutil

from swarm_inference import __version__
from swarm_inference.config.models import (
    Backend,
    OperationKind,
    StageBenchmark,
    WorkerCapability,
)
from swarm_inference.exceptions import BackendIncompatibleError
from swarm_inference.host import detect_host_runtime, split_endpoint
from swarm_inference.protocol.stage_ring import STAGE_RING_PROTOCOL_VERSION
from swarm_inference.security.identity import WorkerIdentity


def _measured_torch_dtypes(torch: Any, device: Any) -> list[str]:
    """Report only dtypes that complete a real device matrix operation."""

    supported: list[str] = []
    for name, dtype in (
        ("float32", torch.float32),
        ("float16", torch.float16),
        ("bfloat16", torch.bfloat16),
    ):
        try:
            values = torch.ones((2, 2), device=device, dtype=dtype)
            result = values @ values
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
            if result.float().sum().item() != 8.0:
                continue
        except (RuntimeError, OSError, TypeError):
            continue
        supported.append(name)
    return supported


def _gpu_details(
    backend: Backend,
    *,
    device_identifier: str | None = None,
) -> tuple[str | None, int, int, list[str]]:
    if backend == Backend.SYNTHETIC:
        return None, 0, 0, ["float32", "float16", "bfloat16"]
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise BackendIncompatibleError(
            f"{backend.value} requested but PyTorch could not be imported: {exc}"
        ) from exc
    if backend == Backend.TORCH_CPU:
        device = torch.device("cpu")
        try:
            probe = torch.ones(1, device=device)
            if probe.item() != 1:
                raise RuntimeError("unexpected CPU probe value")
        except (RuntimeError, OSError) as exc:
            raise BackendIncompatibleError(
                f"torch-cpu requested but a CPU tensor operation failed: {exc}"
            ) from exc
        return None, 0, 0, _measured_torch_dtypes(torch, device)
    if backend == Backend.TORCH_CUDA:
        if not torch.cuda.is_available():
            raise BackendIncompatibleError(
                "torch-cuda requested but CUDA is not visible to PyTorch"
            )
        try:
            device = torch.device(device_identifier or "cuda")
            probe = torch.ones(1, device=device)
            torch.cuda.synchronize(device)
            if probe.item() != 1:
                raise RuntimeError("unexpected CUDA probe value")
        except (RuntimeError, OSError) as exc:
            raise BackendIncompatibleError(
                f"torch-cuda requested but a CUDA tensor operation failed: {exc}"
            ) from exc
        index = device.index if device.index is not None else torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(index)
        return (
            torch.cuda.get_device_name(index),
            int(total),
            int(free),
            _measured_torch_dtypes(torch, device),
        )
    if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
        raise BackendIncompatibleError("torch-mps requested but MPS is not visible to PyTorch")
    try:
        device = torch.device("mps")
        probe = torch.ones(1, device=device)
        if probe.cpu().item() != 1:
            raise RuntimeError("unexpected MPS probe value")
    except (RuntimeError, OSError) as exc:
        raise BackendIncompatibleError(
            f"torch-mps requested but an MPS tensor operation failed: {exc}"
        ) from exc
    memory = psutil.virtual_memory()
    return (
        "Apple Metal Performance Shaders",
        memory.total,
        memory.available,
        _measured_torch_dtypes(torch, device),
    )


def measure_memory_bandwidth(*, bytes_to_copy: int = 8 * 1024 * 1024) -> float:
    source = np.arange(bytes_to_copy, dtype=np.uint8)
    samples: list[float] = []
    for _ in range(3):
        start = time.perf_counter()
        destination = source.copy()
        elapsed = time.perf_counter() - start
        if int(destination[-1]) != int(source[-1]):
            raise RuntimeError("memory bandwidth benchmark copy failed")
        samples.append(bytes_to_copy / max(elapsed, 1e-12))
    return float(np.median(samples))


def measure_synthetic_benchmark(
    *,
    worker_class: str,
    samples: int = 5,
) -> StageBenchmark:
    values = np.arange(64 * 1024, dtype=np.float32)
    timings = []
    for _ in range(samples):
        start = time.perf_counter()
        for layer in range(8):
            values = values * np.float32(1.0001 + layer / 100_000) + np.float32(0.0001)
        timings.append((time.perf_counter() - start) * 1000)
    return StageBenchmark(
        worker_class=worker_class,
        operation=OperationKind.DECODE,
        sequence_length=1,
        batch_size=1,
        mean_ms=float(np.mean(timings)),
        p95_ms=float(np.percentile(timings, 95)),
        samples=samples,
        measured=True,
        median_ms=float(np.median(timings)),
        sample_ms=[float(value) for value in timings],
        device="synthetic-cpu",
        dtype="float32",
        dimensions={"elements": 64 * 1024, "layers": 8},
        benchmark_version="legacy-cpu-numpy-synthetic-v1",
        measured_at_unix_ns=time.time_ns(),
        correctness_passed=True,
        warmup_iterations=0,
        measurement_source="legacy-cpu-numpy-synthetic",
    )


def _normalise_dtype(value: str) -> str:
    return {
        "bf16": "bfloat16",
        "f16": "float16",
        "fp16": "float16",
        "f32": "float32",
        "fp32": "float32",
    }.get(value.lower(), value.lower())


def _synchronise_device(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def measure_selected_device_benchmark(
    *,
    worker_class: str,
    device_identifier: str,
    dtype_name: str,
    samples: int = 7,
    warmup_iterations: int = 2,
    small_hidden_size: int = 256,
    representative_hidden_size: int = 2048,
    memory_copy_bytes: int = 16 * 1024 * 1024,
) -> StageBenchmark:
    """Run bounded decode-shaped compute and copy probes on the selected device."""

    if not 3 <= samples <= 32:
        raise ValueError("selected-device benchmark samples must be in [3, 32]")
    if not 0 <= warmup_iterations <= 8:
        raise ValueError("selected-device benchmark warmups must be in [0, 8]")
    if not 16 <= small_hidden_size <= representative_hidden_size <= 8192:
        raise ValueError("selected-device benchmark hidden sizes are invalid")
    if not 1024 <= memory_copy_bytes <= 256 * 1024 * 1024:
        raise ValueError("selected-device memory copy size is outside the bounded range")
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise BackendIncompatibleError(
            f"selected-device benchmark could not import PyTorch: {exc}"
        ) from exc
    normalised_dtype = _normalise_dtype(dtype_name)
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(normalised_dtype)
    if dtype is None:
        raise BackendIncompatibleError(f"unsupported benchmark dtype {dtype_name!r}")
    try:
        device = torch.device(device_identifier)
        small_input = torch.ones((1, small_hidden_size), device=device, dtype=dtype)
        small_weight = torch.full(
            (small_hidden_size, small_hidden_size),
            1.0 / small_hidden_size,
            device=device,
            dtype=dtype,
        )
        representative_input = torch.ones(
            (1, representative_hidden_size), device=device, dtype=dtype
        )
        representative_weight = torch.full(
            (representative_hidden_size, representative_hidden_size),
            1.0 / representative_hidden_size,
            device=device,
            dtype=dtype,
        )

        def operation() -> tuple[Any, Any]:
            return (
                small_input @ small_weight,
                representative_input @ representative_weight,
            )

        small_result, representative_result = operation()
        _synchronise_device(torch, device)
        if (
            small_result.shape != (1, small_hidden_size)
            or representative_result.shape != (1, representative_hidden_size)
            or not bool(torch.isfinite(small_result).all().item())
            or not bool(torch.isfinite(representative_result).all().item())
            or abs(float(small_result.float().mean().item()) - 1.0) > 0.05
            or abs(float(representative_result.float().mean().item()) - 1.0) > 0.05
        ):
            raise RuntimeError("decode-shaped matrix correctness probe failed")
        for _ in range(warmup_iterations):
            operation()
        _synchronise_device(torch, device)
        timings: list[float] = []
        for _ in range(samples):
            started = time.perf_counter()
            small_result, representative_result = operation()
            _synchronise_device(torch, device)
            elapsed_ms = (time.perf_counter() - started) * 1000
            if not bool(torch.isfinite(representative_result).all().item()):
                raise RuntimeError("decode-shaped matrix benchmark produced non-finite output")
            timings.append(elapsed_ms)
        # Use int32 for a device-local copy with exact deterministic values.  Size
        # the allocation from that tensor's element width so the configured byte
        # bound remains exact for every benchmark dtype.
        copy_element_size = int(torch.empty((), dtype=torch.int32).element_size())
        element_count = max(1, memory_copy_bytes // copy_element_size)
        copy_source = torch.arange(element_count, device=device, dtype=torch.int32)
        copy_destination = torch.empty_like(copy_source)
        actual_copy_bytes = int(copy_source.numel() * copy_source.element_size())
        copy_rates: list[float] = []
        for _ in range(samples):
            started = time.perf_counter()
            copy_destination.copy_(copy_source)
            _synchronise_device(torch, device)
            elapsed = max(time.perf_counter() - started, 1e-12)
            if int(copy_destination[-1].item()) != int(copy_source[-1].item()):
                raise RuntimeError("selected-device memory copy correctness probe failed")
            copy_rates.append(actual_copy_bytes / elapsed)
    except (RuntimeError, OSError, TypeError) as exc:
        raise BackendIncompatibleError(
            f"selected-device {device_identifier}/{normalised_dtype} benchmark failed: {exc}"
        ) from exc
    split = max(1, len(timings) // 2)
    early = float(np.median(timings[:split]))
    late = float(np.median(timings[-split:]))
    sustained_ratio = late / max(early, 1e-12)
    return StageBenchmark(
        worker_class=worker_class,
        operation=OperationKind.DECODE,
        sequence_length=1,
        batch_size=1,
        mean_ms=float(np.mean(timings)),
        median_ms=float(np.median(timings)),
        p95_ms=float(np.percentile(timings, 95)),
        sample_ms=[float(value) for value in timings],
        samples=samples,
        measured=True,
        device=device_identifier,
        dtype=normalised_dtype,
        dimensions={
            "small_hidden_size": small_hidden_size,
            "representative_hidden_size": representative_hidden_size,
            "memory_copy_bytes": actual_copy_bytes,
        },
        benchmark_version="selected-device-decode-v1",
        measured_at_unix_ns=time.time_ns(),
        correctness_passed=True,
        memory_copy_bandwidth_bytes_s=float(np.median(copy_rates)),
        warmup_iterations=warmup_iterations,
        sustained_ratio=sustained_ratio,
        throttling_detected=sustained_ratio > 1.25,
        measurement_source="selected-device-torch",
    )


def measure_coordinator_latency_ms(
    endpoint: str,
    *,
    samples: int = 3,
    timeout_s: float = 3.0,
) -> float:
    """Measure TCP connection setup to the coordinator on any supported OS."""

    host, port = split_endpoint(endpoint)
    if port == 0:
        raise ValueError("coordinator endpoint must use a non-zero port")
    timings: list[float] = []
    for _ in range(samples):
        started = time.perf_counter()
        with socket.create_connection((host, port), timeout=timeout_s):
            timings.append((time.perf_counter() - started) * 1000)
    return float(np.median(timings))


def measure_capabilities(
    *,
    backend: Backend,
    identity: WorkerIdentity,
    worker_id: str | None = None,
    endpoint: str | None = None,
    control_endpoint: str | None = None,
    data_plane_endpoint: str | None = None,
    device_identifier: str | None = None,
    stage_runtime_enabled: bool = False,
    memory_limit_bytes: int | None = None,
    upload_bandwidth_bytes_s: float = 0.0,
    download_bandwidth_bytes_s: float = 0.0,
    coordinator_latency_ms: float | None = None,
    network_rates_measured: bool = False,
    benchmark_dtype: str | None = None,
    node_id: str | None = None,
    service_mode: str = "foreground",
    platform_support_status: str = "unknown",
) -> WorkerCapability:
    memory = psutil.virtual_memory()
    host = detect_host_runtime()
    gpu_model, total_vram, available_vram, dtypes = _gpu_details(
        backend,
        device_identifier=device_identifier,
    )
    hostname = socket.gethostname()
    logical = psutil.cpu_count(logical=True) or 1
    physical = psutil.cpu_count(logical=False)
    worker_class = gpu_model or platform.processor() or platform.machine()
    if backend == Backend.SYNTHETIC:
        benchmark = measure_synthetic_benchmark(worker_class=worker_class)
        measured_dtypes = dtypes
        memory_bandwidth = measure_memory_bandwidth()
    else:
        selected_dtype = _normalise_dtype(benchmark_dtype or "float32")
        if selected_dtype not in dtypes:
            raise BackendIncompatibleError(
                f"dtype {selected_dtype!r} failed the correctness probe on {device_identifier!r}"
            )
        benchmark = measure_selected_device_benchmark(
            worker_class=worker_class,
            device_identifier=device_identifier
            or {
                Backend.TORCH_CPU: "cpu",
                Backend.TORCH_CUDA: "cuda",
                Backend.TORCH_MPS: "mps",
            }[backend],
            dtype_name=selected_dtype,
        )
        if benchmark.memory_copy_bandwidth_bytes_s is None:
            raise BackendIncompatibleError(
                "selected-device benchmark did not report memory copy bandwidth"
            )
        measured_dtypes = [selected_dtype]
        memory_bandwidth = benchmark.memory_copy_bandwidth_bytes_s
    latency = coordinator_latency_ms if coordinator_latency_ms is not None else 0.0
    selected_worker_id = worker_id or f"{hostname}-{uuid4().hex[:8]}"
    selected_node_id = node_id or (
        selected_worker_id.split("/", 1)[0] if "/" in selected_worker_id else None
    )
    build_id = os.environ.get("SWARM_BUILD_ID", f"swarm-inference-lab-{__version__}")
    package_lock_hash = (
        os.environ.get("SWARM_PACKAGE_LOCK_HASH")
        or hashlib.sha256(f"{__version__}:{build_id}".encode()).hexdigest()
    )
    return WorkerCapability(
        worker_id=selected_worker_id,
        node_id=selected_node_id,
        public_key=identity.public_key_b64,
        hostname=hostname,
        operating_system=f"{host.system} {host.release}",
        architecture=host.machine,
        backend=backend,
        cpu_model=platform.processor() or "unknown",
        logical_cpu_count=logical,
        physical_cpu_count=physical,
        total_ram_bytes=memory.total,
        available_ram_bytes=memory.available,
        gpu_model=gpu_model,
        total_vram_bytes=total_vram,
        available_vram_bytes=available_vram,
        supported_dtypes=measured_dtypes,
        supported_quantisation_formats=["none"],
        measured_memory_bandwidth_bytes_s=memory_bandwidth,
        stage_benchmarks=[benchmark],
        upload_bandwidth_bytes_s=upload_bandwidth_bytes_s,
        download_bandwidth_bytes_s=download_bandwidth_bytes_s,
        coordinator_latency_ms=latency,
        reliability_score=1.0,
        memory_limit_bytes=memory_limit_bytes,
        endpoint=endpoint,
        profile_source="measured" if network_rates_measured else "mixed",
        control_endpoint=control_endpoint or endpoint,
        data_plane_endpoint=data_plane_endpoint,
        device_identifier=device_identifier,
        stage_ring_protocol_version=(
            STAGE_RING_PROTOCOL_VERSION if stage_runtime_enabled else None
        ),
        supported_model_adapters=["olmoe"] if stage_runtime_enabled else [],
        supported_stage_execution_backends=(
            ["canonical-contiguous-olmoe"] if stage_runtime_enabled else []
        ),
        supported_activation_dtypes=(list(measured_dtypes) if stage_runtime_enabled else []),
        configured_memory_limit_bytes=memory_limit_bytes,
        stage_runtime_enabled=stage_runtime_enabled,
        agent_version=__version__,
        runtime_version=__version__,
        build_id=build_id,
        package_lock_hash=package_lock_hash,
        product_protocol_major=1,
        product_protocol_minor=0,
        artifact_format_versions=[1],
        service_mode=service_mode,
        platform_support_status=platform_support_status,
    )
