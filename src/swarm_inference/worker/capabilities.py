"""Measured worker capability reports."""

from __future__ import annotations

import platform
import socket
import time
from uuid import uuid4

import numpy as np
import psutil

from swarm_inference.config.models import (
    Backend,
    OperationKind,
    StageBenchmark,
    WorkerCapability,
)
from swarm_inference.exceptions import BackendIncompatibleError
from swarm_inference.host import detect_host_runtime, split_endpoint
from swarm_inference.security.identity import WorkerIdentity


def _gpu_details(backend: Backend) -> tuple[str | None, int, int, list[str]]:
    if backend == Backend.SYNTHETIC:
        return None, 0, 0, ["float32", "float16", "bfloat16"]
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise BackendIncompatibleError(
            f"{backend.value} requested but PyTorch could not be imported: {exc}"
        ) from exc
    if backend == Backend.TORCH_CPU:
        try:
            probe = torch.ones(1, device="cpu")
            if probe.item() != 1:
                raise RuntimeError("unexpected CPU probe value")
        except (RuntimeError, OSError) as exc:
            raise BackendIncompatibleError(
                f"torch-cpu requested but a CPU tensor operation failed: {exc}"
            ) from exc
        return None, 0, 0, ["float32", "float16", "bfloat16"]
    if backend == Backend.TORCH_CUDA:
        if not torch.cuda.is_available():
            raise BackendIncompatibleError(
                "torch-cuda requested but CUDA is not visible to PyTorch"
            )
        try:
            probe = torch.ones(1, device="cuda")
            torch.cuda.synchronize()
            if probe.item() != 1:
                raise RuntimeError("unexpected CUDA probe value")
        except (RuntimeError, OSError) as exc:
            raise BackendIncompatibleError(
                f"torch-cuda requested but a CUDA tensor operation failed: {exc}"
            ) from exc
        free, total = torch.cuda.mem_get_info(0)
        return (
            torch.cuda.get_device_name(0),
            int(total),
            int(free),
            ["float32", "float16", "bfloat16"],
        )
    if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
        raise BackendIncompatibleError("torch-mps requested but MPS is not visible to PyTorch")
    try:
        probe = torch.ones(1, device="mps")
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
        [
            "float32",
            "float16",
        ],
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
    memory_limit_bytes: int | None = None,
    upload_bandwidth_bytes_s: float = 125_000_000,
    download_bandwidth_bytes_s: float = 125_000_000,
    coordinator_latency_ms: float | None = None,
    network_rates_measured: bool = False,
) -> WorkerCapability:
    memory = psutil.virtual_memory()
    host = detect_host_runtime()
    gpu_model, total_vram, available_vram, dtypes = _gpu_details(backend)
    hostname = socket.gethostname()
    logical = psutil.cpu_count(logical=True) or 1
    physical = psutil.cpu_count(logical=False)
    worker_class = gpu_model or platform.processor() or platform.machine()
    benchmark = measure_synthetic_benchmark(worker_class=worker_class)
    latency = coordinator_latency_ms if coordinator_latency_ms is not None else 0.0
    return WorkerCapability(
        worker_id=worker_id or f"{hostname}-{uuid4().hex[:8]}",
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
        supported_dtypes=dtypes,
        supported_quantisation_formats=["none"],
        measured_memory_bandwidth_bytes_s=measure_memory_bandwidth(),
        stage_benchmarks=[benchmark],
        upload_bandwidth_bytes_s=upload_bandwidth_bytes_s,
        download_bandwidth_bytes_s=download_bandwidth_bytes_s,
        coordinator_latency_ms=latency,
        reliability_score=1.0,
        memory_limit_bytes=memory_limit_bytes,
        endpoint=endpoint,
        profile_source="measured" if network_rates_measured else "mixed",
    )
