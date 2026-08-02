"""Measured CPU, CUDA, PCIe, storage, and resource profiling for Experiment 008."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any, ClassVar


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of no observations")
    if not 0 <= percentile_value <= 100:
        raise ValueError("percentile must be between zero and one hundred")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def latency_summary(values_ms: list[float]) -> dict[str, float | int | list[float]]:
    if not values_ms:
        raise ValueError("latency summary requires observations")
    return {
        "sample_count": len(values_ms),
        "median_ms": median(values_ms),
        "p95_ms": percentile(values_ms, 95),
        "minimum_ms": min(values_ms),
        "maximum_ms": max(values_ms),
        "warmup_behaviour_ms": values_ms[: min(5, len(values_ms))],
    }


def profile_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _nvidia_query(fields: list[str]) -> list[str] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return [item.strip() for item in result.stdout.splitlines()[0].split(",")]


def collect_hardware_identity(*, backend: str, model: str, quantization: str) -> dict[str, Any]:
    import psutil
    import torch

    gpu_values = _nvidia_query(["name", "uuid", "driver_version", "memory.total", "vbios_version"])
    gpu = None
    if gpu_values and len(gpu_values) == 5:
        gpu = {
            "name": gpu_values[0],
            "uuid": gpu_values[1],
            "driver": gpu_values[2],
            "memory_total_mib": float(gpu_values[3]),
            "vbios": gpu_values[4],
        }
    identity = {
        "hostname": socket.gethostname(),
        "operating_system": platform.platform(),
        "cpu": platform.processor(),
        "physical_cpu_cores": psutil.cpu_count(logical=False),
        "logical_cpu_cores": psutil.cpu_count(logical=True),
        "system_ram_bytes": int(psutil.virtual_memory().total),
        "gpu": gpu,
        "cuda_runtime": torch.version.cuda,
        "pytorch": torch.__version__,
        "python": platform.python_version(),
        "backend": backend,
        "model": model,
        "quantization": quantization,
    }
    return {**identity, "fingerprint": profile_fingerprint(identity)}


def _cpu_times(
    operation: Callable[[], Any], warmups: int, iterations: int
) -> tuple[list[float], list[float]]:
    warmup_values: list[float] = []
    for _ in range(warmups):
        started = time.perf_counter_ns()
        operation()
        warmup_values.append((time.perf_counter_ns() - started) / 1_000_000)
    values: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        operation()
        values.append((time.perf_counter_ns() - started) / 1_000_000)
    return warmup_values, values


def _cuda_times(
    operation: Callable[[], Any], warmups: int, iterations: int
) -> tuple[list[float], list[float]]:
    import torch

    warmup_values: list[float] = []
    for _ in range(warmups):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        warmup_values.append(float(start.elapsed_time(end)))
    torch.cuda.synchronize()
    values: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    return warmup_values, values


def _kernel_row(
    *,
    operation: str,
    device: str,
    shape: list[int],
    dtype: str,
    values: list[float],
    warmup_values: list[float],
    touched_bytes: int,
    thread_count: int | None,
    affinity: list[int] | None,
) -> dict[str, Any]:
    summary = latency_summary(values)
    median_ms = float(summary["median_ms"])
    return {
        "classification": "MEASURED",
        "operation": operation,
        "device": device,
        "shape": shape,
        "dtype": dtype,
        **summary,
        "warmup_behaviour_ms": warmup_values,
        "effective_bandwidth_bytes_s": touched_bytes / max(median_ms / 1000, 1e-12),
        "thread_count": thread_count,
        "affinity": affinity,
    }


def profile_kernels(
    *,
    decode_shapes: list[list[int]],
    prefill_shapes: list[list[int]],
    cpu_thread_counts: list[int],
    warmups: int,
    iterations: int,
) -> list[dict[str, Any]]:
    import psutil
    import torch

    rows: list[dict[str, Any]] = []
    process = psutil.Process()
    try:
        affinity = list(process.cpu_affinity())
    except (AttributeError, OSError, psutil.Error):
        affinity = None
    original_threads = torch.get_num_threads()
    cpu_shapes = [decode_shapes[0], prefill_shapes[0]]
    for thread_count in sorted(set(cpu_thread_counts)):
        torch.set_num_threads(thread_count)
        for shape in cpu_shapes:
            batch, input_size, output_size = shape
            left = torch.randn((batch, input_size), dtype=torch.float32)
            weight = torch.randn((input_size, output_size), dtype=torch.float32)
            warmup_values, values = _cpu_times(
                lambda left=left, weight=weight: torch.mm(left, weight), warmups, iterations
            )
            touched = (left.numel() + weight.numel() + batch * output_size) * 4
            rows.append(
                _kernel_row(
                    operation="matrix_vector" if batch == 1 else "matrix_matrix",
                    device="cpu",
                    shape=shape,
                    dtype="float32",
                    values=values,
                    warmup_values=warmup_values,
                    touched_bytes=touched,
                    thread_count=thread_count,
                    affinity=affinity,
                )
            )
        hidden = decode_shapes[0][1]
        intermediate = decode_shapes[0][2]
        cpu_batch = max(decode_shapes[0][0], 1)
        cpu_hidden = torch.randn((cpu_batch, hidden), dtype=torch.float32)
        cpu_gate = torch.randn((hidden, intermediate), dtype=torch.float32)
        cpu_up = torch.randn((hidden, intermediate), dtype=torch.float32)
        cpu_down = torch.randn((intermediate, hidden), dtype=torch.float32)

        def cpu_expert_sequence(
            hidden_state: Any = cpu_hidden,
            gate_weight: Any = cpu_gate,
            up_weight: Any = cpu_up,
            down_weight: Any = cpu_down,
        ) -> Any:
            return (
                torch.nn.functional.silu(hidden_state @ gate_weight) * (hidden_state @ up_weight)
            ) @ down_weight

        expert_warmup, expert_values = _cpu_times(cpu_expert_sequence, warmups, iterations)
        rows.append(
            _kernel_row(
                operation="expert_projection_sequence",
                device="cpu",
                shape=[cpu_batch, hidden, intermediate],
                dtype="float32",
                values=expert_values,
                warmup_values=expert_warmup,
                touched_bytes=(
                    cpu_hidden.numel() + cpu_gate.numel() + cpu_up.numel() + cpu_down.numel()
                )
                * 4,
                thread_count=thread_count,
                affinity=affinity,
            )
        )
        cpu_reduction = torch.randn((32, hidden), dtype=torch.float32)
        reduction_warmup, reduction_values = _cpu_times(
            lambda value=cpu_reduction: value.sum(dim=0), warmups, iterations
        )
        rows.append(
            _kernel_row(
                operation="reduction_sum",
                device="cpu",
                shape=list(cpu_reduction.shape),
                dtype="float32",
                values=reduction_values,
                warmup_values=reduction_warmup,
                touched_bytes=cpu_reduction.numel() * 4,
                thread_count=thread_count,
                affinity=affinity,
            )
        )
        cpu_activation = torch.randn((256, hidden), dtype=torch.float32)
        activation_warmup, activation_values = _cpu_times(
            lambda value=cpu_activation: torch.nn.functional.silu(value), warmups, iterations
        )
        rows.append(
            _kernel_row(
                operation="silu_activation",
                device="cpu",
                shape=list(cpu_activation.shape),
                dtype="float32",
                values=activation_values,
                warmup_values=activation_warmup,
                touched_bytes=cpu_activation.numel() * 8,
                thread_count=thread_count,
                affinity=affinity,
            )
        )
        cpu_quantized = torch.randint(-127, 128, (hidden, intermediate), dtype=torch.int8)
        dequant_warmup, dequant_values = _cpu_times(
            lambda value=cpu_quantized: value.float() * 0.02, warmups, iterations
        )
        rows.append(
            _kernel_row(
                operation="int8_dequantization_proxy",
                device="cpu",
                shape=list(cpu_quantized.shape),
                dtype="int8_to_float32",
                values=dequant_values,
                warmup_values=dequant_warmup,
                touched_bytes=cpu_quantized.numel() * 5,
                thread_count=thread_count,
                affinity=affinity,
            )
        )
    torch.set_num_threads(original_threads)
    if not torch.cuda.is_available():
        return rows
    torch.cuda.reset_peak_memory_stats()
    for shape in [*decode_shapes, *prefill_shapes]:
        batch, input_size, output_size = shape
        left = torch.randn((batch, input_size), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn((input_size, output_size), device="cuda", dtype=torch.bfloat16)
        warmup_values, values = _cuda_times(
            lambda left=left, weight=weight: torch.mm(left, weight), warmups, iterations
        )
        touched = (left.numel() + weight.numel() + batch * output_size) * 2
        row = _kernel_row(
            operation="matrix_vector" if batch == 1 else "matrix_matrix",
            device="cuda",
            shape=shape,
            dtype="bfloat16",
            values=values,
            warmup_values=warmup_values,
            touched_bytes=touched,
            thread_count=None,
            affinity=None,
        )
        row["gpu_memory_used_bytes"] = int(torch.cuda.max_memory_allocated())
        rows.append(row)
        del left, weight
    hidden = decode_shapes[0][1]
    intermediate = decode_shapes[0][2]
    batch = max(decode_shapes[0][0], 1)
    hidden_state = torch.randn((batch, hidden), device="cuda", dtype=torch.bfloat16)
    gate = torch.randn((hidden, intermediate), device="cuda", dtype=torch.bfloat16)
    up = torch.randn((hidden, intermediate), device="cuda", dtype=torch.bfloat16)
    down = torch.randn((intermediate, hidden), device="cuda", dtype=torch.bfloat16)

    def expert_sequence() -> Any:
        return (torch.nn.functional.silu(hidden_state @ gate) * (hidden_state @ up)) @ down

    expert_warmup, expert_values = _cuda_times(expert_sequence, warmups, iterations)
    rows.append(
        _kernel_row(
            operation="expert_projection_sequence",
            device="cuda",
            shape=[batch, hidden, intermediate],
            dtype="bfloat16",
            values=expert_values,
            warmup_values=expert_warmup,
            touched_bytes=(hidden_state.numel() + gate.numel() + up.numel() + down.numel()) * 2,
            thread_count=None,
            affinity=None,
        )
    )
    reduction = torch.randn((32, hidden), device="cuda", dtype=torch.float32)
    reduction_warmup, reduction_values = _cuda_times(
        lambda: reduction.sum(dim=0), warmups, iterations
    )
    rows.append(
        _kernel_row(
            operation="reduction_sum",
            device="cuda",
            shape=list(reduction.shape),
            dtype="float32",
            values=reduction_values,
            warmup_values=reduction_warmup,
            touched_bytes=reduction.numel() * 4,
            thread_count=None,
            affinity=None,
        )
    )
    activation = torch.randn((1024, hidden), device="cuda", dtype=torch.bfloat16)
    activation_warmup, activation_values = _cuda_times(
        lambda: torch.nn.functional.silu(activation), warmups, iterations
    )
    rows.append(
        _kernel_row(
            operation="silu_activation",
            device="cuda",
            shape=list(activation.shape),
            dtype="bfloat16",
            values=activation_values,
            warmup_values=activation_warmup,
            touched_bytes=activation.numel() * 4,
            thread_count=None,
            affinity=None,
        )
    )
    quantized = torch.randint(-127, 128, (hidden, intermediate), device="cuda", dtype=torch.int8)
    scale = torch.tensor(0.02, device="cuda")
    dequant_warmup, dequant_values = _cuda_times(
        lambda: quantized.float() * scale, warmups, iterations
    )
    row = _kernel_row(
        operation="int8_dequantization_proxy",
        device="cuda",
        shape=list(quantized.shape),
        dtype="int8_to_float32",
        values=dequant_values,
        warmup_values=dequant_warmup,
        touched_bytes=quantized.numel() * 5,
        thread_count=None,
        affinity=None,
    )
    row["limitation"] = (
        "PyTorch proxy; target GGUF Q4_K kernels are measured end-to-end by llama.cpp"
    )
    rows.append(row)
    torch.cuda.synchronize()
    return rows


def _transfer_once(source: Any, destination: Any, *, non_blocking: bool) -> None:
    destination.copy_(source, non_blocking=non_blocking)


def profile_pcie(
    payload_bytes: list[int], *, warmups: int, iterations: int
) -> list[dict[str, Any]]:
    import torch

    if not torch.cuda.is_available():
        return [
            {
                "classification": "MEASURED",
                "status": "UNSUPPORTED",
                "reason": "CUDA is unavailable",
            }
        ]
    rows: list[dict[str, Any]] = []
    for size in payload_bytes:
        for pinned in (False, True):
            try:
                host = torch.empty(size, dtype=torch.uint8, pin_memory=pinned)
            except RuntimeError as exc:
                rows.append(
                    {
                        "classification": "MEASURED",
                        "status": "UNSUPPORTED",
                        "direction": "both",
                        "memory_kind": "pinned" if pinned else "pageable",
                        "payload_bytes": size,
                        "reason": str(exc),
                    }
                )
                continue
            device = torch.empty(size, dtype=torch.uint8, device="cuda")
            memory_kind = "pinned" if pinned else "pageable"
            for direction, source, destination in (
                ("host_to_device", host, device),
                ("device_to_host", device, host),
            ):

                def operation(
                    source: Any = source,
                    destination: Any = destination,
                    pinned: bool = pinned,
                ) -> None:
                    _transfer_once(source, destination, non_blocking=pinned)

                _, first_values = _cuda_times(operation, 0, 1)
                first = first_values[0]
                transfer_warmup, repeated = _cuda_times(
                    operation,
                    warmups,
                    iterations,
                )
                summary = latency_summary(repeated)
                med = float(summary["median_ms"])
                rows.append(
                    {
                        "classification": "MEASURED",
                        "status": "COMPLETED",
                        "direction": direction,
                        "memory_kind": memory_kind,
                        "payload_bytes": size,
                        "cold_transfer_ms": first,
                        **summary,
                        "warmup_behaviour_ms": transfer_warmup,
                        "effective_bandwidth_bytes_s": size / max(med / 1000, 1e-12),
                    }
                )
            del host, device
    return rows


def profile_async_overlap(trace_path: Path, *, quick: bool) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {
            "classification": "MEASURED",
            "status": "UNSUPPORTED",
            "reason": "CUDA is unavailable",
            "overlap_percent": None,
        }
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    size = 512 if quick else 2048
    cpu_left = torch.randn((size, size), dtype=torch.float32)
    cpu_right = torch.randn((size, size), dtype=torch.float32)
    gpu_left = torch.randn((size, size), dtype=torch.float32, device="cuda")
    gpu_right = torch.randn((size, size), dtype=torch.float32, device="cuda")
    payload = torch.empty((16 if quick else 128) << 20, dtype=torch.uint8, pin_memory=True)
    target = torch.empty_like(payload, device="cuda")
    intervals: dict[str, tuple[int, int]] = {}

    def cpu_work() -> None:
        started = time.perf_counter_ns()
        with torch.profiler.record_function("experiment_008_cpu_expert_work"):
            torch.mm(cpu_left, cpu_right)
        intervals["cpu"] = (started, time.perf_counter_ns())

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with (
        torch.profiler.profile(activities=activities, record_shapes=True) as profiler,
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="exp008-cpu") as pool,
    ):
        future = pool.submit(cpu_work)
        gpu_started = time.perf_counter_ns()
        compute_stream = torch.cuda.Stream()
        transfer_stream = torch.cuda.Stream()
        with (
            torch.cuda.stream(compute_stream),
            torch.profiler.record_function("experiment_008_gpu_shared_expert_work"),
        ):
            torch.mm(gpu_left, gpu_right)
        with (
            torch.cuda.stream(transfer_stream),
            torch.profiler.record_function("experiment_008_expert_prefetch"),
        ):
            target.copy_(payload, non_blocking=True)
        compute_stream.synchronize()
        transfer_stream.synchronize()
        gpu_ended = time.perf_counter_ns()
        future.result()
        intervals["gpu_and_transfer"] = (gpu_started, gpu_ended)
    profiler.export_chrome_trace(str(trace_path))
    cpu_start, cpu_end = intervals["cpu"]
    gpu_start, gpu_end = intervals["gpu_and_transfer"]
    overlap_ns = max(0, min(cpu_end, gpu_end) - max(cpu_start, gpu_start))
    shorter_ns = min(cpu_end - cpu_start, gpu_end - gpu_start)
    return {
        "classification": "MEASURED",
        "status": "COMPLETED",
        "cpu_interval_monotonic_ns": [cpu_start, cpu_end],
        "gpu_and_transfer_interval_monotonic_ns": [gpu_start, gpu_end],
        "overlap_ns": overlap_ns,
        "overlap_percent": overlap_ns / shorter_ns * 100 if shorter_ns > 0 else 0.0,
        "trace_path": str(trace_path),
        "proof_scope": "real concurrent PyTorch CPU matmul, CUDA matmul, and pinned H2D transfer",
        "target_model_scope": False,
    }


def profile_storage(path: Path | None, *, sample_bytes: int, quick: bool) -> dict[str, Any]:
    import psutil

    if path is None or not path.is_file():
        return {
            "classification": "MEASURED",
            "status": "UNSUPPORTED",
            "reason": "no model file was available for storage profiling",
        }
    size = min(path.stat().st_size, sample_bytes if not quick else min(sample_bytes, 64 << 20))
    chunk_size = 8 << 20
    process = psutil.Process()

    def faults() -> int | None:
        value = getattr(process.memory_info(), "num_page_faults", None)
        return int(value) if value is not None else None

    def sequential() -> tuple[float, int | None]:
        before = faults()
        started = time.perf_counter()
        remaining = size
        with path.open("rb", buffering=0) as handle:
            while remaining:
                block = handle.read(min(chunk_size, remaining))
                if not block:
                    break
                remaining -= len(block)
        elapsed = time.perf_counter() - started
        after = faults()
        return elapsed, after - before if before is not None and after is not None else None

    first_seconds, first_faults = sequential()
    warm_seconds, warm_faults = sequential()
    generator = random.Random(8008)
    positions = [generator.randrange(0, max(path.stat().st_size - 4096, 1)) for _ in range(256)]
    started = time.perf_counter()
    with path.open("rb", buffering=0) as handle:
        for position in positions:
            handle.seek(position)
            handle.read(4096)
    random_seconds = time.perf_counter() - started
    return {
        "classification": "MEASURED",
        "status": "COMPLETED",
        "path": str(path),
        "sample_bytes": size,
        "first_read_seconds": first_seconds,
        "first_read_bandwidth_bytes_s": size / max(first_seconds, 1e-12),
        "warm_cache_seconds": warm_seconds,
        "warm_cache_bandwidth_bytes_s": size / max(warm_seconds, 1e-12),
        "random_read_count": len(positions),
        "random_read_iops": len(positions) / max(random_seconds, 1e-12),
        "first_read_page_faults": first_faults,
        "warm_read_page_faults": warm_faults,
        "cold_definition": "first read in this process; operating-system cache was not flushed",
    }


def build_hardware_profile(
    *,
    backend: str,
    model: str,
    quantization: str,
    model_path: Path | None,
    decode_shapes: list[list[int]],
    prefill_shapes: list[list[int]],
    cpu_thread_counts: list[int],
    payload_bytes: list[int],
    warmups: int,
    iterations: int,
    storage_sample_bytes: int,
    trace_path: Path,
    quick: bool,
) -> dict[str, Any]:
    identity = collect_hardware_identity(backend=backend, model=model, quantization=quantization)
    profile = {
        "schema_version": "experiment-008-hardware-profile-v1",
        "classification": "MEASURED",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "fingerprint": identity["fingerprint"],
        "identity": identity,
        "runtime_settings": {
            "warmups": warmups,
            "iterations": iterations,
            "quick": quick,
            "decode_shapes": decode_shapes,
            "prefill_shapes": prefill_shapes,
            "cpu_thread_counts": cpu_thread_counts,
            "payload_bytes": payload_bytes,
        },
        "kernel_measurements": profile_kernels(
            decode_shapes=decode_shapes,
            prefill_shapes=prefill_shapes,
            cpu_thread_counts=cpu_thread_counts,
            warmups=warmups,
            iterations=iterations,
        ),
        "pcie_measurements": profile_pcie(payload_bytes, warmups=warmups, iterations=iterations),
        "storage_measurements": profile_storage(
            model_path, sample_bytes=storage_sample_bytes, quick=quick
        ),
        "overlap_measurement": profile_async_overlap(trace_path, quick=quick),
    }
    profile["profile_key"] = profile_fingerprint(
        {
            "identity": identity,
            "runtime_settings": profile["runtime_settings"],
        }
    )
    return profile


class ResourceSampler:
    """Collect host, process-tree, GPU, PCIe-link, and disk counters until stopped."""

    GPU_FIELDS: ClassVar[list[str]] = [
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "temperature.gpu",
        "power.draw",
        "clocks_throttle_reasons.active",
        "pcie.link.gen.current",
        "pcie.link.width.current",
    ]

    def __init__(self, *, interval_seconds: float, label: str) -> None:
        if interval_seconds <= 0:
            raise ValueError("resource sample interval must be positive")
        self.interval_seconds = interval_seconds
        self.label = label
        self.rows: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._dmon_thread: threading.Thread | None = None
        self._dmon_process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("resource sampler is already running")
        self._thread = threading.Thread(target=self._run, name="exp008-resource", daemon=True)
        self._thread.start()
        self._start_pcie_dmon()

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        if self._dmon_process is not None and self._dmon_process.poll() is None:
            self._dmon_process.terminate()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval_seconds * 4, 2.0))
        if self._dmon_thread is not None:
            self._dmon_thread.join(timeout=3.0)
        if self._dmon_process is not None and self._dmon_process.poll() is None:
            self._dmon_process.kill()
        return self.rows

    def _start_pcie_dmon(self) -> None:
        """Collect the driver's measured PCIe Rx/Tx rate without blocking inference."""

        startup_info = None
        creation_flags = 0
        if os.name == "nt":
            startup_info = subprocess.STARTUPINFO()
            startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            creation_flags = subprocess.CREATE_NO_WINDOW
        try:
            self._dmon_process = subprocess.Popen(
                ["nvidia-smi", "dmon", "-s", "t", "-d", "1"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                errors="replace",
                startupinfo=startup_info,
                creationflags=creation_flags,
            )
        except OSError:
            self._dmon_process = None
            return

        def read_dmon() -> None:
            assert self._dmon_process is not None
            assert self._dmon_process.stdout is not None
            for line in self._dmon_process.stdout:
                if self._stop.is_set():
                    break
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                fields = stripped.split()
                if len(fields) < 3:
                    continue
                try:
                    rx_mib_s = float(fields[1])
                    tx_mib_s = float(fields[2])
                except ValueError:
                    continue
                self.rows.append(
                    {
                        "classification": "MEASURED",
                        "label": self.label,
                        "sampler_source": "nvidia-smi-dmon",
                        "timestamp_utc": datetime.now(UTC).isoformat(),
                        "monotonic_ns": time.perf_counter_ns(),
                        "pcie_gpu_receive_mib_s": rx_mib_s,
                        "pcie_gpu_transmit_mib_s": tx_mib_s,
                    }
                )

        self._dmon_thread = threading.Thread(target=read_dmon, name="exp008-pcie-dmon", daemon=True)
        self._dmon_thread.start()

    def _run(self) -> None:
        import psutil

        process = psutil.Process()
        process.cpu_percent(None)
        psutil.cpu_percent(None, percpu=True)
        while not self._stop.is_set():
            disk = psutil.disk_io_counters()
            memory = psutil.virtual_memory()
            try:
                children = process.children(recursive=True)
            except psutil.Error:
                children = []
            processes = [process, *children]
            cpu = 0.0
            rss = 0
            for item in processes:
                try:
                    cpu += item.cpu_percent(None)
                    rss += item.memory_info().rss
                except psutil.Error:
                    continue
            gpu = _nvidia_query(self.GPU_FIELDS)
            row: dict[str, Any] = {
                "classification": "MEASURED",
                "label": self.label,
                "sampler_source": "host-and-gpu-query",
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "monotonic_ns": time.perf_counter_ns(),
                "process_tree_cpu_percent": cpu,
                "process_tree_rss_bytes": rss,
                "system_ram_used_bytes": int(memory.used),
                "system_ram_available_bytes": int(memory.available),
                "cpu_percent_by_logical_core": psutil.cpu_percent(None, percpu=True),
                "disk_read_bytes": int(disk.read_bytes) if disk else None,
                "disk_write_bytes": int(disk.write_bytes) if disk else None,
            }
            if gpu and len(gpu) == len(self.GPU_FIELDS):
                names = [
                    "gpu_compute_utilisation_percent",
                    "gpu_memory_controller_utilisation_percent",
                    "gpu_memory_used_mib",
                    "gpu_temperature_c",
                    "gpu_power_watts",
                    "gpu_throttle_reasons",
                    "pcie_link_generation",
                    "pcie_link_width",
                ]
                row.update(dict(zip(names, gpu, strict=True)))
            self.rows.append(row)
            self._stop.wait(self.interval_seconds)


def write_resource_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) or ["status", "reason"]
    materialized = rows or [{"status": "INCOMPLETE", "reason": "no resource samples"}]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in materialized:
            writer.writerow(
                {
                    key: json.dumps(value, separators=(",", ":"))
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )
