"""Single-process benchmark worker for Experiment 004.

The orchestrator starts a fresh process per engine/model point so CUDA contexts
and model allocations cannot leak across comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return ordered[index]


def _statistics(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "median": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
            "standard_deviation": 0.0,
            "coefficient_of_variation": 0.0,
        }
    mean = statistics.mean(values)
    deviation = statistics.pstdev(values)
    return {
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "standard_deviation": deviation,
        "coefficient_of_variation": deviation / mean if mean else 0.0,
    }


def _token_hash(rows: list[list[int]]) -> str:
    encoded = json.dumps(rows, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(slots=True)
class Telemetry:
    interval_seconds: float
    samples: list[dict[str, float]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self) -> None:
        process = psutil.Process()

        def sample() -> None:
            process.cpu_percent(interval=None)
            while not self._stop.wait(self.interval_seconds):
                row: dict[str, float] = {
                    "time_monotonic": time.monotonic(),
                    "host_cpu_percent": process.cpu_percent(interval=None),
                    "host_rss_bytes": float(process.memory_info().rss),
                }
                try:
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu,utilization.memory,power.draw,memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=2,
                    )
                    if result.returncode == 0:
                        values = [
                            float(value.strip())
                            for value in result.stdout.splitlines()[0].split(",")
                        ]
                        (
                            row["gpu_utilisation_percent"],
                            row["memory_controller_utilisation_percent"],
                            row["power_watts"],
                            memory_mib,
                        ) = values
                        row["gpu_memory_bytes"] = memory_mib * 1024 * 1024
                except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
                    pass
                self.samples.append(row)

        self._thread = threading.Thread(
            target=sample,
            name="experiment-004-telemetry",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def summary(self) -> dict[str, float | int]:
        result: dict[str, float | int] = {"sample_count": len(self.samples)}
        for field_name in (
            "host_cpu_percent",
            "host_rss_bytes",
            "gpu_utilisation_percent",
            "memory_controller_utilisation_percent",
            "power_watts",
            "gpu_memory_bytes",
        ):
            values = [float(row[field_name]) for row in self.samples if field_name in row]
            result[f"{field_name}_mean"] = statistics.mean(values) if values else 0.0
            result[f"{field_name}_maximum"] = max(values, default=0.0)
        return result


def _environment() -> dict[str, Any]:
    import torch
    import transformers

    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "compute_capability": (
            list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
        ),
    }


def _custom_fast(job: dict[str, Any]) -> dict[str, Any]:
    import torch

    from swarm_inference.model.qwen3_engine import (
        load_qwen3_fast_engine,
        summarise_generation_repeats,
    )
    from swarm_inference.model.qwen3_runtime import Qwen3EngineOptions

    compile_names = {
        "eager": "eager",
        "default": "torch_compile_default",
        "reduce-overhead": "torch_compile_reduce_overhead",
        "max-autotune": "torch_compile_max_autotune",
        "manual-cuda-graph": "manual_cuda_graph",
    }
    inputs = torch.tensor(job["input_token_ids"], dtype=torch.long)
    options = Qwen3EngineOptions.from_values(
        profile="qwen3_fast",
        attention_backend=job["attention_backend"],
        cache_backend=job["cache_backend"],
        cache_dtype=job.get("cache_dtype", "bfloat16"),
        max_sequence_length=int(job["max_sequence_length"]),
        max_batch_size=int(job["batch_size"]),
        compile_mode=compile_names[job["compile_mode"]],
        cuda_graph_batch_sizes=tuple(job["cuda_graph_batch_sizes"]),
        final_worker_sampling=True,
        diagnostic_full_logits=False,
        boundary_diagnostics=False,
        nvtx_enabled=bool(job.get("nvtx_enabled", False)),
    )
    loaded = load_qwen3_fast_engine(
        model_id=job["model_id"],
        model_revision=job["model_revision"],
        model_path=Path(job["model_path"]),
        options=options,
    )
    engine = loaded.engine
    warm_started = time.perf_counter()
    for index in range(int(job["warmup_requests"])):
        engine.generate_batch(
            inputs,
            request_ids=tuple(f"warm-{index}-{member}" for member in range(int(job["batch_size"]))),
            output_tokens=int(job["output_tokens"]),
            scheduler_policy=job["scheduler_policy"],
        )
    warmup_seconds = time.perf_counter() - warm_started
    telemetry = Telemetry(float(job["telemetry_interval_seconds"]))
    telemetry.start()
    measured = []
    try:
        for repeat in range(int(job["repeats"])):
            measured.append(
                engine.generate_batch(
                    inputs,
                    request_ids=tuple(
                        f"measured-{repeat}-{member}" for member in range(int(job["batch_size"]))
                    ),
                    output_tokens=int(job["output_tokens"]),
                    scheduler_policy=job["scheduler_policy"],
                )
            )
    finally:
        telemetry.stop()
    expected = job.get("reference_output_token_ids")
    identity = expected is None or all(result.output_token_ids == expected for result in measured)
    payloads = [result.payload() for result in measured]
    return {
        "status": "PASS" if identity else "FAIL",
        "worker_status": "completed",
        "profile": "qwen3_fast",
        "engine": "custom_fast",
        "model_load_seconds": loaded.model_load_seconds,
        "attention_selection_seconds": loaded.attention_selection_seconds,
        "warmup_seconds": warmup_seconds,
        "attention_backend": engine.stage.attention_backend,
        "attention_backend_evidence": (engine.stage.attention_evidence.payload()),
        "cache_backend": measured[0].metrics.cache_backend,
        "requested_cache_backend": options.cache_backend.value,
        "compile_mode": options.compile_mode.value,
        "cuda_graph_bucket": int(job["batch_size"]),
        "measured_repeats": payloads,
        "statistics": summarise_generation_repeats(measured),
        "decode_statistics": _statistics(
            [item.metrics.decode_output_tokens_per_second for item in measured]
        ),
        "prefill_statistics": _statistics(
            [item.metrics.prefill_tokens_per_second for item in measured]
        ),
        "exact_reference_identity": identity,
        "output_token_hash": _token_hash(measured[0].output_token_ids),
        "output_token_ids": measured[0].output_token_ids,
        "telemetry": telemetry.summary(),
        "telemetry_samples": telemetry.samples,
        "stage_state": engine.stage.state_summary(),
    }


def _build_custom_correctness_stage(job: dict[str, Any]) -> tuple[Any, float]:
    import torch
    from transformers import Qwen3Config

    from swarm_inference.model.qwen3 import Qwen3StageModule
    from swarm_inference.model.qwen3_runtime import Qwen3EngineOptions
    from swarm_inference.model.shard_builder import (
        ResolvedModel,
        build_manifest,
        inspect_native_model,
    )

    started = time.perf_counter()
    path = Path(job["model_path"])
    resolved = ResolvedModel(
        model_id=job["model_id"],
        revision=job["model_revision"],
        path=path,
        downloaded=False,
    )
    description = inspect_native_model(resolved)
    maximum = sum(item.bytes for item in description.tensors) * 2
    manifest = build_manifest(
        description,
        target_stage_bytes=maximum,
        maximum_stage_bytes=maximum,
        stage_count=1,
    )
    stage = Qwen3StageModule(
        config=Qwen3Config.from_pretrained(path, local_files_only=True),
        stage=manifest.stages[0],
        device="cuda:0",
        dtype=torch.bfloat16,
        engine_options=Qwen3EngineOptions.from_values(
            profile="qwen3_correctness",
            max_sequence_length=int(job["max_sequence_length"]),
            max_batch_size=int(job["batch_size"]),
        ),
    )
    stage.load_weights(path, manifest=manifest)
    return stage, time.perf_counter() - started


def _correctness_generate_batch(
    stage: Any,
    input_rows: list[list[int]],
    *,
    request_ids: tuple[str, ...],
    output_tokens: int,
) -> tuple[list[list[int]], dict[str, Any]]:

    from swarm_inference.model.stage_module import (
        BatchExecutionMetadata,
        StageExecutionMetadata,
    )

    if not input_rows:
        raise ValueError("correctness batch requires at least one request")
    prompt_lengths = {len(row) for row in input_rows}
    if len(prompt_lengths) != 1:
        raise ValueError("correctness oracle requires a homogeneous prompt-length batch")
    if len(request_ids) != len(input_rows):
        raise ValueError("correctness request IDs must match the input batch")
    prompt_length = next(iter(prompt_lengths))
    batch_size = len(input_rows)
    torch = stage.torch

    started = time.perf_counter()
    host_inputs = torch.tensor(input_rows, dtype=torch.long)
    cuda_inputs = host_inputs.to(device=stage.device, non_blocking=False)
    output_buffer = torch.empty(
        (batch_size, output_tokens),
        dtype=torch.long,
        device=stage.device,
    )
    prefill_start = torch.cuda.Event(enable_timing=True)
    prefill_end = torch.cuda.Event(enable_timing=True)
    prefill_start.record()
    logits = stage.prefill_batch_cuda(
        cuda_inputs,
        BatchExecutionMetadata(
            requests=tuple(
                StageExecutionMetadata(
                    request_id=request_id,
                    token_position=0,
                    sequence_length=prompt_length,
                )
                for request_id in request_ids
            )
        ),
    )
    output_buffer[:, 0] = torch.argmax(logits[:, -1, :], dim=-1)
    prefill_end.record()
    decode_events: list[tuple[Any, Any]] = []
    for index in range(1, output_tokens):
        decode_start = torch.cuda.Event(enable_timing=True)
        decode_end = torch.cuda.Event(enable_timing=True)
        decode_start.record()
        logits = stage.decode_batch_cuda(
            output_buffer[:, index - 1 : index],
            BatchExecutionMetadata(
                requests=tuple(
                    StageExecutionMetadata(
                        request_id=request_id,
                        token_position=prompt_length + index - 1,
                        sequence_length=1,
                    )
                    for request_id in request_ids
                )
            ),
        )
        output_buffer[:, index] = torch.argmax(logits[:, -1, :], dim=-1)
        decode_end.record()
        decode_events.append((decode_start, decode_end))
    host_outputs = output_buffer.cpu()
    # Correctness is an explicit synchronisation boundary.  No performance
    # result is derived from this profile.
    torch.cuda.current_stream(stage.device).synchronize()
    prefill_ms = float(prefill_start.elapsed_time(prefill_end))
    decode_latencies_ms = [
        float(event_start.elapsed_time(event_end)) for event_start, event_end in decode_events
    ]
    decode_ms = sum(decode_latencies_ms)
    total_seconds = time.perf_counter() - started
    stage.cancel_batch(request_ids)
    output_lists = [[int(value) for value in row] for row in host_outputs.tolist()]
    return output_lists, {
        "profile": "qwen3_correctness",
        "batch_size": batch_size,
        "prefill_ms": prefill_ms,
        "ttft_ms": prefill_ms,
        "prefill_tokens_per_second": (
            batch_size * prompt_length / (prefill_ms / 1000) if prefill_ms else 0.0
        ),
        "decode_ms": decode_ms,
        "decode_output_tokens_per_second": (
            batch_size * max(0, output_tokens - 1) / (decode_ms / 1000) if decode_ms else 0.0
        ),
        "aggregate_verified_output_tokens_per_second": (batch_size * output_tokens / total_seconds),
        "end_to_end_ms": total_seconds * 1000,
        "inter_token_latency_ms_p50": _percentile(decode_latencies_ms, 0.50),
        "inter_token_latency_ms_p95": _percentile(decode_latencies_ms, 0.95),
        "inter_token_latency_ms_p99": _percentile(decode_latencies_ms, 0.99),
        "cuda_synchronisations": 1,
        "full_logits_transferred": False,
        "diagnostic_full_logits": False,
    }


def _custom_correctness(job: dict[str, Any]) -> dict[str, Any]:
    stage, load_seconds = _build_custom_correctness_stage(job)
    inputs = job["input_token_ids"]
    warm_started = time.perf_counter()
    for warmup in range(int(job["warmup_requests"])):
        _correctness_generate_batch(
            stage,
            inputs,
            request_ids=tuple(
                f"correctness-warm-{warmup}-{member}" for member in range(len(inputs))
            ),
            output_tokens=int(job["output_tokens"]),
        )
    warmup_seconds = time.perf_counter() - warm_started
    measured: list[dict[str, Any]] = []
    output_rows: list[list[int]] = []
    for repeat in range(int(job["repeats"])):
        rows, metrics = _correctness_generate_batch(
            stage,
            inputs,
            request_ids=tuple(
                f"correctness-measured-{repeat}-{member}" for member in range(len(inputs))
            ),
            output_tokens=int(job["output_tokens"]),
        )
        if not output_rows:
            output_rows = rows
        measured.append(
            {
                "output_token_ids": rows,
                "metrics": metrics,
            }
        )
    rates = [
        float(item["metrics"]["aggregate_verified_output_tokens_per_second"]) for item in measured
    ]
    repeat_identity = all(item["output_token_ids"] == output_rows for item in measured)
    return {
        "status": "PASS" if repeat_identity else "FAIL",
        "worker_status": "completed",
        "profile": "qwen3_correctness",
        "engine": "custom_correctness",
        "model_load_seconds": load_seconds,
        "warmup_seconds": warmup_seconds,
        "attention_backend": "eager",
        "cache_backend": "dynamic_reference",
        "compile_mode": "eager",
        "measured_repeats": measured,
        "statistics": _statistics(rates),
        "decode_statistics": _statistics(
            [float(item["metrics"]["decode_output_tokens_per_second"]) for item in measured]
        ),
        "prefill_statistics": _statistics(
            [float(item["metrics"]["prefill_tokens_per_second"]) for item in measured]
        ),
        "exact_repeat_identity": repeat_identity,
        "output_token_hash": _token_hash(output_rows),
        "output_token_ids": output_rows,
        "stage_state": stage.state_summary(),
    }


def _transport_microbenchmark(job: dict[str, Any]) -> dict[str, Any]:
    """Measure each explicit stage-boundary tensor path in a disposable context."""

    import torch

    from swarm_inference.transport.tensor_paths import (
        PreallocatedTensorTransport,
        TensorPath,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("transport microbenchmark requires CUDA")
    device = torch.device("cuda:0")
    warmups = int(job["warmup_requests"])
    repeats = int(job["repeats"])
    transfers_per_repeat = int(job.get("transfers_per_repeat", 50))
    telemetry = Telemetry(float(job["telemetry_interval_seconds"]))
    points: list[dict[str, Any]] = []
    telemetry.start()
    try:
        for shape_values in job["tensor_shapes"]:
            shape = tuple(int(value) for value in shape_values)
            element_count = 1
            for value in shape:
                element_count *= value
            source = (
                torch.arange(element_count, dtype=torch.float32, device=device)
                .remainder_(257)
                .reshape(shape)
                .to(torch.bfloat16)
            )
            logical_bytes = int(source.numel() * source.element_size())
            for path in TensorPath:
                transport = PreallocatedTensorTransport(
                    torch_module=torch,
                    device=device,
                    path=path,
                    profile="qwen3_fast",
                    nvtx_enabled=bool(job.get("nvtx_enabled", False)),
                )
                restored = source
                for _ in range(warmups):
                    restored = transport.transfer(source)
                torch.cuda.current_stream(device).synchronize()
                before = transport.metrics.payload()
                repeat_ms: list[float] = []
                exact = True
                direct_reference = True
                for _ in range(repeats):
                    torch.cuda.current_stream(device).synchronize()
                    started = time.perf_counter()
                    for _ in range(transfers_per_repeat):
                        restored = transport.transfer(source)
                        direct_reference = direct_reference and (
                            restored is source if path == TensorPath.IN_PROCESS_GPU else True
                        )
                    torch.cuda.current_stream(device).synchronize()
                    repeat_ms.append((time.perf_counter() - started) * 1000 / transfers_per_repeat)
                    exact = exact and bool(
                        torch.equal(
                            restored.view(torch.uint16),
                            source.view(torch.uint16),
                        )
                    )
                after = transport.metrics.payload()
                deltas = {
                    field_name: float(after[field_name]) - float(before[field_name])
                    for field_name in (
                        "host_to_device_bytes",
                        "device_to_host_bytes",
                        "serialised_bytes",
                        "serialisation_ms",
                        "deserialisation_ms",
                        "explicit_synchronisations",
                        "buffer_allocations",
                    )
                }

                measured_transfers = repeats * transfers_per_repeat
                points.append(
                    {
                        "profile": "qwen3_fast",
                        "path": path.value,
                        "selected_method": after["selected_method"],
                        "shape": list(shape),
                        "logical_bytes_per_transfer": logical_bytes,
                        "warmup_transfers": warmups,
                        "measured_repeats": repeats,
                        "transfers_per_repeat": transfers_per_repeat,
                        "measured_transfers": measured_transfers,
                        "latency_ms": _statistics(repeat_ms),
                        "effective_logical_gigabytes_per_second": (
                            (logical_bytes / 1_000_000_000) / (statistics.median(repeat_ms) / 1000)
                            if statistics.median(repeat_ms)
                            else 0.0
                        ),
                        "host_to_device_bytes_total": int(deltas["host_to_device_bytes"]),
                        "device_to_host_bytes_total": int(deltas["device_to_host_bytes"]),
                        "serialised_bytes_total": int(deltas["serialised_bytes"]),
                        "serialisation_ms_total": deltas["serialisation_ms"],
                        "deserialisation_ms_total": deltas["deserialisation_ms"],
                        "explicit_transport_synchronisations_total": int(
                            deltas["explicit_synchronisations"]
                        ),
                        "buffer_allocations_during_measurement": int(deltas["buffer_allocations"]),
                        "host_to_device_bytes_per_transfer": (
                            deltas["host_to_device_bytes"] / measured_transfers
                        ),
                        "device_to_host_bytes_per_transfer": (
                            deltas["device_to_host_bytes"] / measured_transfers
                        ),
                        "serialised_bytes_per_transfer": (
                            deltas["serialised_bytes"] / measured_transfers
                        ),
                        "explicit_transport_synchronisations_per_transfer": (
                            deltas["explicit_synchronisations"] / measured_transfers
                        ),
                        "bfloat16_bits_exact": exact,
                        "direct_tensor_reference": (
                            direct_reference if path == TensorPath.IN_PROCESS_GPU else False
                        ),
                    }
                )
    finally:
        telemetry.stop()
    exact = all(bool(point["bfloat16_bits_exact"]) for point in points)
    local_direct = all(
        bool(point["direct_tensor_reference"])
        for point in points
        if point["path"] == TensorPath.IN_PROCESS_GPU.value
    )
    return {
        "status": "PASS" if exact and local_direct else "FAIL",
        "engine": "transport_paths",
        "profile": "qwen3_fast",
        "attention_backend": "not_applicable",
        "cache_backend": "not_applicable",
        "compile_mode": "not_applicable",
        "exact_bfloat16_identity": exact,
        "gpu_resident_direct_reference_verified": local_direct,
        "paths": points,
        "telemetry": telemetry.summary(),
    }


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    engine = str(job["engine"])
    if engine == "custom_fast":
        return _custom_fast(job)
    if engine == "custom_correctness":
        return _custom_correctness(job)
    if engine == "transport_paths":
        return _transport_microbenchmark(job)
    raise ValueError(f"unsupported project benchmark engine {engine!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    job = json.loads(arguments.job.read_text(encoding="utf-8"))
    started = time.time()
    try:
        result = run_job(job)
        result["worker_status"] = "completed"
        return_code = 0 if result["status"] == "PASS" else 1
    except Exception as exc:
        result = {
            "status": "FAIL",
            "worker_status": "failed",
            "profile": job.get("profile", "unknown"),
            "engine": job.get("engine", "unknown"),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        return_code = 1
    result["job"] = job
    result["environment"] = _environment()
    result["started_unix_seconds"] = started
    result["finished_unix_seconds"] = time.time()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
