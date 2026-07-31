"""Standalone SGLang offline-engine benchmark for Experiment 004 Linux runs."""

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
from itertools import pairwise
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Telemetry:
    interval_seconds: float
    samples: list[dict[str, float]] = field(default_factory=list)
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            import psutil

            process = psutil.Process()
        except ImportError:
            process = None

        def sample() -> None:
            if process is not None:
                process.cpu_percent(interval=None)
            while not self.stop_event.wait(self.interval_seconds):
                row: dict[str, float] = {"time_monotonic": time.monotonic()}
                if process is not None:
                    row["host_cpu_percent"] = process.cpu_percent(interval=None)
                    row["host_rss_bytes"] = float(process.memory_info().rss)
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
                except (
                    OSError,
                    ValueError,
                    IndexError,
                    subprocess.TimeoutExpired,
                ):
                    pass
                self.samples.append(row)

        self.thread = threading.Thread(
            target=sample,
            name="sglang-engine-telemetry",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)

    def summary(self) -> dict[str, float | int]:
        summary: dict[str, float | int] = {"sample_count": len(self.samples)}
        for name in (
            "host_cpu_percent",
            "host_rss_bytes",
            "gpu_utilisation_percent",
            "memory_controller_utilisation_percent",
            "power_watts",
            "gpu_memory_bytes",
        ):
            values = [float(row[name]) for row in self.samples if name in row]
            summary[f"{name}_mean"] = statistics.mean(values) if values else 0.0
            summary[f"{name}_maximum"] = max(values, default=0.0)
        return summary


def stats(values: list[float]) -> dict[str, float]:
    mean = statistics.mean(values)
    deviation = statistics.pstdev(values)
    return {
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "standard_deviation": deviation,
        "coefficient_of_variation": deviation / mean if mean else 0.0,
    }


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return ordered[index]


def token_hash(rows: list[list[int]]) -> str:
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode("utf-8")).hexdigest()


def generated_ids(output: dict[str, Any]) -> list[int]:
    direct = output.get("output_ids")
    if direct is not None:
        return [int(value) for value in direct]
    meta = output.get("meta_info", {})
    for key in ("output_ids", "completion_token_ids"):
        if key in meta:
            return [int(value) for value in meta[key]]
    raise RuntimeError(
        "SGLang response did not expose output token IDs; "
        f"available keys={sorted(output)} meta={sorted(meta)}"
    )


def run(job: dict[str, Any]) -> dict[str, Any]:
    import sglang as sgl
    import torch

    requested_attention_backend = str(job.get("attention_backend", "auto"))
    engine_options: dict[str, Any] = {}
    if requested_attention_backend != "auto":
        engine_options["attention_backend"] = requested_attention_backend
    if bool(job.get("deterministic_inference", False)):
        engine_options["enable_deterministic_inference"] = True
    if job.get("model_impl") is not None:
        engine_options["model_impl"] = str(job["model_impl"])
    load_started = time.perf_counter()
    engine = sgl.Engine(
        model_path=job["model_path"],
        dtype="bfloat16",
        skip_tokenizer_init=True,
        random_seed=int(job.get("seed", 4)),
        max_running_requests=int(job["batch_size"]),
        mem_fraction_static=float(job.get("mem_fraction_static", 0.70)),
        cuda_graph_max_bs_decode=max(int(job["batch_size"]), 1),
        disable_radix_cache=not bool(job.get("prefix_reuse_enabled", False)),
        log_level="warning",
        **engine_options,
    )
    model_load_seconds = time.perf_counter() - load_started
    sampling = {
        "temperature": 0.0,
        "max_new_tokens": int(job["output_tokens"]),
        "ignore_eos": True,
        "skip_special_tokens": False,
    }

    generation_index = 0

    def generate_once() -> tuple[list[list[int]], dict[str, Any]]:
        nonlocal generation_index
        request_ids = [
            f"experiment-004-{generation_index}-{index}" for index in range(int(job["batch_size"]))
        ]
        generation_index += 1
        started = time.perf_counter()
        stream = engine.generate(
            input_ids=job["input_token_ids"],
            sampling_params=sampling,
            stream=True,
            rid=request_ids,
        )
        latest_by_request: dict[str, dict[str, Any]] = {}
        arrivals: dict[str, list[float]] = {request_id: [] for request_id in request_ids}
        previous_lengths = {request_id: 0 for request_id in request_ids}
        first_chunk_at: float | None = None
        last_chunk_at: float | None = None
        for chunk in stream:
            observed_at = time.perf_counter()
            if first_chunk_at is None:
                first_chunk_at = observed_at
            last_chunk_at = observed_at
            payloads = chunk if isinstance(chunk, list) else [chunk]
            for fallback_index, output in enumerate(payloads):
                metadata = output.get("meta_info", {})
                request_id = str(
                    metadata.get("id")
                    or output.get("id")
                    or request_ids[min(fallback_index, len(request_ids) - 1)]
                )
                if request_id not in arrivals:
                    request_id = request_ids[min(fallback_index, len(request_ids) - 1)]
                latest_by_request[request_id] = output
                current_length = len(generated_ids(output))
                new_tokens = max(0, current_length - previous_lengths[request_id])
                arrivals[request_id].extend([observed_at] * new_tokens)
                previous_lengths[request_id] = current_length
        outputs = [latest_by_request[request_id] for request_id in request_ids]
        elapsed = time.perf_counter() - started
        rows = [generated_ids(output) for output in outputs]
        metadata = [output.get("meta_info", {}) for output in outputs]
        output_count = sum(len(row) for row in rows)
        inter_token_latencies = [
            (right - left) * 1000
            for request_arrivals in arrivals.values()
            for left, right in pairwise(request_arrivals)
        ]
        decode_span = (
            last_chunk_at - first_chunk_at
            if first_chunk_at is not None and last_chunk_at is not None
            else 0.0
        )
        decode_tokens = sum(max(0, len(row) - 1) for row in rows)
        decode_rate = decode_tokens / decode_span if decode_span else 0.0
        ttft_ms = (first_chunk_at - started) * 1000 if first_chunk_at is not None else 0.0
        return rows, {
            "profile": "sglang",
            "batch_size": int(job["batch_size"]),
            "prompt_tokens_per_request": len(job["input_token_ids"][0]),
            "output_tokens_per_request": int(job["output_tokens"]),
            "aggregate_verified_output_tokens_per_second": (output_count / elapsed),
            "end_to_end_ms": elapsed * 1000,
            "queue_wait_ms": sum(float(item.get("queue_time") or 0.0) for item in metadata),
            "ttft_ms": ttft_ms,
            "prefill_ms": 0.0,
            "prefill_tokens_per_second": 0.0,
            "prefill_measurement_status": "unavailable_from_sglang_engine_api",
            "decode_output_tokens_per_second": decode_rate,
            "inter_token_latency_ms_p50": (
                statistics.median(inter_token_latencies) if inter_token_latencies else 0.0
            ),
            "inter_token_latency_ms_p95": percentile(inter_token_latencies, 0.95),
            "inter_token_latency_ms_p99": percentile(inter_token_latencies, 0.99),
            "sampling_ms": 0.0,
            "scheduler_ms": 0.0,
            "serialisation_ms": 0.0,
            "tokenisation_ms": 0.0,
            "host_to_device_bytes": 0,
            "device_to_host_bytes": 0,
            "full_logits_transferred": False,
            "cached_prompt_tokens": sum(int(item.get("cached_tokens") or 0) for item in metadata),
            "prefill_mode": "engine_managed_chunked_or_radix_cache",
            "chunked_prefill_supported": True,
            "kernels_per_decode_token": None,
            "kernel_count_status": "unavailable_nsight_systems_not_installed",
            "cuda_graph_capture_ms": None,
            "cuda_graph_verified": False,
            "cuda_graph_status": "enabled_but_capture_cost_and_replay_not_exposed_by_engine_api",
            "streaming_timing_included": True,
            "request_metadata": metadata,
            "unavailable_measurements": [
                "prefill_time_separate_from_ttft",
                "prefill_tokens_per_second",
                "sampling_time",
                "scheduler_time",
                "host_device_bytes",
                "kernels_per_decode_token",
                "cuda_graph_capture_time",
            ],
        }

    warm_started = time.perf_counter()
    for _ in range(int(job["warmup_requests"])):
        generate_once()
    warmup_seconds = time.perf_counter() - warm_started
    engine.flush_cache()
    measured = []
    first_rows: list[list[int]] = []
    telemetry = Telemetry(float(job["telemetry_interval_seconds"]))
    telemetry.start()
    try:
        for _ in range(int(job["repeats"])):
            if bool(job.get("prefix_reuse_enabled", False)):
                engine.flush_cache()
            rows, metrics = generate_once()
            if not first_rows:
                first_rows = rows
            measured.append({"output_token_ids": rows, "metrics": metrics})
    finally:
        telemetry.stop()
    reference = job.get("reference_output_token_ids")
    identity = reference is None or all(item["output_token_ids"] == reference for item in measured)
    rates = [
        float(item["metrics"]["aggregate_verified_output_tokens_per_second"]) for item in measured
    ]
    decode_rates = [float(item["metrics"]["decode_output_tokens_per_second"]) for item in measured]
    engine.shutdown()
    return {
        "status": "PASS" if identity else "FAIL",
        "worker_status": "completed",
        "profile": "sglang",
        "engine": "sglang",
        "attention_backend": requested_attention_backend,
        "deterministic_inference": bool(job.get("deterministic_inference", False)),
        "model_impl": str(job.get("model_impl", "auto")),
        "model_load_seconds": model_load_seconds,
        "warmup_seconds": warmup_seconds,
        "compile_diagnostics": {
            "requested_mode": "cuda-graph",
            "compile_seconds": None,
            "cuda_graph_capture_seconds": None,
            "capture_status": "included_in_model_load_not_separately_exposed",
        },
        "measured_repeats": measured,
        "statistics": stats(rates),
        "decode_statistics": stats(decode_rates),
        "exact_reference_identity": identity,
        "output_token_hash": token_hash(first_rows),
        "output_token_ids": first_rows,
        "telemetry": telemetry.summary(),
        "telemetry_samples": telemetry.samples,
        "environment": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "sglang_version": getattr(sgl, "__version__", None),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    job = json.loads(arguments.job.read_text(encoding="utf-8"))
    try:
        payload = run(job)
        code = 0 if payload["status"] == "PASS" else 1
    except Exception as exc:
        payload = {
            "status": "FAIL",
            "worker_status": "failed",
            "profile": "sglang",
            "engine": "sglang",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        code = 1
    payload["job"] = job
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
