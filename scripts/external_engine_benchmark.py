"""Standalone Hugging Face benchmark used by Experiment 004 environments.

This file deliberately imports no project modules so an isolated environment
can execute it without installing the repository.
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


@dataclass(slots=True)
class Telemetry:
    interval_seconds: float
    samples: list[dict[str, float]] = field(default_factory=list)
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None

    def start(self) -> None:
        import psutil

        process = psutil.Process()

        def sample() -> None:
            process.cpu_percent(interval=None)
            while not self.stop_event.wait(self.interval_seconds):
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
            name="hf-engine-telemetry",
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


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((len(values) - 1) * quantile)))
    return values[index]


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


def token_hash(rows: list[list[int]]) -> str:
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode("utf-8")).hexdigest()


def generate(
    model: Any,
    torch: Any,
    host_inputs: Any,
    *,
    output_tokens: int,
    cache_backend: str,
    max_sequence_length: int,
    logits_to_keep: int,
) -> tuple[list[list[int]], dict[str, Any]]:
    batch_size, prompt_length = (int(value) for value in host_inputs.shape)
    request_started = time.perf_counter()
    host_to_device_bytes = int(host_inputs.numel() * host_inputs.element_size())
    h2d_start = torch.cuda.Event(enable_timing=True)
    h2d_end = torch.cuda.Event(enable_timing=True)
    h2d_start.record()
    cuda_inputs = host_inputs.to("cuda", non_blocking=False)
    h2d_end.record()
    output_buffer = torch.empty((batch_size, output_tokens), dtype=torch.long, device="cuda")
    selected_logits = torch.empty((batch_size, output_tokens), dtype=torch.bfloat16, device="cuda")
    position_buffer = torch.arange(max_sequence_length, dtype=torch.long, device="cuda")
    cache = None
    if cache_backend == "static":
        from transformers import StaticCache

        cache = StaticCache(model.config, max_cache_len=max_sequence_length)
    prefill_start = torch.cuda.Event(enable_timing=True)
    prefill_end = torch.cuda.Event(enable_timing=True)
    prefill_start.record()
    kwargs: dict[str, Any] = {
        "input_ids": cuda_inputs,
        "use_cache": True,
        "return_dict": True,
        # Official Qwen3 supports projecting only the newest hidden state.
        # Production decoding never consumes prompt-position logits, and
        # retaining them can allocate roughly 10 GiB at the required
        # 2048-token, batch-16 point.
        "logits_to_keep": logits_to_keep,
    }
    if cache is not None:
        kwargs["past_key_values"] = cache
        kwargs["cache_position"] = position_buffer[:prompt_length]
    if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
        torch.compiler.cudagraph_mark_step_begin()
    output = model(**kwargs)
    cache = output.past_key_values
    first_scores = output.logits[:, -1, :]
    output_buffer[:, 0] = torch.argmax(first_scores, dim=-1)
    selected_logits[:, 0] = first_scores.gather(1, output_buffer[:, 0, None]).squeeze(1)
    prefill_end.record()
    decode_events: list[tuple[Any, Any]] = []
    sampling_events: list[tuple[Any, Any]] = []
    for index in range(1, output_tokens):
        decode_start = torch.cuda.Event(enable_timing=True)
        decode_end = torch.cuda.Event(enable_timing=True)
        decode_start.record()
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        output = model(
            input_ids=output_buffer[:, index - 1 : index],
            past_key_values=cache,
            cache_position=position_buffer[prompt_length + index - 1 : prompt_length + index],
            use_cache=True,
            return_dict=True,
            logits_to_keep=logits_to_keep,
        )
        decode_end.record()
        decode_events.append((decode_start, decode_end))
        sample_start = torch.cuda.Event(enable_timing=True)
        sample_end = torch.cuda.Event(enable_timing=True)
        sample_start.record()
        scores = output.logits[:, -1, :]
        output_buffer[:, index] = torch.argmax(scores, dim=-1)
        selected_logits[:, index] = scores.gather(1, output_buffer[:, index, None]).squeeze(1)
        sample_end.record()
        sampling_events.append((sample_start, sample_end))
    copy_start = torch.cuda.Event(enable_timing=True)
    copy_end = torch.cuda.Event(enable_timing=True)
    copy_start.record()
    host_outputs = output_buffer.cpu()
    host_selected = selected_logits.float().cpu()
    copy_end.record()
    copy_end.synchronize()
    h2d_ms = float(h2d_start.elapsed_time(h2d_end))
    prefill_ms = float(prefill_start.elapsed_time(prefill_end))
    decode_latencies = [float(start.elapsed_time(end)) for start, end in decode_events]
    decode_ms = sum(decode_latencies)
    sampling_ms = sum(float(start.elapsed_time(end)) for start, end in sampling_events)
    copy_ms = float(copy_start.elapsed_time(copy_end))
    rows = [[int(value) for value in row] for row in host_outputs.tolist()]
    selected_rows = host_selected.tolist()
    end_to_end_ms = (time.perf_counter() - request_started) * 1000
    gpu_kernel_and_transfer_ms = h2d_ms + prefill_ms + decode_ms + sampling_ms + copy_ms
    output_count = batch_size * output_tokens
    decode_count = batch_size * max(output_tokens - 1, 0)
    del cache
    return rows, {
        "profile": (
            "huggingface_eager"
            if model.config._attn_implementation == "eager"
            else "huggingface_optimised"
        ),
        "attention_backend": model.config._attn_implementation,
        "cache_backend": cache_backend,
        "batch_size": batch_size,
        "prompt_tokens_per_request": prompt_length,
        "output_tokens_per_request": output_tokens,
        "host_to_device_bytes": host_to_device_bytes,
        "device_to_host_bytes": int(
            output_buffer.numel() * output_buffer.element_size()
            + len(selected_rows) * (len(selected_rows[0]) if selected_rows else 0) * 4
        ),
        "cuda_synchronisations": 1,
        "host_to_device_ms": h2d_ms,
        "prefill_ms": prefill_ms,
        "prefill_tokens_per_second": (batch_size * prompt_length / (prefill_ms / 1000)),
        "decode_ms": decode_ms,
        "decode_output_tokens_per_second": (
            decode_count / (decode_ms / 1000) if decode_ms else 0.0
        ),
        "aggregate_verified_output_tokens_per_second": (output_count / (end_to_end_ms / 1000)),
        "end_to_end_ms": end_to_end_ms,
        "sampling_ms": sampling_ms,
        "device_to_host_ms": copy_ms,
        "gpu_kernel_and_transfer_ms": gpu_kernel_and_transfer_ms,
        "scheduler_ms": 0.0,
        "queue_wait_ms": 0.0,
        "serialisation_ms": 0.0,
        "tokenisation_ms": 0.0,
        "inter_token_latency_ms_p50": percentile(decode_latencies, 0.50),
        "inter_token_latency_ms_p95": percentile(decode_latencies, 0.95),
        "inter_token_latency_ms_p99": percentile(decode_latencies, 0.99),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "full_logits_transferred": False,
        "prefill_mode": "homogeneous_full_prompt",
        "chunked_prefill_supported": False,
        "kernels_per_decode_token": None,
        "kernel_count_status": "unavailable_nsight_systems_not_installed",
    }


def run(job: dict[str, Any]) -> dict[str, Any]:
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in isolated Hugging Face environment")
    accumulation_mode = str(job.get("matmul_accumulation", "fp32"))
    if accumulation_mode not in {"fp32", "reduced_precision"}:
        raise ValueError(f"unsupported matmul_accumulation {accumulation_mode!r}")
    allow_reduced_precision = accumulation_mode == "reduced_precision"
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = allow_reduced_precision
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = allow_reduced_precision
    attention_backend = str(job["attention_backend"])
    load_started = time.perf_counter()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        job["model_path"],
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=attention_backend,
    ).to("cuda")
    model.eval()
    model.requires_grad_(False)
    model_load_seconds = time.perf_counter() - load_started
    compile_mode = str(job.get("compile_mode", "eager"))
    compile_diagnostics: dict[str, Any] = {
        "requested_mode": compile_mode,
        "compiled": False,
        "compile_seconds": 0.0,
        "fallback_used": False,
        "fallback_reason": None,
    }
    if compile_mode != "eager":
        mode = {
            "default": None,
            "reduce-overhead": "reduce-overhead",
            "max-autotune": "max-autotune",
        }[compile_mode]
        compile_started = time.perf_counter()
        try:
            kwargs: dict[str, Any] = {"fullgraph": False, "dynamic": False}
            if mode is not None:
                kwargs["mode"] = mode
            model.forward = torch.compile(model.forward, **kwargs)
            compile_diagnostics["compiled"] = True
        except Exception as exc:
            compile_diagnostics["fallback_used"] = True
            compile_diagnostics["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        compile_diagnostics["compile_seconds"] = time.perf_counter() - compile_started
    host_inputs = torch.tensor(job["input_token_ids"], dtype=torch.long)
    warm_started = time.perf_counter()
    warmup_request_seconds: list[float] = []
    for _ in range(int(job["warmup_requests"])):
        request_started = time.perf_counter()
        generate(
            model,
            torch,
            host_inputs,
            output_tokens=int(job["output_tokens"]),
            cache_backend=job["cache_backend"],
            max_sequence_length=int(job["max_sequence_length"]),
            logits_to_keep=int(job.get("logits_to_keep", 1)),
        )
        warmup_request_seconds.append(time.perf_counter() - request_started)
    warmup_seconds = time.perf_counter() - warm_started
    if compile_diagnostics["compiled"] and warmup_request_seconds:
        compile_diagnostics["compile_and_first_warmup_seconds"] = warmup_request_seconds[0]
    compile_diagnostics["warmup_request_seconds"] = warmup_request_seconds
    measured: list[dict[str, Any]] = []
    output_rows: list[list[int]] = []
    telemetry = Telemetry(float(job["telemetry_interval_seconds"]))
    telemetry.start()
    try:
        for _ in range(int(job["repeats"])):
            torch.cuda.reset_peak_memory_stats()
            rows, metrics = generate(
                model,
                torch,
                host_inputs,
                output_tokens=int(job["output_tokens"]),
                cache_backend=job["cache_backend"],
                max_sequence_length=int(job["max_sequence_length"]),
                logits_to_keep=int(job.get("logits_to_keep", 1)),
            )
            if not output_rows:
                output_rows = rows
            measured.append({"output_token_ids": rows, "metrics": metrics})
    finally:
        telemetry.stop()
    reference = job.get("reference_output_token_ids")
    identity = reference is None or all(item["output_token_ids"] == reference for item in measured)
    rates = [
        float(item["metrics"]["aggregate_verified_output_tokens_per_second"]) for item in measured
    ]
    decode_rates = [float(item["metrics"]["decode_output_tokens_per_second"]) for item in measured]
    return {
        "status": "PASS" if identity else "FAIL",
        "worker_status": "completed",
        "profile": (
            "huggingface_eager" if job["engine"] == "huggingface_eager" else "huggingface_optimised"
        ),
        "engine": job["engine"],
        "model_load_seconds": model_load_seconds,
        "warmup_seconds": warmup_seconds,
        "attention_backend": attention_backend,
        "cache_backend": job["cache_backend"],
        "compile_mode": compile_mode,
        "matmul_accumulation": accumulation_mode,
        "compile_diagnostics": compile_diagnostics,
        "measured_repeats": measured,
        "statistics": stats(rates),
        "decode_statistics": stats(decode_rates),
        "exact_reference_identity": identity,
        "output_token_hash": token_hash(output_rows),
        "output_token_ids": output_rows,
        "telemetry": telemetry.summary(),
        "telemetry_samples": telemetry.samples,
        "environment": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
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
            "profile": job.get("profile", "unknown"),
            "engine": job.get("engine", "unknown"),
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
