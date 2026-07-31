"""Standalone vLLM offline-engine benchmark for Experiment 004 Linux runs."""

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
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2)

    def _sample(self) -> None:
        while not self.stop_event.is_set():
            observed = {"timestamp": time.time()}
            command = [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,power.draw,memory.used",
                "--format=csv,noheader,nounits",
            ]
            result = subprocess.run(command, check=False, capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                values = [item.strip() for item in result.stdout.splitlines()[0].split(",")]
                if len(values) == 4:
                    observed.update(
                        {
                            "gpu_utilisation_percent": float(values[0]),
                            "memory_controller_utilisation_percent": float(values[1]),
                            "power_watts": float(values[2]),
                            "gpu_memory_bytes": float(values[3]) * 1024 * 1024,
                        }
                    )
            self.samples.append(observed)
            self.stop_event.wait(self.interval_seconds)

    def summary(self) -> dict[str, float]:
        summary: dict[str, float] = {"sample_count": float(len(self.samples))}
        for name in (
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


def metric_seconds(metrics: Any, name: str) -> float | None:
    value = getattr(metrics, name, None)
    return float(value) if value is not None else None


def first_metric_seconds(metrics: Any, *names: str) -> float | None:
    for name in names:
        value = metric_seconds(metrics, name)
        if value is not None:
            return value
    return None


def run(job: dict[str, Any]) -> dict[str, Any]:
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    load_started = time.perf_counter()
    llm = LLM(
        model=job["model_path"],
        tokenizer=job["model_path"],
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=int(job["max_sequence_length"]),
        max_num_seqs=int(job["batch_size"]),
        enable_prefix_caching=bool(job.get("prefix_reuse_enabled", False)),
        enforce_eager=False,
        seed=int(job.get("seed", 4)),
        gpu_memory_utilization=float(job.get("mem_fraction_static", 0.70)),
        trust_remote_code=False,
    )
    model_load_seconds = time.perf_counter() - load_started
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=int(job["output_tokens"]),
        ignore_eos=True,
        detokenize=False,
        seed=int(job.get("seed", 4)),
    )
    prompts = [{"prompt_token_ids": row} for row in job["input_token_ids"]]

    def generate_once() -> tuple[list[list[int]], dict[str, Any]]:
        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        elapsed = time.perf_counter() - started
        rows = [[int(token) for token in output.outputs[0].token_ids] for output in outputs]
        output_count = sum(len(row) for row in rows)
        request_metrics = [output.metrics for output in outputs]
        ttft_values = []
        queue_values = []
        decode_spans = []
        for metrics, row in zip(request_metrics, rows, strict=True):
            arrival = metric_seconds(metrics, "arrival_time")
            first = first_metric_seconds(metrics, "first_token_ts", "first_token_time")
            last = first_metric_seconds(metrics, "last_token_ts", "last_token_time")
            scheduled = first_metric_seconds(metrics, "scheduled_ts", "first_scheduled_time")
            if arrival is not None and first is not None:
                ttft_values.append((first - arrival) * 1000)
            if arrival is not None and scheduled is not None:
                queue_values.append((scheduled - arrival) * 1000)
            if first is not None and last is not None and len(row) > 1:
                decode_spans.append(last - first)
        decode_tokens = sum(max(0, len(row) - 1) for row in rows)
        decode_span = max(decode_spans, default=0.0)
        per_request_itl = [
            span * 1000 / max(1, len(row) - 1)
            for span, row in zip(decode_spans, rows, strict=False)
        ]
        return rows, {
            "profile": "vllm",
            "batch_size": int(job["batch_size"]),
            "prompt_tokens_per_request": len(job["input_token_ids"][0]),
            "output_tokens_per_request": int(job["output_tokens"]),
            "aggregate_verified_output_tokens_per_second": output_count / elapsed,
            "end_to_end_ms": elapsed * 1000,
            "ttft_ms": statistics.median(ttft_values) if ttft_values else 0.0,
            "queue_wait_ms": sum(queue_values),
            "prefill_ms": 0.0,
            "prefill_tokens_per_second": 0.0,
            "prefill_measurement_status": "unavailable_from_vllm_offline_api",
            "decode_output_tokens_per_second": (
                decode_tokens / decode_span if decode_span else 0.0
            ),
            "inter_token_latency_ms_p50": percentile(per_request_itl, 0.50),
            "inter_token_latency_ms_p95": percentile(per_request_itl, 0.95),
            "inter_token_latency_ms_p99": percentile(per_request_itl, 0.99),
            "sampling_ms": 0.0,
            "scheduler_ms": 0.0,
            "serialisation_ms": 0.0,
            "tokenisation_ms": 0.0,
            "host_to_device_bytes": 0,
            "device_to_host_bytes": 0,
            "full_logits_transferred": False,
            "prefill_mode": "engine_managed_chunked",
            "chunked_prefill_supported": True,
            "kernels_per_decode_token": None,
            "kernel_count_status": "unavailable_nsight_systems_not_installed",
            "cuda_graph_capture_ms": None,
            "cuda_graph_verified": False,
            "cuda_graph_status": "enabled_but_capture_cost_and_replay_not_exposed_by_offline_api",
            "cached_prompt_tokens": sum(int(output.num_cached_tokens or 0) for output in outputs),
            "request_metrics": [
                {
                    name: metric_seconds(metrics, name)
                    for name in (
                        "arrival_time",
                        "queued_ts",
                        "scheduled_ts",
                        "first_token_ts",
                        "last_token_ts",
                        "first_scheduled_time",
                        "first_token_time",
                        "last_token_time",
                        "finished_time",
                        "scheduler_time",
                        "model_forward_time",
                        "model_execute_time",
                    )
                }
                for metrics in request_metrics
            ],
            "unavailable_measurements": [
                "prefill_time_separate_from_ttft",
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
    measured = []
    first_rows: list[list[int]] = []
    telemetry = Telemetry(float(job["telemetry_interval_seconds"]))
    telemetry.start()
    try:
        for _ in range(int(job["repeats"])):
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
    return {
        "status": "PASS" if identity else "FAIL",
        "worker_status": "completed",
        "profile": "vllm",
        "engine": "vllm",
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
            "vllm_version": getattr(vllm, "__version__", None),
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
            "profile": "vllm",
            "engine": "vllm",
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
