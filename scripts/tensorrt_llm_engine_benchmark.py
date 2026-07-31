"""Standalone TensorRT-LLM Qwen3 benchmark for Experiment 004."""

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
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,utilization.memory,power.draw,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
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
        payload: dict[str, float] = {"sample_count": float(len(self.samples))}
        for name in (
            "gpu_utilisation_percent",
            "memory_controller_utilisation_percent",
            "power_watts",
            "gpu_memory_bytes",
        ):
            values = [float(row[name]) for row in self.samples if name in row]
            payload[f"{name}_mean"] = statistics.mean(values) if values else 0.0
            payload[f"{name}_maximum"] = max(values, default=0.0)
        return payload


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
    encoded = json.dumps(rows, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run(job: dict[str, Any]) -> dict[str, Any]:
    import tensorrt_llm
    import torch
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    batch_size = int(job["batch_size"])
    max_sequence_length = int(job["max_sequence_length"])
    max_num_tokens = max(
        max_sequence_length,
        sum(len(row) for row in job["input_token_ids"]),
    )
    kv_max_tokens = ((max_num_tokens + 31) // 32) * 32
    generation_path = Path(job["model_path"]) / "generation_config.json"
    generation = (
        json.loads(generation_path.read_text(encoding="utf-8")) if generation_path.is_file() else {}
    )
    eos_token_id = generation.get("eos_token_id", 151645)
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]
    pad_token_id = int(generation.get("pad_token_id", 151643))
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    llm = LLM(
        model=job["model_path"],
        tokenizer=job["model_path"],
        skip_tokenizer_init=True,
        dtype="bfloat16",
        backend="pytorch",
        max_seq_len=max_sequence_length,
        max_batch_size=batch_size,
        max_num_tokens=max_num_tokens,
        enable_chunked_prefill=True,
        disable_overlap_scheduler=False,
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=bool(job.get("prefix_reuse_enabled", False)),
            free_gpu_memory_fraction=0.70,
            max_tokens=kv_max_tokens,
            # TensorRT-LLM uses "auto" for an unquantised cache that follows
            # the model dtype; its explicit override accepts quantised formats only.
            dtype="auto",
        ),
        cuda_graph_config=CudaGraphConfig(
            batch_sizes=[batch_size],
            enable_padding=False,
        ),
        trust_remote_code=False,
    )
    model_load_seconds = time.perf_counter() - load_started
    sampling = SamplingParams(
        end_id=int(eos_token_id),
        pad_id=pad_token_id,
        max_tokens=int(job["output_tokens"]),
        temperature=0.0,
        top_k=1,
        seed=int(job.get("seed", 4)),
        ignore_eos=True,
        detokenize=False,
        add_special_tokens=False,
        return_perf_metrics=True,
    )
    prompts = [[int(token) for token in row] for row in job["input_token_ids"]]

    def generate_once() -> tuple[list[list[int]], dict[str, Any]]:
        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        elapsed = time.perf_counter() - started
        if not isinstance(outputs, list):
            outputs = [outputs]
        rows = [[int(token) for token in output.outputs[0].token_ids] for output in outputs]
        output_count = sum(len(row) for row in rows)
        return rows, {
            "profile": "tensorrt_llm",
            "batch_size": batch_size,
            "prompt_tokens_per_request": len(prompts[0]),
            "output_tokens_per_request": int(job["output_tokens"]),
            "aggregate_verified_output_tokens_per_second": output_count / elapsed,
            "decode_output_tokens_per_second": 0.0,
            "decode_measurement_status": ("unavailable_from_synchronous_tensorrt_llm_api"),
            "end_to_end_ms": elapsed * 1000,
            "ttft_ms": 0.0,
            "ttft_measurement_status": ("unavailable_from_synchronous_tensorrt_llm_api"),
            "queue_wait_ms": 0.0,
            "prefill_ms": 0.0,
            "prefill_tokens_per_second": 0.0,
            "prefill_measurement_status": ("unavailable_from_synchronous_tensorrt_llm_api"),
            "inter_token_latency_ms_p50": 0.0,
            "inter_token_latency_ms_p95": 0.0,
            "inter_token_latency_ms_p99": 0.0,
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
            "cuda_graph_status": ("enabled_but_capture_cost_and_replay_not_exposed_by_llm_api"),
            "unavailable_measurements": [
                "prefill_time_separate_from_ttft",
                "decode_time_separate_from_request_latency",
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
    measured: list[dict[str, Any]] = []
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
    result = {
        "status": "PASS" if identity else "FAIL",
        "worker_status": "completed",
        "profile": "tensorrt_llm",
        "engine": "tensorrt_llm",
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
        "decode_statistics": stats([0.0 for _ in rates]),
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
            "tensorrt_llm_version": getattr(tensorrt_llm, "__version__", None),
        },
    }
    shutdown = getattr(llm, "shutdown", None)
    if callable(shutdown):
        shutdown()
    return result


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
            "profile": "tensorrt_llm",
            "engine": "tensorrt_llm",
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
