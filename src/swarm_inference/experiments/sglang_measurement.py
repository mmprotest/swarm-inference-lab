"""Measured stock-SGLang workloads for the heterogeneous experiment."""

from __future__ import annotations

import json
import re
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path
from typing import Any
from uuid import uuid4

from swarm_inference.backends.sglang import _sglang_output_ids

_BATCH_KIND = re.compile(r"(?P<kind>Prefill|Decode) batch,")
_NEW_SEQUENCE_COUNT = re.compile(r"#new-seq: (?P<value>\d+)")
_RUNNING_REQUEST_COUNT = re.compile(r"#running-req: (?P<value>\d+)")
_QUEUED_REQUEST_COUNT = re.compile(r"#queue-req: (?P<value>\d+)")
_TOKEN_USAGE = re.compile(r"token usage: (?P<value>[0-9.]+)")


def parse_sglang_scheduler_log(
    path: Path,
    *,
    maximum_running_requests: int,
) -> dict[str, Any]:
    """Extract scheduler, batch, CUDA-graph and KV-usage evidence from SGLang logs."""

    records: list[dict[str, Any]] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            kind = _BATCH_KIND.search(line)
            running_match = _RUNNING_REQUEST_COUNT.search(line)
            queued_match = _QUEUED_REQUEST_COUNT.search(line)
            token_usage_match = _TOKEN_USAGE.search(line)
            if (
                kind is None
                or running_match is None
                or queued_match is None
                or token_usage_match is None
            ):
                continue
            new_sequence_match = _NEW_SEQUENCE_COUNT.search(line)
            running_count = int(running_match.group("value"))
            new_sequences = int(new_sequence_match.group("value")) if new_sequence_match else 0
            records.append(
                {
                    "batch_kind": kind.group("kind").lower(),
                    "batch_size": new_sequences if new_sequences else running_count,
                    "running_requests": running_count,
                    "queue_requests": int(queued_match.group("value")),
                    "kv_cache_token_usage_fraction": float(token_usage_match.group("value")),
                    "cuda_graph": "cuda graph: True" in line,
                }
            )
    if not records:
        return {"status": "UNAVAILABLE", "records": []}
    running_counts = [int(item["running_requests"]) for item in records]
    queue = [int(item["queue_requests"]) for item in records]
    batches = [int(item["batch_size"]) for item in records]
    token_usage = [float(item["kv_cache_token_usage_fraction"]) for item in records]
    return {
        "status": "PASS",
        "record_count": len(records),
        "maximum_running_requests_configured": maximum_running_requests,
        "scheduler_occupancy_fraction_mean": statistics.mean(running_counts)
        / max(maximum_running_requests, 1),
        "scheduler_occupancy_fraction_maximum": max(running_counts)
        / max(maximum_running_requests, 1),
        "batch_size_mean": statistics.mean(batches),
        "batch_size_maximum": max(batches),
        "queue_requests_mean": statistics.mean(queue),
        "queue_requests_maximum": max(queue),
        "kv_cache_token_usage_fraction_mean": statistics.mean(token_usage),
        "kv_cache_token_usage_fraction_maximum": max(token_usage),
        "cuda_graph_batch_fraction": sum(bool(item["cuda_graph"]) for item in records)
        / len(records),
        "records": records,
    }


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return ordered[index]


def stream_generate(
    endpoint: str,
    *,
    input_ids: list[int],
    output_tokens: int,
    request_id: str,
    timeout_seconds: float = 600,
    ignore_eos: bool = True,
) -> dict[str, Any]:
    body = json.dumps(
        {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": output_tokens,
                "ignore_eos": ignore_eos,
                "skip_special_tokens": False,
            },
            "stream": True,
            "rid": request_id,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{endpoint.rstrip('/')}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_at: float | None = None
    arrivals: list[float] = []
    previous_count = 0
    latest: dict[str, Any] | None = None
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
            if line == "[DONE]":
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(value, dict):
                continue
            observed = time.perf_counter()
            latest = value
            token_ids = _sglang_output_ids(value)
            added = max(0, len(token_ids) - previous_count)
            if added:
                if first_at is None:
                    first_at = observed
                arrivals.extend([observed] * added)
                previous_count = len(token_ids)
    ended = time.perf_counter()
    if latest is None:
        raise RuntimeError("SGLang streaming response contained no JSON result")
    output_ids = _sglang_output_ids(latest)
    raw_meta = latest.get("meta_info")
    meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
    ttft_ms = ((first_at or ended) - started) * 1000
    inter_token_ms = [(right - left) * 1000 for left, right in pairwise(arrivals)]
    return {
        "request_id": request_id,
        "input_tokens": len(input_ids),
        "output_tokens": len(output_ids),
        "output_token_ids": output_ids,
        "ttft_ms": ttft_ms,
        "end_to_end_ms": (ended - started) * 1000,
        "decode_tokens_per_second": (
            max(0, len(output_ids) - 1) / max(ended - (first_at or ended), 1e-12)
        ),
        "inter_token_latency_ms_p50": statistics.median(inter_token_ms) if inter_token_ms else 0.0,
        "inter_token_latency_ms_p95": percentile(inter_token_ms, 0.95),
        "inter_token_latency_ms_p99": percentile(inter_token_ms, 0.99),
        "cached_prompt_tokens": int(meta.get("cached_tokens") or 0),
        "meta_info": meta,
    }


def run_sglang_point(
    endpoint: str,
    *,
    prompts: list[list[int]],
    output_tokens: int,
    concurrency: int,
    repeats: int,
    workload: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(prompts) < concurrency:
        raise ValueError("SGLang point requires at least one prompt per concurrent request")
    repeat_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    for repeat in range(repeats):
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(
                    stream_generate,
                    endpoint,
                    input_ids=prompts[index],
                    output_tokens=output_tokens,
                    request_id=f"exp007-{workload}-c{concurrency}-r{repeat}-{index}-{uuid4().hex[:8]}",
                )
                for index in range(concurrency)
            ]
            rows = [future.result() for future in futures]
        wall_seconds = time.perf_counter() - started
        tokens = sum(int(row["output_tokens"]) for row in rows)
        input_count = sum(int(row["input_tokens"]) for row in rows)
        latencies = [float(row["end_to_end_ms"]) for row in rows]
        ttft = [float(row["ttft_ms"]) for row in rows]
        repeat_rows.append(
            {
                "classification": "measured_cuda",
                "workload": workload,
                "input_tokens": len(prompts[0]),
                "output_tokens": output_tokens,
                "concurrency": concurrency,
                "repeat": repeat,
                "wall_seconds": wall_seconds,
                "aggregate_verified_throughput": tokens / max(wall_seconds, 1e-12),
                "prefill_tokens_per_second": input_count / max(max(ttft) / 1000, 1e-12),
                "decode_tokens_per_second": sum(
                    float(row["decode_tokens_per_second"]) for row in rows
                ),
                "ttft_p50_ms": statistics.median(ttft),
                "ttft_p95_ms": percentile(ttft, 0.95),
                "ttft_p99_ms": percentile(ttft, 0.99),
                "latency_p50_ms": statistics.median(latencies),
                "latency_p95_ms": percentile(latencies, 0.95),
                "latency_p99_ms": percentile(latencies, 0.99),
                "prefix_cache_reuse": False,
            }
        )
        for row in rows:
            request_rows.append(
                {**row, "workload": workload, "concurrency": concurrency, "repeat": repeat}
            )
    numeric = (
        "aggregate_verified_throughput",
        "prefill_tokens_per_second",
        "decode_tokens_per_second",
        "ttft_p50_ms",
        "ttft_p95_ms",
        "ttft_p99_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
    )
    summary: dict[str, Any] = {
        "classification": "measured_cuda",
        "workload": workload,
        "input_tokens": len(prompts[0]),
        "output_tokens": output_tokens,
        "concurrency": concurrency,
        "repeats": repeats,
    }
    for key in numeric:
        values = [float(row[key]) for row in repeat_rows]
        summary[key] = statistics.median(values)
        summary[f"{key}_minimum"] = min(values)
        summary[f"{key}_maximum"] = max(values)
    return summary, request_rows


def measure_prefix_cache(
    endpoint: str,
    *,
    prompt: list[int],
    output_tokens: int,
) -> dict[str, Any]:
    first = stream_generate(
        endpoint,
        input_ids=prompt,
        output_tokens=output_tokens,
        request_id=f"exp007-prefix-cold-{uuid4().hex[:8]}",
    )
    second = stream_generate(
        endpoint,
        input_ids=prompt,
        output_tokens=output_tokens,
        request_id=f"exp007-prefix-warm-{uuid4().hex[:8]}",
    )
    return {
        "classification": "measured_cuda",
        "input_tokens": len(prompt),
        "output_tokens": output_tokens,
        "cold_ttft_ms": first["ttft_ms"],
        "warm_ttft_ms": second["ttft_ms"],
        "warm_cached_prompt_tokens": second["cached_prompt_tokens"],
        "token_identity": first["output_token_ids"] == second["output_token_ids"],
        "ttft_speedup": float(first["ttft_ms"]) / max(float(second["ttft_ms"]), 1e-12),
    }
