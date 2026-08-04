"""Real streaming workloads and metric aggregation for Experiment 008."""

from __future__ import annotations

import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from swarm_inference.experiments.experiment_008.backend import (
    GenerationResult,
    LlamaCppClient,
)
from swarm_inference.experiments.experiment_008.hardware import ResourceSampler, percentile
from swarm_inference.experiments.experiment_008.schemas import (
    BenchmarkObservation,
    EvidenceClass,
    ExecutionStatus,
)
from swarm_inference.experiments.experiment_008.workloads import WorkloadPrompt


@dataclass(slots=True)
class WorkloadExecution:
    observation: BenchmarkObservation
    generations: list[GenerationResult]
    resource_rows: list[dict[str, Any]]


def _numeric(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def _resource_metrics(rows: list[dict[str, Any]], *, output_tokens: int) -> dict[str, Any]:
    gpu_memory_mib = _numeric(rows, "gpu_memory_used_mib")
    ram = _numeric(rows, "system_ram_used_bytes")
    rss = _numeric(rows, "process_tree_rss_bytes")
    gpu_util = _numeric(rows, "gpu_compute_utilisation_percent")
    gpu_memory_util = _numeric(rows, "gpu_memory_controller_utilisation_percent")
    cpu_util = _numeric(rows, "process_tree_cpu_percent")
    temperatures = _numeric(rows, "gpu_temperature_c")
    power = _numeric(rows, "gpu_power_watts")
    dmon = [row for row in rows if row.get("sampler_source") == "nvidia-smi-dmon"]
    # dmon emits rates at a one-second period.  Integrating each reported rate over
    # that period yields sampled transfer volume; this is recorded as an estimate,
    # not an exact CUDA API byte counter.
    pcie_h2d = sum(_numeric(dmon, "pcie_gpu_receive_mib_s")) * 1024**2
    pcie_d2h = sum(_numeric(dmon, "pcie_gpu_transmit_mib_s")) * 1024**2
    pcie_total = pcie_h2d + pcie_d2h if dmon else None
    return {
        "peak_vram_bytes": max(gpu_memory_mib) * 1024**2 if gpu_memory_mib else None,
        "peak_system_ram_bytes": max(ram) if ram else None,
        "peak_process_tree_rss_bytes": max(rss) if rss else None,
        "mean_gpu_compute_utilisation_percent": statistics.fmean(gpu_util) if gpu_util else None,
        "mean_gpu_memory_controller_utilisation_percent": (
            statistics.fmean(gpu_memory_util) if gpu_memory_util else None
        ),
        "mean_process_tree_cpu_percent": statistics.fmean(cpu_util) if cpu_util else None,
        "peak_gpu_temperature_c": max(temperatures) if temperatures else None,
        "peak_gpu_power_watts": max(power) if power else None,
        "pcie_host_to_device_bytes_sampled": pcie_h2d if dmon else None,
        "pcie_device_to_host_bytes_sampled": pcie_d2h if dmon else None,
        "pcie_bytes_sampled": pcie_total,
        "pcie_bytes_per_output_token": (
            pcie_total / output_tokens if pcie_total is not None and output_tokens > 0 else None
        ),
        "pcie_measurement_method": (
            "integral of one-second nvidia-smi dmon PCIe Rx/Tx samples"
            if dmon
            else "unavailable: no nvidia-smi dmon sample completed during this workload"
        ),
    }


def _generation_metrics(generations: list[GenerationResult]) -> dict[str, Any]:
    successful = [item for item in generations if item.success]
    ttft = [item.time_to_first_token_ms for item in successful]
    ttft_values = [float(value) for value in ttft if value is not None]
    inter_token = [value for item in successful for value in item.inter_token_latencies_ms]
    decode_rates = [
        float(value) for item in successful if (value := item.decode_tokens_per_second) is not None
    ]
    prompt_rates: list[float] = []
    for item in successful:
        value = item.timings.get("prompt_per_second")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            prompt_rates.append(float(value))
    return {
        "request_count": len(generations),
        "successful_request_count": len(successful),
        "failed_request_count": len(generations) - len(successful),
        "prompt_token_count_total": sum(len(item.prompt_token_ids) for item in successful),
        "prompt_token_count_min": min(
            (len(item.prompt_token_ids) for item in successful), default=None
        ),
        "prompt_token_count_max": max(
            (len(item.prompt_token_ids) for item in successful), default=None
        ),
        "output_token_count": sum(len(item.output_token_ids) for item in successful),
        "decode_tokens_per_second": statistics.median(decode_rates) if decode_rates else None,
        "decode_tokens_per_second_p95": percentile(decode_rates, 95) if decode_rates else None,
        "median_inter_token_latency_ms": statistics.median(inter_token) if inter_token else None,
        "p95_inter_token_latency_ms": percentile(inter_token, 95) if inter_token else None,
        "time_to_first_token_ms": statistics.median(ttft_values) if ttft_values else None,
        "time_to_first_token_p95_ms": percentile(ttft_values, 95) if ttft_values else None,
        "prefill_tokens_per_second": statistics.median(prompt_rates) if prompt_rates else None,
        "errors": [item.error for item in generations if item.error],
    }


def execute_prompt_batch(
    *,
    configuration: str,
    workload: str,
    plan_id: str,
    client: LlamaCppClient,
    prompts: list[WorkloadPrompt],
    seed: int,
    sample_interval_seconds: float,
) -> WorkloadExecution:
    sampler = ResourceSampler(
        interval_seconds=sample_interval_seconds,
        label=f"{configuration}:{workload}:{plan_id}",
    )
    results: list[GenerationResult] = []
    sampler.start()
    try:
        for index, prompt in enumerate(prompts):
            results.append(
                client.generate(
                    prompt.token_ids,
                    output_tokens=prompt.requested_output_tokens,
                    seed=seed + index,
                )
            )
    finally:
        rows = sampler.stop()
    metrics = _generation_metrics(results)
    metrics.update(_resource_metrics(rows, output_tokens=int(metrics["output_token_count"] or 0)))
    status = (
        ExecutionStatus.COMPLETED
        if results and all(item.success for item in results)
        else ExecutionStatus.FAILED
    )
    observation = BenchmarkObservation(
        configuration=configuration,  # type: ignore[arg-type]
        workload=workload,  # type: ignore[arg-type]
        plan_id=plan_id,
        status=status,
        evidence_class=EvidenceClass.MEASURED if status == ExecutionStatus.COMPLETED else None,
        metrics=metrics,
        unavailable_reason=(
            None
            if status == ExecutionStatus.COMPLETED
            else "one or more real llama.cpp generation requests failed; see per-request logs"
        ),
        exit_code=None,
    )
    return WorkloadExecution(observation, results, rows)


def execute_mixed_service(
    *,
    configuration: str,
    plan_id: str,
    client: LlamaCppClient,
    interactive: WorkloadPrompt,
    background: WorkloadPrompt,
    seed: int,
    sample_interval_seconds: float,
    minimum_measurement_seconds: float | None = None,
) -> WorkloadExecution:
    sampler = ResourceSampler(
        interval_seconds=sample_interval_seconds,
        label=f"{configuration}:mixed:{plan_id}",
    )
    admitted = time.perf_counter_ns()
    sampler.start()
    try:
        interactive_results: list[GenerationResult] = []
        background_results: list[GenerationResult] = []
        cycle = 0
        target_seconds = max(float(minimum_measurement_seconds or 0.0), 0.0)
        while cycle == 0 or (time.perf_counter_ns() - admitted) / 1_000_000_000 < target_seconds:
            cycle_admitted = time.perf_counter_ns()
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="exp008-mixed") as pool:
                interactive_future = pool.submit(
                    client.generate,
                    interactive.token_ids,
                    output_tokens=interactive.requested_output_tokens,
                    seed=seed + cycle * 2,
                    admitted_monotonic_ns=cycle_admitted,
                )
                background_future = pool.submit(
                    client.generate,
                    background.token_ids,
                    output_tokens=background.requested_output_tokens,
                    seed=seed + cycle * 2 + 1,
                    admitted_monotonic_ns=cycle_admitted,
                )
                interactive_results.append(interactive_future.result())
                background_results.append(background_future.result())
            cycle += 1
            if not interactive_results[-1].success or not background_results[-1].success:
                break
    finally:
        rows = sampler.stop()
    results = [
        item for pair in zip(interactive_results, background_results, strict=True) for item in pair
    ]
    started = min(item.started_monotonic_ns for item in results)
    completed = max(item.completed_monotonic_ns for item in results)
    window_seconds = (completed - started) / 1_000_000_000
    generated_tokens = sum(len(item.output_token_ids) for item in results if item.success)
    interactive_intervals = [
        value for item in interactive_results for value in item.inter_token_latencies_ms
    ]
    background_intervals = [
        value for item in background_results for value in item.inter_token_latencies_ms
    ]
    interactive_rates = [
        float(value)
        for item in interactive_results
        if (value := item.decode_tokens_per_second) is not None
    ]
    background_rates = [
        float(value)
        for item in background_results
        if (value := item.decode_tokens_per_second) is not None
    ]
    metrics: dict[str, Any] = {
        "request_count": len(results),
        "successful_request_count": sum(item.success for item in results),
        "prompt_token_count_total": sum(len(item.prompt_token_ids) for item in results),
        "prompt_token_count_min": min(
            (len(item.prompt_token_ids) for item in results), default=None
        ),
        "prompt_token_count_max": max(
            (len(item.prompt_token_ids) for item in results), default=None
        ),
        "output_token_count": generated_tokens,
        "measurement_window_seconds": window_seconds,
        "measurement_target_seconds": target_seconds,
        "measurement_cycle_count": cycle,
        "interactive_p50_latency_ms": (
            statistics.median(interactive_intervals) if interactive_intervals else None
        ),
        "interactive_p95_latency_ms": (
            percentile(interactive_intervals, 95) if interactive_intervals else None
        ),
        "interactive_tokens_per_second": (
            statistics.median(interactive_rates) if interactive_rates else None
        ),
        "background_tokens_per_second": (
            statistics.median(background_rates) if background_rates else None
        ),
        "combined_generated_tokens_per_second": (
            generated_tokens / window_seconds if window_seconds > 0 else None
        ),
        "mixed_verified_tokens_per_second": None,
        "verification_status": "PENDING_DETERMINISTIC_COMPARISON",
        "interactive_client_dispatch_delay_ms": (
            statistics.median(
                item.started_monotonic_ns - item.admitted_monotonic_ns
                for item in interactive_results
            )
        )
        / 1_000_000,
        "background_client_dispatch_delay_ms": (
            statistics.median(
                item.started_monotonic_ns - item.admitted_monotonic_ns
                for item in background_results
            )
        )
        / 1_000_000,
        "interactive_scheduling_delay_ms": None,
        "background_scheduling_delay_ms": None,
        "scheduling_delay_measurement": (
            "UNSUPPORTED: the llama.cpp HTTP stream does not expose server queue-entry and "
            "execution-start timestamps; client dispatch delay is recorded separately"
        ),
        "interactive_starvation_time_ms": max(interactive_intervals)
        if interactive_intervals
        else None,
        "background_starvation_time_ms": max(background_intervals)
        if background_intervals
        else None,
        "errors": [item.error for item in results if item.error],
    }
    metrics.update(_resource_metrics(rows, output_tokens=generated_tokens))
    status = (
        ExecutionStatus.COMPLETED
        if all(item.success for item in results)
        else ExecutionStatus.FAILED
    )
    observation = BenchmarkObservation(
        configuration=configuration,  # type: ignore[arg-type]
        workload="mixed",
        plan_id=plan_id,
        status=status,
        evidence_class=EvidenceClass.MEASURED if status == ExecutionStatus.COMPLETED else None,
        metrics=metrics,
        unavailable_reason=(
            None if status == ExecutionStatus.COMPLETED else "a concurrent generation stream failed"
        ),
        exit_code=None,
    )
    return WorkloadExecution(observation, results, rows)
