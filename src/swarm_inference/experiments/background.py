"""Independent CPU background capacity under interactive GPU load."""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from swarm_inference.worker.abi import (
    BackendAdapter,
    GenerationParameters,
    TokenPayload,
    WorkerJob,
    WorkerJobStatus,
    WorkerJobType,
)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


@dataclass(order=True, slots=True)
class PrioritisedWork:
    priority: int
    sequence: int
    job: WorkerJob = field(compare=False)


class BackgroundAdmissionController:
    def __init__(
        self,
        *,
        maximum_memory_pressure_fraction: float = 0.90,
        maximum_interactive_p95_increase_fraction: float = 0.05,
    ) -> None:
        self.maximum_memory_pressure_fraction = maximum_memory_pressure_fraction
        self.maximum_interactive_p95_increase_fraction = maximum_interactive_p95_increase_fraction
        self.suspended = False
        self.suspension_reason: str | None = None

    def observe_pressure(self, memory_pressure_fraction: float) -> bool:
        if memory_pressure_fraction > self.maximum_memory_pressure_fraction:
            self.suspended = True
            self.suspension_reason = "host memory pressure limit exceeded"
        return not self.suspended

    def observe_latency(self, baseline_p95_ms: float, observed_p95_ms: float) -> bool:
        if baseline_p95_ms <= 0:
            raise ValueError("baseline p95 latency must be positive")
        increase = observed_p95_ms / baseline_p95_ms - 1
        if increase > self.maximum_interactive_p95_increase_fraction:
            self.suspended = True
            self.suspension_reason = "interactive p95 non-degradation limit exceeded"
        return not self.suspended

    def observe_throughput(
        self,
        baseline_tokens_per_second: float,
        observed_tokens_per_second: float,
        *,
        maximum_decrease_fraction: float,
    ) -> bool:
        if baseline_tokens_per_second <= 0:
            raise ValueError("baseline throughput must be positive")
        decrease = 1 - observed_tokens_per_second / baseline_tokens_per_second
        if decrease > maximum_decrease_fraction:
            self.suspended = True
            self.suspension_reason = "interactive throughput non-degradation limit exceeded"
        return not self.suspended

    def resume(self) -> None:
        self.suspended = False
        self.suspension_reason = None


async def _one_generation(
    adapter: BackendAdapter,
    *,
    role: WorkerJobType,
    model_id: str,
    model_revision: str,
    prompt_token_ids: list[int],
    tokenizer_hash: str,
    output_tokens: int,
    priority: int,
    sequence: int,
) -> dict[str, Any]:
    job = WorkerJob(
        job_id=uuid4().hex,
        request_id=f"background-matrix-{role.value}-{sequence}-{uuid4().hex[:8]}",
        role=role,
        model_id=model_id,
        model_revision=model_revision,
        input_payload=TokenPayload(
            token_ids=prompt_token_ids,
            tokenizer_hash=tokenizer_hash,
        ),
        generation_parameters=GenerationParameters(
            max_new_tokens=output_tokens,
            temperature=0.0,
            ignore_eos=True,
        ),
        deadline_ms=600_000,
        priority=priority,
    )
    started = time.perf_counter()
    result = await adapter.execute(job)
    elapsed = time.perf_counter() - started
    if result.status != WorkerJobStatus.ACCEPTED:
        raise RuntimeError(f"{adapter.backend_id}: {result.status.value}: {result.detail}")
    if not isinstance(result.output_payload, TokenPayload):
        raise RuntimeError(f"{adapter.backend_id} returned a non-token result")
    return {
        "elapsed_seconds": elapsed,
        "output_tokens": len(result.output_payload.token_ids),
        "metrics": result.metrics,
    }


async def _run_lane(
    adapter: BackendAdapter,
    *,
    role: WorkerJobType,
    model_id: str,
    model_revision: str,
    prompts: list[list[int]],
    tokenizer_hash: str,
    output_tokens: int,
    concurrency: int,
    priority: int,
) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(concurrency)

    async def run(index: int, prompt: list[int]) -> dict[str, Any]:
        async with semaphore:
            return await _one_generation(
                adapter,
                role=role,
                model_id=model_id,
                model_revision=model_revision,
                prompt_token_ids=prompt,
                tokenizer_hash=tokenizer_hash,
                output_tokens=output_tokens,
                priority=priority,
                sequence=index,
            )

    started = time.perf_counter()
    results = await asyncio.gather(*(run(index, prompt) for index, prompt in enumerate(prompts)))
    wall_seconds = time.perf_counter() - started
    latencies_ms = [float(item["elapsed_seconds"]) * 1000 for item in results]
    tokens = sum(int(item["output_tokens"]) for item in results)
    return {
        "request_count": len(results),
        "wall_seconds": wall_seconds,
        "output_tokens": tokens,
        "aggregate_tokens_per_second": tokens / max(wall_seconds, 1e-12),
        "latency_p50_ms": statistics.median(latencies_ms),
        "latency_p95_ms": _percentile(latencies_ms, 0.95),
        "latency_p99_ms": _percentile(latencies_ms, 0.99),
        "requests": results,
    }


async def run_background_capacity_point(
    *,
    gpu_adapter: BackendAdapter,
    cpu_adapter: BackendAdapter,
    gpu_model_id: str,
    gpu_revision: str,
    cpu_model_id: str,
    cpu_revision: str,
    tokenizer_hash: str,
    gpu_prompts: list[list[int]],
    cpu_prompts: list[list[int]],
    gpu_output_tokens: int,
    cpu_output_tokens: int,
    gpu_concurrency: int,
    cpu_concurrency: int,
    baseline_gpu: dict[str, Any],
    maximum_p95_increase_fraction: float = 0.05,
    maximum_throughput_decrease_fraction: float = 0.05,
) -> dict[str, Any]:
    controller = BackgroundAdmissionController(
        maximum_interactive_p95_increase_fraction=maximum_p95_increase_fraction
    )
    paired_started = time.perf_counter()
    gpu_task = asyncio.create_task(
        _run_lane(
            gpu_adapter,
            role=WorkerJobType.TARGET_DECODE,
            model_id=gpu_model_id,
            model_revision=gpu_revision,
            prompts=gpu_prompts,
            tokenizer_hash=tokenizer_hash,
            output_tokens=gpu_output_tokens,
            concurrency=gpu_concurrency,
            priority=100,
        )
    )
    cpu_task = asyncio.create_task(
        _run_lane(
            cpu_adapter,
            role=WorkerJobType.BACKGROUND_GENERATE,
            model_id=cpu_model_id,
            model_revision=cpu_revision,
            prompts=cpu_prompts,
            tokenizer_hash=tokenizer_hash,
            output_tokens=cpu_output_tokens,
            concurrency=cpu_concurrency,
            priority=1,
        )
    )
    gpu, cpu = await asyncio.gather(gpu_task, cpu_task)
    paired_seconds = time.perf_counter() - paired_started
    baseline_p95 = float(baseline_gpu["latency_p95_ms"])
    p95_increase = float(gpu["latency_p95_ms"]) / baseline_p95 - 1
    controller.observe_latency(baseline_p95, float(gpu["latency_p95_ms"]))
    combined_tokens = int(gpu["output_tokens"]) + int(cpu["output_tokens"])
    combined_tps = combined_tokens / max(paired_seconds, 1e-12)
    baseline_tps = float(baseline_gpu["aggregate_tokens_per_second"])
    gpu_throughput_change = float(gpu["aggregate_tokens_per_second"]) / max(baseline_tps, 1e-12) - 1
    controller.observe_throughput(
        baseline_tps,
        float(gpu["aggregate_tokens_per_second"]),
        maximum_decrease_fraction=maximum_throughput_decrease_fraction,
    )
    combined_gain = combined_tps / baseline_tps - 1
    scheduler_overhead_ms = (
        max(
            0.0,
            paired_seconds - max(float(gpu["wall_seconds"]), float(cpu["wall_seconds"])),
        )
        * 1000
    )
    non_degradation_pass = (
        p95_increase <= maximum_p95_increase_fraction
        and gpu_throughput_change >= -maximum_throughput_decrease_fraction
    )
    useful = combined_gain >= 0.10 and non_degradation_pass
    return {
        "classification": "measured_mixed_backend",
        "gpu_concurrency": gpu_concurrency,
        "cpu_concurrency": cpu_concurrency,
        "gpu_interactive_p50_ms": gpu["latency_p50_ms"],
        "gpu_interactive_p95_ms": gpu["latency_p95_ms"],
        "gpu_interactive_p99_ms": gpu["latency_p99_ms"],
        "baseline_gpu_p95_ms": baseline_p95,
        "interactive_p95_increase_fraction": p95_increase,
        "gpu_aggregate_tokens_per_second": gpu["aggregate_tokens_per_second"],
        "baseline_gpu_tokens_per_second": baseline_tps,
        "cpu_background_tokens_per_second": cpu["aggregate_tokens_per_second"],
        "total_combined_verified_tokens_per_second": combined_tps,
        "combined_throughput_gain_fraction": combined_gain,
        "paired_wall_seconds": paired_seconds,
        "scheduler_overhead_ms": scheduler_overhead_ms,
        "gpu_throughput_interference_fraction": gpu_throughput_change,
        # The background model is CPU-resident and exchanges token IDs over loopback;
        # no model activation or weight payload crosses PCIe in this role.
        "pcie_payload_bytes": 0,
        "pcie_interference_status": "no_cross_device_model_payload",
        "background_priority": 1,
        "interactive_priority": 100,
        "background_suspended": controller.suspended,
        "suspension_reason": controller.suspension_reason,
        "non_degradation_pass": non_degradation_pass,
        "positive_contribution_pass": useful,
        "gpu_request_metrics": gpu["requests"],
        "cpu_request_metrics": cpu["requests"],
    }


async def measure_gpu_baseline_lane(
    *,
    adapter: BackendAdapter,
    model_id: str,
    revision: str,
    tokenizer_hash: str,
    prompts: list[list[int]],
    output_tokens: int,
    concurrency: int,
) -> dict[str, Any]:
    return await _run_lane(
        adapter,
        role=WorkerJobType.TARGET_DECODE,
        model_id=model_id,
        model_revision=revision,
        prompts=prompts,
        tokenizer_hash=tokenizer_hash,
        output_tokens=output_tokens,
        concurrency=concurrency,
        priority=100,
    )
