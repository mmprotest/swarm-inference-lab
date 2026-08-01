"""Measured CPU recomputation audit role."""

from __future__ import annotations

import asyncio
import hashlib
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AuditItem:
    operation_id: str
    primary_payload: bytes
    corrupt: bool = False


AuditFunction = Callable[[AuditItem], Awaitable[bytes]]


def _hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


async def run_integrity_audit_rate(
    items: list[AuditItem],
    *,
    audit_fraction: float,
    audit_function: AuditFunction,
    baseline_gpu_latency_ms: float,
    observed_gpu_latency_ms: float,
    maximum_queue_depth: int = 64,
    seed: int = 7007,
) -> dict[str, Any]:
    if not 0 <= audit_fraction <= 1:
        raise ValueError("audit fraction must be between zero and one")
    rng = random.Random(seed)
    selected = [item for item in items if rng.random() < audit_fraction]
    queue_delay_ms: list[float] = []
    execution_ms: list[float] = []
    detected = 0
    corrupt_selected = 0
    semaphore = asyncio.Semaphore(1)
    queued_at = {item.operation_id: time.perf_counter() for item in selected}

    async def audit(item: AuditItem) -> None:
        nonlocal detected, corrupt_selected
        async with semaphore:
            queue_delay_ms.append((time.perf_counter() - queued_at[item.operation_id]) * 1000)
            started = time.perf_counter()
            reference = await audit_function(item)
            execution_ms.append((time.perf_counter() - started) * 1000)
            mismatch = _hash(reference) != _hash(item.primary_payload)
            if item.corrupt:
                corrupt_selected += 1
            if mismatch:
                detected += 1

    overflow = max(0, len(selected) - maximum_queue_depth)
    admitted = selected[:maximum_queue_depth]
    started = time.perf_counter()
    await asyncio.gather(*(audit(item) for item in admitted))
    wall_ms = (time.perf_counter() - started) * 1000
    p95_impact = observed_gpu_latency_ms / baseline_gpu_latency_ms - 1
    sustainable = overflow == 0 and p95_impact <= 0.05
    return {
        "classification": "measured_mixed_backend",
        "audit_fraction": audit_fraction,
        "operation_count": len(items),
        "selected_operation_count": len(selected),
        "audited_operation_count": len(admitted),
        "audit_queue_overflow": overflow,
        "corrupt_operations": sum(item.corrupt for item in items),
        "corrupt_operations_audited": corrupt_selected,
        "detected_corrupt_operations": detected,
        "detection_coverage": detected / max(sum(item.corrupt for item in items), 1),
        "conditional_detection_rate": detected / max(corrupt_selected, 1),
        "cpu_audit_wall_ms": wall_ms,
        "mean_cpu_audit_execution_ms": sum(execution_ms) / max(len(execution_ms), 1),
        "mean_audit_queue_delay_ms": sum(queue_delay_ms) / max(len(queue_delay_ms), 1),
        "maximum_audit_queue_delay_ms": max(queue_delay_ms, default=0.0),
        "gpu_latency_impact_fraction": p95_impact,
        "sustainable": sustainable,
    }


def maximum_sustainable_audit_fraction(rows: list[dict[str, Any]]) -> float:
    return max(
        (float(row["audit_fraction"]) for row in rows if bool(row["sustainable"])),
        default=0.0,
    )
