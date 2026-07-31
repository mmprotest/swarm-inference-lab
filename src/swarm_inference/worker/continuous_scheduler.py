"""Iteration-level continuous scheduler for the Qwen3 performance engine."""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class SchedulerPolicy(StrEnum):
    LATENCY = "latency"
    BALANCED = "balanced"
    THROUGHPUT = "throughput"


@dataclass(slots=True)
class ScheduledRequest:
    request_id: str
    prompt_tokens: int
    output_tokens: int
    admitted_ns: int
    admitted_iteration: int = 0
    prompt_consumed: int = 0
    output_produced: int = 0
    last_served_iteration: int = -1

    @property
    def prefill_complete(self) -> bool:
        return self.prompt_consumed >= self.prompt_tokens

    @property
    def complete(self) -> bool:
        return self.output_produced >= self.output_tokens

    @property
    def sequence_length(self) -> int:
        return self.prompt_consumed + self.output_produced


@dataclass(frozen=True, slots=True)
class SchedulingIteration:
    index: int
    policy: str
    decode_request_ids: tuple[str, ...]
    prefill_request_ids: tuple[str, ...]
    prefill_chunk_size: int
    active_request_count: int
    waiting_request_count: int
    kv_tokens_reserved: int
    decode_batch_occupancy: float
    scheduler_overhead_ms: float
    starved_request_ids: tuple[str, ...]


@dataclass(slots=True)
class SchedulerMetrics:
    iterations: int = 0
    scheduler_overhead_ms: float = 0.0
    batch_size_distribution: Counter[int] = field(default_factory=Counter)
    prefill_chunk_distribution: Counter[int] = field(default_factory=Counter)
    starvation_events: int = 0
    maximum_wait_ms: float = 0.0
    useful_tokens: int = 0
    padding_tokens: int = 0

    def payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["batch_size_distribution"] = {
            str(key): value for key, value in sorted(self.batch_size_distribution.items())
        }
        payload["prefill_chunk_distribution"] = {
            str(key): value for key, value in sorted(self.prefill_chunk_distribution.items())
        }
        return payload


class ContinuousBatchScheduler:
    """Schedule homogeneous decode batches plus bounded prefill work."""

    def __init__(
        self,
        *,
        policy: SchedulerPolicy | str,
        max_batch_size: int,
        prefill_chunk_size: int,
        kv_token_capacity: int,
        starvation_limit_iterations: int = 32,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be positive")
        if kv_token_capacity <= 0:
            raise ValueError("kv_token_capacity must be positive")
        if starvation_limit_iterations <= 0:
            raise ValueError("starvation_limit_iterations must be positive")
        self.policy = SchedulerPolicy(policy)
        self.max_batch_size = max_batch_size
        self.prefill_chunk_size = prefill_chunk_size
        self.kv_token_capacity = kv_token_capacity
        self.starvation_limit_iterations = starvation_limit_iterations
        self._waiting: deque[ScheduledRequest] = deque()
        self._active: dict[str, ScheduledRequest] = {}
        self._iteration = 0
        self.metrics = SchedulerMetrics()

    @property
    def active_request_count(self) -> int:
        return len(self._active)

    @property
    def waiting_request_count(self) -> int:
        return len(self._waiting)

    @property
    def kv_tokens_reserved(self) -> int:
        return sum(item.sequence_length for item in self._active.values())

    def admit(
        self,
        request_id: str,
        *,
        prompt_tokens: int,
        output_tokens: int,
        admitted_ns: int | None = None,
    ) -> None:
        if not request_id:
            raise ValueError("request_id cannot be empty")
        if request_id in self._active or any(
            item.request_id == request_id for item in self._waiting
        ):
            raise ValueError(f"duplicate scheduler request {request_id!r}")
        if prompt_tokens <= 0 or output_tokens <= 0:
            raise ValueError("prompt_tokens and output_tokens must be positive")
        if prompt_tokens > self.kv_token_capacity:
            raise ValueError(
                f"request prompt requires {prompt_tokens} KV tokens, capacity is "
                f"{self.kv_token_capacity}"
            )
        self._waiting.append(
            ScheduledRequest(
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                admitted_ns=admitted_ns or time.monotonic_ns(),
                admitted_iteration=self._iteration,
            )
        )

    def next_iteration(self, *, now_ns: int | None = None) -> SchedulingIteration:
        started = time.perf_counter_ns()
        current_ns = now_ns or time.monotonic_ns()
        self._iteration += 1
        decodable = [
            item for item in self._active.values() if item.prefill_complete and not item.complete
        ]
        decodable.sort(
            key=lambda item: (
                item.last_served_iteration,
                item.admitted_ns,
                item.request_id,
            )
        )
        starved_active = tuple(
            item.request_id
            for item in decodable
            if item.last_served_iteration >= 0
            and self._iteration - item.last_served_iteration > self.starvation_limit_iterations
        )
        starved_waiting = tuple(
            item.request_id
            for item in self._waiting
            if self._iteration - item.admitted_iteration > self.starvation_limit_iterations
        )
        decode_limit = {
            SchedulerPolicy.LATENCY: 1,
            SchedulerPolicy.BALANCED: max(1, self.max_batch_size // 2),
            SchedulerPolicy.THROUGHPUT: self.max_batch_size,
        }[self.policy]
        decode = decodable[:decode_limit]
        remaining_slots = self.max_batch_size - len(decode)
        if starved_waiting and remaining_slots == 0 and decode:
            decode.pop()
            remaining_slots = 1
        starved = (*starved_active, *starved_waiting)
        prefill: list[ScheduledRequest] = []
        chunk_size = 0
        permit_prefill = not decode or self.policy != SchedulerPolicy.LATENCY or bool(starved)
        if permit_prefill and remaining_slots > 0 and self._waiting:
            first_length = self._waiting[0].prompt_tokens
            while self._waiting and len(prefill) < remaining_slots:
                candidate = self._waiting[0]
                if candidate.prompt_tokens != first_length:
                    break
                projected = self.kv_tokens_reserved + sum(
                    min(self.prefill_chunk_size, item.prompt_tokens) for item in prefill
                )
                candidate_chunk = min(self.prefill_chunk_size, candidate.prompt_tokens)
                if projected + candidate_chunk > self.kv_token_capacity:
                    break
                prefill.append(self._waiting.popleft())
            if prefill:
                chunk_size = min(
                    self.prefill_chunk_size,
                    min(item.prompt_tokens - item.prompt_consumed for item in prefill),
                )
                for item in prefill:
                    self._active[item.request_id] = item
        overhead_ms = (time.perf_counter_ns() - started) / 1_000_000
        waits = [
            max(0.0, (current_ns - item.admitted_ns) / 1_000_000)
            for item in (*self._waiting, *prefill)
        ]
        self.metrics.iterations += 1
        self.metrics.scheduler_overhead_ms += overhead_ms
        self.metrics.batch_size_distribution[len(decode)] += 1
        if chunk_size:
            self.metrics.prefill_chunk_distribution[chunk_size] += 1
        self.metrics.starvation_events += len(starved)
        self.metrics.maximum_wait_ms = max(
            self.metrics.maximum_wait_ms,
            max(waits, default=0.0),
        )
        return SchedulingIteration(
            index=self._iteration,
            policy=self.policy.value,
            decode_request_ids=tuple(item.request_id for item in decode),
            prefill_request_ids=tuple(item.request_id for item in prefill),
            prefill_chunk_size=chunk_size,
            active_request_count=len(self._active),
            waiting_request_count=len(self._waiting),
            kv_tokens_reserved=self.kv_tokens_reserved,
            decode_batch_occupancy=len(decode) / self.max_batch_size,
            scheduler_overhead_ms=overhead_ms,
            starved_request_ids=starved,
        )

    def mark_prefill(
        self,
        request_ids: tuple[str, ...],
        *,
        useful_tokens: int,
        padding_tokens: int = 0,
    ) -> None:
        if padding_tokens < 0:
            raise ValueError("padding_tokens cannot be negative")
        if useful_tokens < 0:
            raise ValueError("useful_tokens cannot be negative")
        per_request = useful_tokens // len(request_ids) if request_ids else 0
        for request_id in request_ids:
            request = self._active[request_id]
            request.prompt_consumed = min(
                request.prompt_tokens,
                request.prompt_consumed + per_request,
            )
            request.last_served_iteration = self._iteration
        self.metrics.useful_tokens += useful_tokens
        self.metrics.padding_tokens += padding_tokens

    def mark_decode(self, request_ids: tuple[str, ...]) -> tuple[str, ...]:
        completed: list[str] = []
        for request_id in request_ids:
            request = self._active[request_id]
            if not request.prefill_complete:
                raise ValueError(f"cannot decode request {request_id!r} before prefill")
            request.output_produced += 1
            request.last_served_iteration = self._iteration
            self.metrics.useful_tokens += 1
            if request.complete:
                completed.append(request_id)
        for request_id in completed:
            self._active.pop(request_id)
        return tuple(completed)

    def state(self) -> dict[str, Any]:
        return {
            "policy": self.policy.value,
            "iteration": self._iteration,
            "active_requests": {key: asdict(value) for key, value in sorted(self._active.items())},
            "waiting_requests": [asdict(value) for value in self._waiting],
            "kv_token_capacity": self.kv_token_capacity,
            "kv_tokens_reserved": self.kv_tokens_reserved,
            "metrics": self.metrics.payload(),
        }
