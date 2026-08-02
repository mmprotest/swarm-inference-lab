"""Reusable real-tensor cache, prefetch, and asymmetric expert execution primitives.

These PyTorch primitives are backend-neutral building blocks.  They are exercised
by correctness fixtures; target-model results remain capability-gated because a
llama.cpp server cannot accept Python-owned expert tensors at runtime.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ExpertWeightTriple:
    layer_id: int
    expert_id: int
    up: Any
    gate: Any
    down: Any

    def validate(self) -> None:
        if self.layer_id < 0 or self.expert_id < 0:
            raise ValueError("expert identity must be non-negative")
        if self.up.ndim != 2 or self.gate.ndim != 2 or self.down.ndim != 2:
            raise ValueError("expert up, gate, and down weights must be matrices")
        if tuple(self.up.shape) != tuple(self.gate.shape):
            raise ValueError("up and gate projections must have matching shapes")
        if self.down.shape[0] != self.up.shape[1] or self.down.shape[1] != self.up.shape[0]:
            raise ValueError("down projection must match the expert intermediate range")
        if any(tensor.device.type != "cpu" for tensor in (self.up, self.gate, self.down)):
            raise ValueError("expert source weights must reside in CPU memory")

    @property
    def key(self) -> tuple[int, int]:
        return self.layer_id, self.expert_id

    @property
    def byte_size(self) -> int:
        return sum(
            int(tensor.numel() * tensor.element_size())
            for tensor in (self.up, self.gate, self.down)
        )


@dataclass(slots=True)
class PrefetchTicket:
    key: tuple[int, int]
    byte_size: int
    event: Any
    submitted_monotonic_ns: int
    completed_monotonic_ns: int | None = None


class ExpertTensorCache:
    """Byte-bounded LRU that performs real pinned-host to CUDA tensor copies."""

    def __init__(self, *, capacity_bytes: int, device: str = "cuda") -> None:
        import torch

        if capacity_bytes < 0:
            raise ValueError("cache capacity cannot be negative")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA expert cache requested but CUDA is unavailable")
        self.capacity_bytes = capacity_bytes
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(device=self.device)
        self.entries: OrderedDict[tuple[int, int], tuple[ExpertWeightTriple, Any]] = OrderedDict()
        self.tickets: dict[tuple[int, int], PrefetchTicket] = {}
        self.used_bytes = 0
        self.hits = 0
        self.misses = 0
        self.bytes_prefetched = 0
        self.useful_bytes = 0
        self.wasted_bytes = 0

    @staticmethod
    def _pinned(tensor: Any) -> Any:
        return tensor if tensor.is_pinned() else tensor.pin_memory()

    def _evict_for(self, byte_size: int) -> list[tuple[int, int]]:
        evicted: list[tuple[int, int]] = []
        while self.used_bytes + byte_size > self.capacity_bytes and self.entries:
            key, (weights, _event) = self.entries.popitem(last=False)
            self.used_bytes -= weights.byte_size
            ticket = self.tickets.pop(key, None)
            if ticket is not None:
                self.wasted_bytes += ticket.byte_size
            evicted.append(key)
        return evicted

    def prefetch(self, weights: ExpertWeightTriple) -> PrefetchTicket | None:
        import torch

        weights.validate()
        if weights.key in self.entries:
            self.entries.move_to_end(weights.key)
            return self.tickets.get(weights.key)
        if weights.byte_size > self.capacity_bytes:
            return None
        self._evict_for(weights.byte_size)
        submitted = time.perf_counter_ns()
        with torch.cuda.stream(self.stream):
            gpu = ExpertWeightTriple(
                weights.layer_id,
                weights.expert_id,
                self._pinned(weights.up).to(self.device, non_blocking=True),
                self._pinned(weights.gate).to(self.device, non_blocking=True),
                self._pinned(weights.down).to(self.device, non_blocking=True),
            )
            event = torch.cuda.Event()
            event.record(self.stream)
        ticket = PrefetchTicket(weights.key, weights.byte_size, event, submitted)
        self.entries[weights.key] = (gpu, event)
        self.tickets[weights.key] = ticket
        self.used_bytes += weights.byte_size
        self.bytes_prefetched += weights.byte_size
        return ticket

    def get(self, weights: ExpertWeightTriple) -> tuple[ExpertWeightTriple, bool]:
        import torch

        existing = self.entries.get(weights.key)
        hit = existing is not None
        if existing is None:
            self.misses += 1
            ticket = self.prefetch(weights)
            if ticket is None:
                raise MemoryError(
                    f"expert {weights.key} ({weights.byte_size} bytes) exceeds cache capacity"
                )
            existing = self.entries[weights.key]
        else:
            self.hits += 1
            self.useful_bytes += weights.byte_size
            self.entries.move_to_end(weights.key)
        gpu, event = existing
        torch.cuda.current_stream(self.device).wait_event(event)
        ticket = self.tickets.get(weights.key)
        if ticket is not None and ticket.completed_monotonic_ns is None:
            event.synchronize()
            ticket.completed_monotonic_ns = time.perf_counter_ns()
        return gpu, hit

    def project(self, hidden_state: Any, weights: ExpertWeightTriple) -> tuple[Any, bool]:
        import torch

        gpu, hit = self.get(weights)
        value = hidden_state.to(self.device, non_blocking=True)
        result = torch.nn.functional.silu(value @ gpu.gate) * (value @ gpu.up)
        return result @ gpu.down, hit

    def metrics(self) -> dict[str, float | int]:
        accesses = self.hits + self.misses
        return {
            "capacity_bytes": self.capacity_bytes,
            "used_bytes": self.used_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / accesses if accesses else 0.0,
            "bytes_prefetched": self.bytes_prefetched,
            "useful_bytes": self.useful_bytes,
            "wasted_bytes": self.wasted_bytes,
        }


def asymmetric_expert_projection(
    hidden_state: Any,
    weights: ExpertWeightTriple,
    *,
    gpu_intermediate_fraction: float,
) -> tuple[Any, dict[str, Any]]:
    """Execute matched projection ranges concurrently on CPU and CUDA, then reduce."""

    import torch

    weights.validate()
    if not torch.cuda.is_available():
        raise RuntimeError("asymmetric CPU/GPU fixture requires CUDA")
    if hidden_state.device.type != "cpu":
        raise ValueError("fixture hidden state must begin in CPU memory")
    if not 0 <= gpu_intermediate_fraction <= 1:
        raise ValueError("GPU intermediate fraction must be between zero and one")
    intermediate = int(weights.up.shape[1])
    boundary = max(0, min(intermediate, round(intermediate * gpu_intermediate_fraction)))
    cpu_range = (boundary, intermediate)
    gpu_range = (0, boundary)
    intervals: dict[str, tuple[int, int]] = {}

    def cpu_projection() -> Any:
        started = time.perf_counter_ns()
        if cpu_range[0] == cpu_range[1]:
            result = torch.zeros(
                (hidden_state.shape[0], weights.down.shape[1]), dtype=hidden_state.dtype
            )
        else:
            up = hidden_state @ weights.up[:, cpu_range[0] : cpu_range[1]]
            gate = hidden_state @ weights.gate[:, cpu_range[0] : cpu_range[1]]
            result = (torch.nn.functional.silu(gate) * up) @ weights.down[
                cpu_range[0] : cpu_range[1], :
            ]
        intervals["cpu"] = (started, time.perf_counter_ns())
        return result

    transfer_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()
    gpu_started = time.perf_counter_ns()
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="exp008-expert-cpu") as pool:
        cpu_future = pool.submit(cpu_projection)
        if gpu_range[0] == gpu_range[1]:
            gpu_result_cpu = torch.zeros(
                (hidden_state.shape[0], weights.down.shape[1]), dtype=hidden_state.dtype
            )
        else:
            with torch.cuda.stream(transfer_stream):
                x_gpu = ExpertTensorCache._pinned(hidden_state).to("cuda", non_blocking=True)
                up_gpu = ExpertTensorCache._pinned(
                    weights.up[:, gpu_range[0] : gpu_range[1]].contiguous()
                ).to("cuda", non_blocking=True)
                gate_gpu = ExpertTensorCache._pinned(
                    weights.gate[:, gpu_range[0] : gpu_range[1]].contiguous()
                ).to("cuda", non_blocking=True)
                down_gpu = ExpertTensorCache._pinned(
                    weights.down[gpu_range[0] : gpu_range[1], :].contiguous()
                ).to("cuda", non_blocking=True)
                transfer_ready = torch.cuda.Event()
                transfer_ready.record(transfer_stream)
            with torch.cuda.stream(compute_stream):
                compute_stream.wait_event(transfer_ready)
                gpu_hidden = torch.nn.functional.silu(x_gpu @ gate_gpu) * (x_gpu @ up_gpu)
                gpu_result = gpu_hidden @ down_gpu
                compute_done = torch.cuda.Event()
                compute_done.record(compute_stream)
            compute_done.synchronize()
            gpu_result_cpu = gpu_result.cpu()
        gpu_ended = time.perf_counter_ns()
        cpu_result = cpu_future.result()
    intervals["gpu"] = (gpu_started, gpu_ended)
    cpu_start, cpu_end = intervals["cpu"]
    gpu_start, gpu_end = intervals["gpu"]
    overlap_ns = max(0, min(cpu_end, gpu_end) - max(cpu_start, gpu_start))
    shorter_ns = min(cpu_end - cpu_start, gpu_end - gpu_start)
    return cpu_result + gpu_result_cpu, {
        "classification": "MEASURED",
        "cpu_projection_range": list(cpu_range),
        "gpu_projection_range": list(gpu_range),
        "cpu_interval_monotonic_ns": [cpu_start, cpu_end],
        "gpu_interval_monotonic_ns": [gpu_start, gpu_end],
        "overlap_ns": overlap_ns,
        "overlap_percent": overlap_ns / shorter_ns * 100 if shorter_ns > 0 else 0.0,
        "matching_slice_contract": {
            "up": list(gpu_range),
            "gate": list(gpu_range),
            "down": list(gpu_range),
        },
    }


def validate_tensor_runtime_fixture(*, seed: int = 8008) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {
            "classification": "EMULATED",
            "status": "UNSUPPORTED",
            "reason": "CUDA is unavailable",
        }
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden, intermediate, batch = 64, 96, 8
    x = torch.randn((batch, hidden), generator=generator)
    weights = ExpertWeightTriple(
        0,
        0,
        torch.randn((hidden, intermediate), generator=generator),
        torch.randn((hidden, intermediate), generator=generator),
        torch.randn((intermediate, hidden), generator=generator),
    )
    reference = (torch.nn.functional.silu(x @ weights.gate) * (x @ weights.up)) @ weights.down
    split, overlap = asymmetric_expert_projection(x, weights, gpu_intermediate_fraction=0.7)
    split_difference = (reference - split).abs()
    cache = ExpertTensorCache(capacity_bytes=weights.byte_size * 2)
    first, first_hit = cache.project(x, weights)
    second, second_hit = cache.project(x, weights)
    torch.cuda.synchronize()
    first_cpu = first.cpu()
    second_cpu = second.cpu()
    return {
        "classification": "EMULATED",
        "status": "COMPLETED",
        "scope": "real PyTorch CPU/CUDA tensor movement and compute on a tiny deterministic fixture",
        "cpu_gpu_split_equivalence": {
            "allclose": bool(torch.allclose(reference, split, atol=2e-4, rtol=2e-4)),
            "maximum_absolute_error": float(split_difference.max()),
            "mean_absolute_error": float(split_difference.mean()),
        },
        "cache_miss_equivalence": {
            "allclose": bool(torch.allclose(reference, first_cpu, atol=2e-4, rtol=2e-4)),
            "reported_hit": first_hit,
        },
        "cache_hit_equivalence": {
            "allclose": bool(torch.allclose(reference, second_cpu, atol=2e-4, rtol=2e-4)),
            "reported_hit": second_hit,
        },
        "cache_metrics": cache.metrics(),
        "overlap": overlap,
        "passed": bool(
            torch.allclose(reference, split, atol=2e-4, rtol=2e-4)
            and torch.allclose(reference, first_cpu, atol=2e-4, rtol=2e-4)
            and torch.allclose(reference, second_cpu, atol=2e-4, rtol=2e-4)
            and not first_hit
            and second_hit
        ),
    }
