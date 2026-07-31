"""Mathematically correct logical collectives and future network interface."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

import torch

from swarm_inference.microsharding.schemas import CollectivePlan

TensorMap = dict[str, torch.Tensor]
T = TypeVar("T")


@runtime_checkable
class CollectiveBackend(Protocol):
    async def broadcast(
        self, plan: CollectivePlan, source_rank: str, value: torch.Tensor
    ) -> TensorMap: ...

    async def all_reduce_sum(self, plan: CollectivePlan, values: TensorMap) -> TensorMap: ...

    async def all_gather(
        self, plan: CollectivePlan, values: TensorMap, *, dim: int = 0
    ) -> TensorMap: ...

    async def reduce_scatter_sum(
        self, plan: CollectivePlan, values: TensorMap, *, dim: int = 0
    ) -> TensorMap: ...

    async def all_to_all(
        self,
        plan: CollectivePlan,
        values: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, dict[str, torch.Tensor]]: ...

    async def gather_to_leader(
        self,
        plan: CollectivePlan,
        values: TensorMap,
        *,
        leader_rank: str,
        dim: int = 0,
    ) -> dict[str, torch.Tensor | None]: ...

    async def distributed_argmax(
        self,
        plan: CollectivePlan,
        candidates: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]: ...

    async def barrier(self, plan: CollectivePlan) -> None: ...


@runtime_checkable
class NetworkCollectiveBackend(CollectiveBackend, Protocol):
    """Interface only for later gRPC, QUIC, NCCL, MPI, or MLX transports."""

    @property
    def transport_name(self) -> str: ...

    async def connect(self, rank_id: str, endpoint: str) -> None: ...

    async def close(self) -> None: ...


@dataclass(slots=True)
class CollectiveMeasurement:
    collective_id: str
    operation: str
    rank_count: int
    payload_bytes: int
    logical_aggregate_bytes: int
    actual_same_device_time_ms: float
    deterministic_rank_order: bool

    def payload(self) -> dict[str, Any]:
        return {
            "classification": "logical_single_gpu_measurement",
            "collective_id": self.collective_id,
            "event_type": "collective_complete",
            "operation": self.operation,
            "rank_count": self.rank_count,
            "payload_bytes": self.payload_bytes,
            "logical_aggregate_bytes": self.logical_aggregate_bytes,
            "actual_same_device_time_ms": self.actual_same_device_time_ms,
            "deterministic_rank_order": self.deterministic_rank_order,
        }


class SingleDeviceLogicalBackend:
    """Execute rank-local tensors in one process and one CUDA context."""

    def __init__(
        self,
        *,
        deterministic_rank_order: bool = True,
        measure_cuda_time: bool = True,
        simulated_delay_ms: float = 0.0,
    ) -> None:
        self.deterministic_rank_order = deterministic_rank_order
        self.measure_cuda_time = measure_cuda_time
        self.simulated_delay_ms = simulated_delay_ms
        self.trace: list[dict[str, Any]] = []

    def _ordered(self, plan: CollectivePlan, values: dict[str, T]) -> list[tuple[str, T]]:
        expected = set(plan.rank_ids)
        actual = set(values)
        if expected != actual:
            raise ValueError(
                f"collective group membership mismatch: missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
        return [(rank, values[rank]) for rank in plan.rank_ids]

    async def _before(self, plan: CollectivePlan) -> None:
        if self.simulated_delay_ms > plan.timeout_ms:
            raise TimeoutError(
                f"collective {plan.collective_id} exceeded {plan.timeout_ms} ms timeout"
            )
        if self.simulated_delay_ms > 0:
            await asyncio.sleep(self.simulated_delay_ms / 1000)

    @staticmethod
    def _logical_bytes(operation: str, rank_count: int, payload_bytes: int) -> int:
        if rank_count <= 1 or operation == "barrier":
            return 0
        if operation in {"all_reduce_sum", "reduce_scatter_sum"}:
            return 2 * (rank_count - 1) * payload_bytes
        if operation == "all_gather":
            return (rank_count - 1) * payload_bytes
        if operation in {"broadcast", "gather_to_leader"}:
            return (rank_count - 1) * payload_bytes
        if operation == "all_to_all":
            return rank_count * (rank_count - 1) * payload_bytes
        if operation == "distributed_argmax":
            return 2 * (rank_count - 1) * payload_bytes
        raise ValueError(f"unknown collective operation {operation}")

    def _measure_start(self, sample: torch.Tensor) -> tuple[int, torch.cuda.Event | None]:
        event = None
        if self.measure_cuda_time and sample.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            event.record()
        return time.perf_counter_ns(), event

    def _record(
        self,
        *,
        plan: CollectivePlan,
        sample: torch.Tensor,
        payload_bytes: int,
        started_ns: int,
        start_event: torch.cuda.Event | None,
    ) -> None:
        if start_event is not None:
            end_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            end_event.record()
            end_event.synchronize()
            elapsed = float(start_event.elapsed_time(end_event))
        else:
            elapsed = (time.perf_counter_ns() - started_ns) / 1_000_000
        measurement = CollectiveMeasurement(
            collective_id=plan.collective_id,
            operation=plan.operation,
            rank_count=len(plan.rank_ids),
            payload_bytes=payload_bytes,
            logical_aggregate_bytes=self._logical_bytes(
                plan.operation, len(plan.rank_ids), payload_bytes
            ),
            actual_same_device_time_ms=elapsed,
            deterministic_rank_order=self.deterministic_rank_order,
        )
        self.trace.append(measurement.payload())

    async def broadcast(
        self, plan: CollectivePlan, source_rank: str, value: torch.Tensor
    ) -> TensorMap:
        await self._before(plan)
        if source_rank not in plan.rank_ids:
            raise ValueError("broadcast source is not a group member")
        started, event = self._measure_start(value)
        result = {rank: value.clone() for rank in plan.rank_ids}
        self._record(
            plan=plan,
            sample=value,
            payload_bytes=int(value.numel() * value.element_size()),
            started_ns=started,
            start_event=event,
        )
        return result

    async def all_reduce_sum(self, plan: CollectivePlan, values: TensorMap) -> TensorMap:
        await self._before(plan)
        ordered = self._ordered(plan, values)
        sample = ordered[0][1]
        started, event = self._measure_start(sample)
        reduced = sample.clone()
        for _, value in ordered[1:]:
            if tuple(value.shape) != tuple(reduced.shape):
                raise ValueError("all-reduce tensor shapes differ")
            reduced.add_(value)
        result = {rank: reduced.clone() for rank in plan.rank_ids}
        self._record(
            plan=plan,
            sample=sample,
            payload_bytes=int(sample.numel() * sample.element_size()),
            started_ns=started,
            start_event=event,
        )
        return result

    async def all_gather(
        self, plan: CollectivePlan, values: TensorMap, *, dim: int = 0
    ) -> TensorMap:
        await self._before(plan)
        ordered = self._ordered(plan, values)
        sample = ordered[0][1]
        started, event = self._measure_start(sample)
        gathered = torch.cat([value for _, value in ordered], dim=dim)
        result = {rank: gathered.clone() for rank in plan.rank_ids}
        payload = sum(int(value.numel() * value.element_size()) for _, value in ordered)
        self._record(
            plan=plan,
            sample=sample,
            payload_bytes=payload,
            started_ns=started,
            start_event=event,
        )
        return result

    async def reduce_scatter_sum(
        self, plan: CollectivePlan, values: TensorMap, *, dim: int = 0
    ) -> TensorMap:
        await self._before(plan)
        ordered = self._ordered(plan, values)
        sample = ordered[0][1]
        if sample.shape[dim] % len(plan.rank_ids):
            raise ValueError("reduce-scatter dimension is not divisible by rank count")
        started, event = self._measure_start(sample)
        reduced = sample.clone()
        for _, value in ordered[1:]:
            if tuple(value.shape) != tuple(reduced.shape):
                raise ValueError("reduce-scatter tensor shapes differ")
            reduced.add_(value)
        chunks = torch.chunk(reduced, len(plan.rank_ids), dim=dim)
        result = {
            rank: chunk.contiguous() for rank, chunk in zip(plan.rank_ids, chunks, strict=True)
        }
        self._record(
            plan=plan,
            sample=sample,
            payload_bytes=int(sample.numel() * sample.element_size()),
            started_ns=started,
            start_event=event,
        )
        return result

    async def all_to_all(
        self,
        plan: CollectivePlan,
        values: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, dict[str, torch.Tensor]]:
        await self._before(plan)
        ordered = self._ordered(plan, values)
        for source, destinations in ordered:
            if set(destinations) != set(plan.rank_ids):
                raise ValueError(f"all-to-all source {source} does not address every rank")
        sample = next(iter(ordered[0][1].values()))
        started, event = self._measure_start(sample)
        result = {
            destination: {source: values[source][destination].clone() for source in plan.rank_ids}
            for destination in plan.rank_ids
        }
        total = sum(
            int(value.numel() * value.element_size())
            for destinations in values.values()
            for value in destinations.values()
        )
        payload = total // max(len(plan.rank_ids) ** 2, 1)
        self._record(
            plan=plan,
            sample=sample,
            payload_bytes=payload,
            started_ns=started,
            start_event=event,
        )
        return result

    async def gather_to_leader(
        self,
        plan: CollectivePlan,
        values: TensorMap,
        *,
        leader_rank: str,
        dim: int = 0,
    ) -> dict[str, torch.Tensor | None]:
        await self._before(plan)
        ordered = self._ordered(plan, values)
        if leader_rank not in plan.rank_ids:
            raise ValueError("gather leader is not a group member")
        sample = ordered[0][1]
        started, event = self._measure_start(sample)
        gathered = torch.cat([value for _, value in ordered], dim=dim)
        result: dict[str, torch.Tensor | None] = {rank: None for rank in plan.rank_ids}
        result[leader_rank] = gathered
        payload = sum(int(value.numel() * value.element_size()) for _, value in ordered)
        self._record(
            plan=plan,
            sample=sample,
            payload_bytes=payload,
            started_ns=started,
            start_event=event,
        )
        return result

    async def distributed_argmax(
        self,
        plan: CollectivePlan,
        candidates: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        await self._before(plan)
        ordered = self._ordered(plan, candidates)
        sample = ordered[0][1][0]
        started, event = self._measure_start(sample)
        values = torch.stack([candidate[0] for _, candidate in ordered], dim=0)
        tokens = torch.stack([candidate[1] for _, candidate in ordered], dim=0)
        maximum = values.max(dim=0).values
        sentinel = torch.iinfo(tokens.dtype).max
        selected = torch.where(values == maximum.unsqueeze(0), tokens, sentinel).min(dim=0).values
        result = {rank: (maximum.clone(), selected.clone()) for rank in plan.rank_ids}
        payload = int(sample.numel() * (sample.element_size() + ordered[0][1][1].element_size()))
        self._record(
            plan=plan,
            sample=sample,
            payload_bytes=payload,
            started_ns=started,
            start_event=event,
        )
        return result

    async def barrier(self, plan: CollectivePlan) -> None:
        await self._before(plan)
        self.trace.append(
            CollectiveMeasurement(
                collective_id=plan.collective_id,
                operation="barrier",
                rank_count=len(plan.rank_ids),
                payload_bytes=0,
                logical_aggregate_bytes=0,
                actual_same_device_time_ms=0.0,
                deterministic_rank_order=self.deterministic_rank_order,
            ).payload()
        )


class UnimplementedNetworkCollectiveBackend:
    """Explicit non-implementation used to prevent accidental network claims."""

    def __init__(self, transport_name: str) -> None:
        self._transport_name = transport_name

    @property
    def transport_name(self) -> str:
        return self._transport_name

    async def connect(self, rank_id: str, endpoint: str) -> None:
        raise NotImplementedError(f"{self.transport_name} collective transport is interface-only")

    async def close(self) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        async def unavailable(*args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise NotImplementedError(f"{self.transport_name} collective {name} is interface-only")

        return unavailable
