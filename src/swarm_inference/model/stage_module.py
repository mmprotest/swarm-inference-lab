"""Common stage-module execution interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from swarm_inference.config.models import OperationKind

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True)
class StageExecutionMetadata:
    """GPU-native metadata for one stage request."""

    request_id: str
    token_position: int
    sequence_length: int
    cache_generation: int = 0
    route_generation: int = 0
    diagnostic: bool = False


@dataclass(frozen=True, slots=True)
class BatchExecutionMetadata:
    """Metadata for a true model batch.

    The initial production path intentionally accepts homogeneous sequence
    lengths and cache positions. Heterogeneous requests remain visible to the
    scheduler instead of being padded without accounting.
    """

    requests: tuple[StageExecutionMetadata, ...]
    padded_sequence_length: int | None = None
    padding_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.requests:
            raise ValueError("batch metadata requires at least one request")
        if self.padding_tokens < 0:
            raise ValueError("padding_tokens cannot be negative")

    @property
    def batch_size(self) -> int:
        return len(self.requests)

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(item.request_id for item in self.requests)

    @property
    def token_position(self) -> int:
        positions = {item.token_position for item in self.requests}
        if len(positions) != 1:
            raise ValueError("fast batch requires one homogeneous token position")
        return next(iter(positions))

    @property
    def sequence_length(self) -> int:
        lengths = {item.sequence_length for item in self.requests}
        if len(lengths) != 1:
            raise ValueError("fast batch requires one homogeneous sequence length")
        return next(iter(lengths))

    @property
    def cache_generation(self) -> int:
        generations = {item.cache_generation for item in self.requests}
        if len(generations) != 1:
            raise ValueError("fast batch requires one homogeneous cache generation")
        return next(iter(generations))

    @property
    def route_generation(self) -> int:
        generations = {item.route_generation for item in self.requests}
        if len(generations) != 1:
            raise ValueError("fast batch requires one homogeneous route generation")
        return next(iter(generations))


@runtime_checkable
class StageModule(Protocol):
    stage_id: int
    required_memory_bytes: int

    def execute(
        self,
        activation: np.ndarray,
        *,
        request_id: str,
        operation: OperationKind,
        token_position: int,
        sequence_length: int,
        cache_generation: int,
        route_generation: int = 0,
    ) -> np.ndarray: ...

    def cancel(self, request_id: str) -> None: ...

    def cache_bytes(self) -> int: ...

    def state_summary(self) -> dict[str, Any]: ...


@runtime_checkable
class FastStageModule(Protocol):
    """GPU-native interface used only at a local CUDA stage boundary."""

    stage_id: int
    required_memory_bytes: int
    execution_profile: str

    def prefill_cuda(
        self,
        input_tensor: torch.Tensor,
        metadata: StageExecutionMetadata,
    ) -> torch.Tensor: ...

    def decode_cuda(
        self,
        input_tensor: torch.Tensor,
        metadata: StageExecutionMetadata,
    ) -> torch.Tensor: ...

    def prefill_batch_cuda(
        self,
        input_tensors: torch.Tensor,
        metadata: BatchExecutionMetadata,
    ) -> torch.Tensor: ...

    def decode_batch_cuda(
        self,
        input_tensors: torch.Tensor,
        metadata: BatchExecutionMetadata,
    ) -> torch.Tensor: ...


@runtime_checkable
class BatchStageModule(Protocol):
    """Remote-boundary compatibility interface with one true batched forward."""

    stage_id: int
    required_memory_bytes: int

    def execute_batch(
        self,
        activations: np.ndarray,
        *,
        metadata: BatchExecutionMetadata,
        operation: OperationKind,
    ) -> np.ndarray: ...
