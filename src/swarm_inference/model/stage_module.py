"""Common stage-module execution interface."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from swarm_inference.config.models import OperationKind


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
    ) -> np.ndarray: ...

    def cancel(self, request_id: str) -> None: ...

    def cache_bytes(self) -> int: ...

    def state_summary(self) -> dict[str, Any]: ...
