"""Narrow, typed interfaces for stateful stage execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch


@dataclass(frozen=True, slots=True)
class WeightOwnership:
    stage_id: int
    layer_start: int
    layer_end: int
    parameter_names: tuple[str, ...]
    parameter_bytes: int
    parameter_count: int
    owns_embeddings: bool
    owns_final_norm: bool
    owns_output_projection: bool
    ownership_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "parameter_names": list(self.parameter_names),
            "parameter_bytes": self.parameter_bytes,
            "parameter_count": self.parameter_count,
            "owns_embeddings": self.owns_embeddings,
            "owns_final_norm": self.owns_final_norm,
            "owns_output_projection": self.owns_output_projection,
            "ownership_hash": self.ownership_hash,
        }


@dataclass(frozen=True, slots=True)
class StageExecutionResult:
    hidden_states: torch.Tensor
    stage_boundary_hidden_states: torch.Tensor
    router_logits: tuple[torch.Tensor, ...]
    final_hidden_states: torch.Tensor | None
    logits: torch.Tensor | None
    sampled_token_ids: torch.Tensor | None
    all_sampled_token_ids: torch.Tensor | None
    cache_sequence_length: int
    compute_ns: int


@runtime_checkable
class StageExecutor(Protocol):
    """Stateful execution contract shared by contiguous model stages."""

    @property
    def ownership(self) -> WeightOwnership: ...

    def open_session(self, session_id: str) -> None: ...

    def execute_prefill(
        self,
        *,
        session_id: str,
        token_ids: torch.Tensor,
        cache_position_start: int,
    ) -> StageExecutionResult: ...

    def execute_decode(
        self,
        *,
        session_id: str,
        hidden_states: torch.Tensor,
        cache_position_start: int,
    ) -> StageExecutionResult: ...

    def close_session(self, session_id: str) -> int: ...

    def cancel_session(self, session_id: str) -> int: ...

    def kv_cache_bytes(self, session_id: str) -> int: ...

    def close(self) -> None: ...


__all__ = ["StageExecutionResult", "StageExecutor", "WeightOwnership"]
