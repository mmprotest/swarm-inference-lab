"""Optional prefill-only sequence-parallel activation operations."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class SequenceParallelResult:
    output: torch.Tensor
    position_ranges: tuple[tuple[int, int], ...]
    collective_operations: tuple[str, ...]
    logical_collective_bytes: int
    enabled: bool
    disable_reason: str | None


def sequence_parallel_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    *,
    degree: int,
    eps: float,
    decode: bool = False,
) -> SequenceParallelResult:
    """Shard token positions for local RMSNorm, then all-gather in order.

    Attention still receives the complete gathered sequence.  This deliberately
    models sequence parallelism, not context parallelism.  Single-token decode
    is disabled because it creates collective overhead without useful work.
    """

    if hidden_states.ndim != 3:
        raise ValueError("sequence-parallel input must have [batch, sequence, hidden]")
    if tuple(weight.shape) != (hidden_states.shape[-1],):
        raise ValueError("RMSNorm weight does not match hidden width")
    if degree <= 0:
        raise ValueError("sequence-parallel degree must be positive")
    sequence = int(hidden_states.shape[1])
    if decode or sequence == 1 or degree == 1:
        normalised = _rms_norm(hidden_states, weight, eps)
        reason = (
            "single-token decode is not latency-beneficial" if decode or sequence == 1 else None
        )
        return SequenceParallelResult(
            output=normalised,
            position_ranges=((0, sequence),),
            collective_operations=(),
            logical_collective_bytes=0,
            enabled=False,
            disable_reason=reason,
        )
    if degree > sequence:
        raise ValueError("sequence-parallel degree cannot exceed the sequence length")
    quotient, remainder = divmod(sequence, degree)
    ranges: list[tuple[int, int]] = []
    cursor = 0
    parts: list[torch.Tensor] = []
    for rank in range(degree):
        end = cursor + quotient + (1 if rank < remainder else 0)
        ranges.append((cursor, end))
        parts.append(_rms_norm(hidden_states[:, cursor:end], weight, eps))
        cursor = end
    gathered = torch.cat(parts, dim=1)
    payload = int(gathered.numel() * gathered.element_size())
    return SequenceParallelResult(
        output=gathered,
        position_ranges=tuple(ranges),
        collective_operations=("all_gather",),
        logical_collective_bytes=(degree - 1) * payload,
        enabled=True,
        disable_reason=None,
    )


def _rms_norm(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    source_dtype = value.dtype
    variance = value.float().pow(2).mean(dim=-1, keepdim=True)
    return weight * (value.float() * torch.rsqrt(variance + eps)).to(source_dtype)
