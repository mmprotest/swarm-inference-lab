"""Native matched-microshard ownership and deterministic reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from swarm_inference.execution.expert import ExpertWeights, reduce_partials
from swarm_inference.protocol.expert import ReductionMode


@dataclass(frozen=True, slots=True, order=True)
class MicroshardRange:
    worker_id: str
    layer_id: int
    expert_id: int
    hidden_start: int
    hidden_end: int
    logical_intermediate_dimension: int
    content_hash: str
    quantization_group_size: int | None = None
    shard_dimension: str = "intermediate"
    reduction_semantics: str = "sum"
    tensor_group_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.layer_id < 0 or self.expert_id < 0:
            raise ValueError("microshard layer and expert IDs must be non-negative")
        if (
            self.hidden_start < 0
            or self.hidden_end <= self.hidden_start
            or self.hidden_end > self.logical_intermediate_dimension
        ):
            raise ValueError("microshard range is outside the logical expert")
        if not self.worker_id or not self.content_hash or not self.shard_dimension:
            raise ValueError("microshard worker identity and content hash are required")
        if not self.reduction_semantics:
            raise ValueError("microshard reduction semantics are required")
        group = self.quantization_group_size
        if group is not None:
            if group <= 0:
                raise ValueError("quantisation group size must be positive")
            if self.hidden_start % group:
                raise ValueError("microshard start splits a native quantisation group")
            if self.hidden_end != self.logical_intermediate_dimension and self.hidden_end % group:
                raise ValueError("microshard end splits a native quantisation group")

    @property
    def width(self) -> int:
        return self.hidden_end - self.hidden_start

    @property
    def shard_start(self) -> int:
        return self.hidden_start

    @property
    def shard_end(self) -> int:
        return self.hidden_end

    @property
    def logical_shard_extent(self) -> int:
        return self.logical_intermediate_dimension

    @property
    def owns_full_expert(self) -> bool:
        return self.hidden_start == 0 and self.hidden_end == self.logical_intermediate_dimension


def physical_microshard_ownership(
    ownership: list[MicroshardRange],
    *,
    require_distributed: bool = True,
) -> dict[str, Any]:
    """Validate an exact, gap-free union and prove per-worker physical bounds."""

    if not ownership:
        raise ValueError("at least one microshard is required")
    identities = {
        (
            item.layer_id,
            item.expert_id,
            item.shard_dimension,
            item.reduction_semantics,
            item.tensor_group_ids,
        )
        for item in ownership
    }
    logical_widths = {item.logical_intermediate_dimension for item in ownership}
    if len(identities) != 1 or len(logical_widths) != 1:
        raise ValueError("microshards must describe exactly one logical expert")
    ordered = sorted(
        ownership, key=lambda item: (item.hidden_start, item.hidden_end, item.worker_id)
    )
    cursor = 0
    for item in ordered:
        if item.hidden_start != cursor:
            kind = "overlap" if item.hidden_start < cursor else "gap"
            raise ValueError(f"microshard union contains a {kind} at {cursor}")
        cursor = item.hidden_end
    logical_width = next(iter(logical_widths))
    if cursor != logical_width:
        raise ValueError("microshard union does not cover the complete expert")
    by_worker: dict[str, int] = {}
    for item in ordered:
        by_worker[item.worker_id] = by_worker.get(item.worker_id, 0) + item.width
    full_owners = [worker for worker, width in by_worker.items() if width >= logical_width]
    if require_distributed and (len(by_worker) < 2 or full_owners):
        raise ValueError("native microsharding requires multiple workers and no full-expert owner")
    return {
        "valid": True,
        "layer_id": ordered[0].layer_id,
        "expert_id": ordered[0].expert_id,
        "logical_intermediate_dimension": logical_width,
        "shard_dimension": ordered[0].shard_dimension,
        "reduction_semantics": ordered[0].reduction_semantics,
        "tensor_group_ids": list(ordered[0].tensor_group_ids),
        "worker_count": len(by_worker),
        "shard_count": len(ordered),
        "worker_hidden_units": by_worker,
        "workers_owning_full_expert": full_owners,
        "no_worker_owns_full_expert": not full_owners,
    }


def validate_resident_microshard(weights: ExpertWeights, ownership: MicroshardRange) -> None:
    if weights.hidden_offset != ownership.hidden_start:
        raise ValueError("resident microshard start does not match its ownership descriptor")
    if weights.hidden_offset + weights.intermediate_dimension != ownership.hidden_end:
        raise ValueError("resident microshard end does not match its ownership descriptor")
    if weights.logical_width != ownership.logical_intermediate_dimension:
        raise ValueError("resident microshard logical width does not match its descriptor")
    if weights.content_hash != ownership.content_hash:
        raise ValueError("resident microshard content hash does not match its descriptor")


def reconstruct_microshard_result(
    partials: list[tuple[MicroshardRange, np.ndarray]],
    *,
    mode: ReductionMode = ReductionMode.FIXED_ORDER_FP32,
) -> np.ndarray:
    ownership = [item for item, _ in partials]
    physical_microshard_ownership(ownership, require_distributed=False)
    # The range is included in the stable key so retry/reordering cannot alter
    # exact reduction order.
    keyed = [
        (
            f"{item.hidden_start:012d}:{item.hidden_end:012d}:{item.worker_id}",
            partial,
        )
        for item, partial in partials
    ]
    return reduce_partials(keyed, mode=mode)


__all__ = [
    "MicroshardRange",
    "physical_microshard_ownership",
    "reconstruct_microshard_result",
    "validate_resident_microshard",
]
