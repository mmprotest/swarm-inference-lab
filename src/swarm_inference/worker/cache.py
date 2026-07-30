"""Bounded accounting for stage-local request caches."""

from __future__ import annotations

from dataclasses import dataclass

from swarm_inference.exceptions import MemoryLimitExceededError


@dataclass(slots=True)
class CacheAllocation:
    request_id: str
    generation: int
    bytes_used: int


class CacheBudget:
    def __init__(self, *, maximum_bytes: int) -> None:
        if maximum_bytes < 0:
            raise ValueError("maximum_bytes cannot be negative")
        self.maximum_bytes = maximum_bytes
        self._allocations: dict[tuple[str, int], CacheAllocation] = {}

    @property
    def used_bytes(self) -> int:
        return sum(item.bytes_used for item in self._allocations.values())

    def set(self, request_id: str, generation: int, bytes_used: int) -> None:
        if bytes_used < 0:
            raise ValueError("bytes_used cannot be negative")
        key = (request_id, generation)
        previous = self._allocations.get(key)
        projected = self.used_bytes - (previous.bytes_used if previous else 0) + bytes_used
        if projected > self.maximum_bytes:
            raise MemoryLimitExceededError(
                f"stage-local caches require {projected} bytes, limit is {self.maximum_bytes}"
            )
        self._allocations[key] = CacheAllocation(
            request_id=request_id,
            generation=generation,
            bytes_used=bytes_used,
        )

    def delete_request(self, request_id: str) -> int:
        removed = 0
        for key in [key for key in self._allocations if key[0] == request_id]:
            removed += self._allocations.pop(key).bytes_used
        return removed
