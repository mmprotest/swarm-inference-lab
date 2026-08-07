"""Planner-visible tensor residency, movement, and cache-effect accounting."""

from __future__ import annotations

import threading
import time
from enum import StrEnum

from pydantic import ConfigDict, Field, NonNegativeInt

from swarm_inference.config.models import StrictModel


class ResidencyTier(StrEnum):
    VRAM = "vram"
    RAM = "ram"
    MAPPED = "mapped"
    STORAGE = "storage"


class ResidencyKind(StrEnum):
    MODEL_TENSOR = "model_tensor"
    EXPERT_CACHE = "expert_cache"
    KV_CACHE = "kv_cache"
    CUDA_GRAPH = "cuda_graph"
    COMPILE_ARTIFACT = "compile_artifact"
    GGUF_RPC_CACHE = "gguf_rpc_cache"


class ResidencyRecord(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    allocation_id: str
    worker_id: str
    model_fingerprint: str
    kind: ResidencyKind
    tier: ResidencyTier
    bytes: NonNegativeInt
    device_id: str | None = None
    content_hash: str | None = None
    created_unix_ns: NonNegativeInt
    last_accessed_unix_ns: NonNegativeInt


class MovementRecord(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    movement_id: str
    model_fingerprint: str
    source_worker_id: str | None = None
    destination_worker_id: str
    source_tier: ResidencyTier
    destination_tier: ResidencyTier
    bytes: NonNegativeInt
    elapsed_ns: NonNegativeInt
    timestamp_unix_ns: NonNegativeInt
    reason: str


class CacheEffectMetrics(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hits: NonNegativeInt = 0
    misses: NonNegativeInt = 0
    bytes_loaded: NonNegativeInt = 0
    bytes_evicted: NonNegativeInt = 0
    prefetch_useful_bytes: NonNegativeInt = 0
    prefetch_wasted_bytes: NonNegativeInt = 0
    stall_ns: NonNegativeInt = 0


class ResidencySnapshot(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    records: tuple[ResidencyRecord, ...]
    recent_movements: tuple[MovementRecord, ...]
    cache: CacheEffectMetrics
    bytes_by_worker_tier: dict[str, NonNegativeInt] = Field(default_factory=dict)
    captured_unix_ns: NonNegativeInt


class ResidencyTracker:
    """Thread-safe physical-state inventory; hit percentage is never its objective."""

    def __init__(self, *, maximum_movement_records: int = 4096) -> None:
        if maximum_movement_records <= 0:
            raise ValueError("movement history bound must be positive")
        self.maximum_movement_records = maximum_movement_records
        self._records: dict[str, ResidencyRecord] = {}
        self._movements: list[MovementRecord] = []
        self._cache = CacheEffectMetrics()
        self._lock = threading.RLock()

    def put(
        self,
        *,
        allocation_id: str,
        worker_id: str,
        model_fingerprint: str,
        kind: ResidencyKind,
        tier: ResidencyTier,
        bytes: int,
        device_id: str | None = None,
        content_hash: str | None = None,
        now_unix_ns: int | None = None,
    ) -> ResidencyRecord:
        if bytes < 0:
            raise ValueError("resident bytes cannot be negative")
        now = time.time_ns() if now_unix_ns is None else now_unix_ns
        with self._lock:
            previous = self._records.get(allocation_id)
            created = previous.created_unix_ns if previous is not None else now
            record = ResidencyRecord(
                allocation_id=allocation_id,
                worker_id=worker_id,
                model_fingerprint=model_fingerprint,
                kind=kind,
                tier=tier,
                bytes=bytes,
                device_id=device_id,
                content_hash=content_hash,
                created_unix_ns=created,
                last_accessed_unix_ns=now,
            )
            self._records[allocation_id] = record
            return record

    def touch(self, allocation_id: str, *, now_unix_ns: int | None = None) -> None:
        now = time.time_ns() if now_unix_ns is None else now_unix_ns
        with self._lock:
            try:
                record = self._records[allocation_id]
            except KeyError as exc:
                raise KeyError(f"unknown residency allocation {allocation_id!r}") from exc
            self._records[allocation_id] = record.model_copy(
                update={"last_accessed_unix_ns": now}
            )

    def release(self, allocation_id: str) -> ResidencyRecord | None:
        with self._lock:
            return self._records.pop(allocation_id, None)

    def record_movement(self, movement: MovementRecord) -> None:
        with self._lock:
            self._movements.append(movement)
            overflow = len(self._movements) - self.maximum_movement_records
            if overflow > 0:
                del self._movements[:overflow]

    def record_cache_effect(
        self,
        *,
        hit: bool | None = None,
        bytes_loaded: int = 0,
        bytes_evicted: int = 0,
        prefetch_useful_bytes: int = 0,
        prefetch_wasted_bytes: int = 0,
        stall_ns: int = 0,
    ) -> None:
        values = (
            bytes_loaded,
            bytes_evicted,
            prefetch_useful_bytes,
            prefetch_wasted_bytes,
            stall_ns,
        )
        if any(value < 0 for value in values):
            raise ValueError("cache-effect counters cannot be negative")
        with self._lock:
            current = self._cache
            self._cache = CacheEffectMetrics(
                hits=current.hits + int(hit is True),
                misses=current.misses + int(hit is False),
                bytes_loaded=current.bytes_loaded + bytes_loaded,
                bytes_evicted=current.bytes_evicted + bytes_evicted,
                prefetch_useful_bytes=(
                    current.prefetch_useful_bytes + prefetch_useful_bytes
                ),
                prefetch_wasted_bytes=(
                    current.prefetch_wasted_bytes + prefetch_wasted_bytes
                ),
                stall_ns=current.stall_ns + stall_ns,
            )

    def snapshot(self, *, now_unix_ns: int | None = None) -> ResidencySnapshot:
        now = time.time_ns() if now_unix_ns is None else now_unix_ns
        with self._lock:
            records = tuple(self._records[key] for key in sorted(self._records))
            movements = tuple(self._movements)
            cache = self._cache
        totals: dict[str, int] = {}
        for record in records:
            key = f"{record.worker_id}:{record.tier.value}"
            totals[key] = totals.get(key, 0) + int(record.bytes)
        return ResidencySnapshot(
            records=records,
            recent_movements=movements,
            cache=cache,
            bytes_by_worker_tier=totals,
            captured_unix_ns=now,
        )


__all__ = [
    "CacheEffectMetrics",
    "MovementRecord",
    "ResidencyKind",
    "ResidencyRecord",
    "ResidencySnapshot",
    "ResidencyTier",
    "ResidencyTracker",
]
