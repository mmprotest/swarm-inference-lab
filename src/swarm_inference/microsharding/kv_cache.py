"""Rank-local KV-cache ownership, lifecycle, snapshot, and rollback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class KVCacheRecord:
    request_id: str
    layer_id: int
    tp_rank: int
    global_kv_head_ids: tuple[int, ...]
    key: Any
    value: Any
    cache_generation: int = 0

    @property
    def sequence_length(self) -> int:
        return int(self.key.shape[-2])

    @property
    def dtype(self) -> str:
        return str(self.key.dtype).replace("torch.", "")

    @property
    def bytes(self) -> int:
        return int(
            self.key.numel() * self.key.element_size()
            + self.value.numel() * self.value.element_size()
        )

    def summary(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "layer_id": self.layer_id,
            "tp_rank": self.tp_rank,
            "global_kv_head_ids": list(self.global_kv_head_ids),
            "sequence_length": self.sequence_length,
            "cache_generation": self.cache_generation,
            "dtype": self.dtype,
            "bytes": self.bytes,
        }


@dataclass(frozen=True, slots=True)
class KVCacheSnapshot:
    request_id: str
    lengths: dict[tuple[int, int], int]
    generations: dict[tuple[int, int], int]


class PartitionedKVCache:
    """Own only rank-assigned heads and make all lifecycle changes observable."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, int, int], KVCacheRecord] = {}
        self._history: list[dict[str, Any]] = []

    def append(
        self,
        *,
        request_id: str,
        layer_id: int,
        tp_rank: int,
        global_kv_head_ids: list[int] | tuple[int, ...],
        key: Any,
        value: Any,
        cache_generation: int = 0,
    ) -> tuple[Any, Any]:
        import torch

        if tuple(key.shape) != tuple(value.shape):
            raise ValueError("key and value cache shapes must match")
        if key.ndim != 4:
            raise ValueError("KV tensors must have [batch, heads, sequence, head_dim] shape")
        head_ids = tuple(int(item) for item in global_kv_head_ids)
        if len(head_ids) != key.shape[1] or len(set(head_ids)) != len(head_ids):
            raise ValueError("global KV head IDs must uniquely describe the local head axis")
        cache_key = (request_id, layer_id, tp_rank)
        existing = self._records.get(cache_key)
        if existing is None:
            record = KVCacheRecord(
                request_id=request_id,
                layer_id=layer_id,
                tp_rank=tp_rank,
                global_kv_head_ids=head_ids,
                key=key,
                value=value,
                cache_generation=cache_generation,
            )
            self._records[cache_key] = record
        else:
            if existing.global_kv_head_ids != head_ids:
                raise ValueError("KV-head ownership changed during a request")
            if existing.cache_generation != cache_generation:
                raise ValueError("cache generation changed without cleanup or rollback")
            if existing.key.shape[:2] != key.shape[:2] or existing.key.shape[-1] != key.shape[-1]:
                raise ValueError("KV append shape is incompatible with the existing record")
            record = existing
            record.key = torch.cat((record.key, key), dim=-2)
            record.value = torch.cat((record.value, value), dim=-2)
        self._history.append({"event": "append", **record.summary()})
        return record.key, record.value

    def get(self, request_id: str, layer_id: int, tp_rank: int) -> KVCacheRecord | None:
        return self._records.get((request_id, layer_id, tp_rank))

    def snapshot(self, request_id: str) -> KVCacheSnapshot:
        records = {
            (layer, rank): record
            for (candidate, layer, rank), record in self._records.items()
            if candidate == request_id
        }
        snapshot = KVCacheSnapshot(
            request_id=request_id,
            lengths={key: record.sequence_length for key, record in records.items()},
            generations={key: record.cache_generation for key, record in records.items()},
        )
        self._history.append(
            {
                "event": "snapshot",
                "request_id": request_id,
                "record_count": len(records),
                "lengths": {f"{key[0]}:{key[1]}": value for key, value in snapshot.lengths.items()},
            }
        )
        return snapshot

    def rollback(self, snapshot: KVCacheSnapshot) -> None:
        request_keys = [key for key in self._records if key[0] == snapshot.request_id]
        expected = {(snapshot.request_id, layer, rank) for layer, rank in snapshot.lengths}
        for key in request_keys:
            if key not in expected:
                del self._records[key]
                continue
            record = self._records[key]
            short_key = (key[1], key[2])
            length = snapshot.lengths[short_key]
            if length > record.sequence_length:
                raise ValueError("snapshot length exceeds the current KV-cache length")
            record.key = record.key[..., :length, :].contiguous()
            record.value = record.value[..., :length, :].contiguous()
            record.cache_generation = snapshot.generations[short_key] + 1
        self._history.append(
            {"event": "rollback", "request_id": snapshot.request_id, "record_count": len(expected)}
        )

    def branch(self, source_request_id: str, target_request_id: str) -> None:
        if source_request_id == target_request_id:
            raise ValueError("cache branch target must differ from the source")
        if any(key[0] == target_request_id for key in self._records):
            raise ValueError("cache branch target already exists")
        source = [record for key, record in self._records.items() if key[0] == source_request_id]
        if not source:
            raise KeyError(source_request_id)
        for record in source:
            clone = KVCacheRecord(
                request_id=target_request_id,
                layer_id=record.layer_id,
                tp_rank=record.tp_rank,
                global_kv_head_ids=record.global_kv_head_ids,
                key=record.key.clone(),
                value=record.value.clone(),
                cache_generation=record.cache_generation + 1,
            )
            self._records[(target_request_id, clone.layer_id, clone.tp_rank)] = clone
        self._history.append(
            {
                "event": "branch",
                "request_id": source_request_id,
                "target_request_id": target_request_id,
                "record_count": len(source),
            }
        )

    def cleanup(self, request_id: str) -> int:
        keys = [key for key in self._records if key[0] == request_id]
        released = sum(self._records[key].bytes for key in keys)
        for key in keys:
            del self._records[key]
        self._history.append(
            {"event": "cleanup", "request_id": request_id, "released_bytes": released}
        )
        return released

    def inspect(self, request_id: str | None = None) -> list[dict[str, Any]]:
        return [
            record.summary()
            for key, record in sorted(self._records.items())
            if request_id is None or key[0] == request_id
        ]

    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def bytes(self, request_id: str | None = None) -> int:
        return sum(item["bytes"] for item in self.inspect(request_id))

    def validate_ownership(
        self,
        *,
        request_id: str,
        layer_id: int,
        global_kv_head_count: int,
    ) -> dict[str, Any]:
        records = [
            record
            for (candidate, candidate_layer, _), record in self._records.items()
            if candidate == request_id and candidate_layer == layer_id
        ]
        owners: dict[int, list[int]] = {head: [] for head in range(global_kv_head_count)}
        for record in records:
            for head in record.global_kv_head_ids:
                if head not in owners:
                    raise ValueError(f"rank owns out-of-range KV head {head}")
                owners[head].append(record.tp_rank)
        missing = [head for head, ranks in owners.items() if not ranks]
        replicated = {head: ranks for head, ranks in owners.items() if len(ranks) > 1}
        actual_bytes = sum(record.bytes for record in records)
        unique_bytes = 0
        if records:
            sample = records[0]
            bytes_per_head = sample.bytes // max(len(sample.global_kv_head_ids), 1)
            unique_bytes = bytes_per_head * global_kv_head_count
        return {
            "status": "PASS" if not missing else "FAIL",
            "request_id": request_id,
            "layer_id": layer_id,
            "global_kv_head_count": global_kv_head_count,
            "ownership": owners,
            "missing_heads": missing,
            "replicated_heads": replicated,
            "actual_bytes": actual_bytes,
            "unique_bytes": unique_bytes,
            "replicated_bytes": max(actual_bytes - unique_bytes, 0),
        }
