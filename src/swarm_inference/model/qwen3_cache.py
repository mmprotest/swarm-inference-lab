"""Preallocated stage-local KV storage for the Qwen3 performance profile."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from swarm_inference.exceptions import UnsupportedCacheFormatError


@dataclass(slots=True)
class StaticCacheSnapshot:
    sequence_length: int
    keys: tuple[Any, ...]
    values: tuple[Any, ...]


@dataclass(slots=True)
class StaticCacheLayer:
    keys: Any
    values: Any
    is_initialized: bool = True


class StaticStageKVCache:
    """A contiguous, fixed-capacity cache for one stage and one request batch.

    Storage is allocated once. Decoder-layer calls update it in place and
    receive views into the occupied prefix. `fixed_shape=True` returns full
    capacity and is reserved for compile/CUDA-graph experiments that also pass
    a mask covering unoccupied positions.
    """

    def __init__(
        self,
        *,
        torch_module: Any,
        request_ids: tuple[str, ...],
        model_revision: str,
        stage_id: int,
        layer_start: int,
        layer_end: int,
        route_generation: int,
        cache_generation: int,
        max_sequence_length: int,
        key_value_head_count: int,
        head_dimension: int,
        dtype: Any,
        device: Any,
        fixed_shape: bool = False,
    ) -> None:
        if not request_ids:
            raise ValueError("static cache requires at least one request")
        if layer_end <= layer_start:
            raise ValueError("static cache layer interval must be non-empty")
        if max_sequence_length <= 0:
            raise ValueError("static cache max_sequence_length must be positive")
        self.torch = torch_module
        self.request_ids = request_ids
        self.request_slots = {request_id: index for index, request_id in enumerate(request_ids)}
        if len(self.request_slots) != len(request_ids):
            raise ValueError("static cache request IDs must be unique")
        self.model_revision = model_revision
        self.stage_id = stage_id
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.route_generation = route_generation
        self.cache_generation = cache_generation
        self.max_sequence_length = max_sequence_length
        self.batch_size = len(request_ids)
        self.key_value_head_count = key_value_head_count
        self.head_dimension = head_dimension
        self.dtype = dtype
        self.device = device
        self.fixed_shape = fixed_shape
        shape = (
            self.batch_size,
            key_value_head_count,
            max_sequence_length,
            head_dimension,
        )
        self._keys = [
            torch_module.zeros(shape, dtype=dtype, device=device)
            for _ in range(layer_end - layer_start)
        ]
        self._values = [
            torch_module.zeros(shape, dtype=dtype, device=device)
            for _ in range(layer_end - layer_start)
        ]
        self.layers = [
            *([None] * layer_start),
            *[
                StaticCacheLayer(keys=key, values=value)
                for key, value in zip(self._keys, self._values, strict=True)
            ],
        ]
        self.sequence_length = 0
        self._pending_end: int | None = None
        self._snapshots: dict[str, StaticCacheSnapshot] = {}
        self.deleted = False
        self.append_count = 0
        self.allocation_count = 2 * (layer_end - layer_start)

    def global_to_local(self, global_layer_index: int) -> int:
        if not self.layer_start <= global_layer_index < self.layer_end:
            raise IndexError(
                f"global layer {global_layer_index} is outside stage "
                f"[{self.layer_start}, {self.layer_end})"
            )
        return global_layer_index - self.layer_start

    def local_to_global(self, local_layer_index: int) -> int:
        if not 0 <= local_layer_index < self.layer_end - self.layer_start:
            raise IndexError(f"local cache layer is out of range: {local_layer_index}")
        return self.layer_start + local_layer_index

    def prepare_append(self, *, token_position: int, query_length: int) -> None:
        self._require_live()
        if query_length <= 0:
            raise ValueError("query_length must be positive")
        if token_position != self.sequence_length:
            raise UnsupportedCacheFormatError(
                f"stage {self.stage_id} static cache position mismatch: "
                f"expected={self.sequence_length} actual={token_position}"
            )
        end = token_position + query_length
        if end > self.max_sequence_length:
            raise UnsupportedCacheFormatError(
                f"stage {self.stage_id} static cache capacity exceeded: "
                f"required={end} maximum={self.max_sequence_length}"
            )
        self._pending_end = end

    def commit_append(self) -> None:
        self._require_live()
        if self._pending_end is None:
            raise UnsupportedCacheFormatError("static cache has no prepared append to commit")
        self.sequence_length = self._pending_end
        self._pending_end = None
        self.append_count += 1

    def graph_advance(self, *, expected_position: int) -> None:
        """Advance metadata after a captured one-token cache update replays.

        CUDA graph replay executes the recorded ``index_copy_`` operations
        without re-entering Python, so the cache's accounting must be advanced
        explicitly at the measurement boundary.
        """

        self._require_live()
        if expected_position != self.sequence_length:
            raise UnsupportedCacheFormatError(
                f"stage {self.stage_id} graph cache position mismatch: "
                f"expected={self.sequence_length} actual={expected_position}"
            )
        if self.sequence_length >= self.max_sequence_length:
            raise UnsupportedCacheFormatError(
                f"stage {self.stage_id} graph cache capacity exceeded"
            )
        self.sequence_length += 1
        self.append_count += 1

    def update(
        self,
        key_states: Any,
        value_states: Any,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        self._require_live()
        if self._pending_end is None:
            raise UnsupportedCacheFormatError(
                "static cache update occurred without prepare_append()"
            )
        local_index = self.global_to_local(layer_idx)
        if int(key_states.shape[0]) != self.batch_size:
            raise UnsupportedCacheFormatError(
                f"static cache batch mismatch: cache={self.batch_size} "
                f"key={int(key_states.shape[0])}"
            )
        if key_states.dtype != self.dtype or value_states.dtype != self.dtype:
            raise UnsupportedCacheFormatError(
                f"static cache dtype mismatch: cache={self.dtype} "
                f"key={key_states.dtype} value={value_states.dtype}"
            )
        cache_position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        if cache_position is None:
            raise UnsupportedCacheFormatError(
                "Qwen3 static cache requires an explicit cache_position tensor"
            )
        keys = self._keys[local_index]
        values = self._values[local_index]
        keys.index_copy_(2, cache_position, key_states)
        values.index_copy_(2, cache_position, value_states)
        if self.fixed_shape:
            return keys, values
        return (
            keys[:, :, : self._pending_end, :],
            values[:, :, : self._pending_end, :],
        )

    def read(
        self,
        global_layer_index: int,
        *,
        through_position: int | None = None,
    ) -> tuple[Any, Any]:
        self._require_live()
        local_index = self.global_to_local(global_layer_index)
        end = self.sequence_length if through_position is None else through_position
        if not 0 <= end <= self.sequence_length:
            raise ValueError(
                f"read position {end} is outside occupied cache length {self.sequence_length}"
            )
        return (
            self._keys[local_index][:, :, :end, :],
            self._values[local_index][:, :, :end, :],
        )

    def snapshot(self, name: str) -> StaticCacheSnapshot:
        self._require_live()
        if not name:
            raise ValueError("cache snapshot name cannot be empty")
        snapshot = StaticCacheSnapshot(
            sequence_length=self.sequence_length,
            keys=tuple(value[:, :, : self.sequence_length, :].clone() for value in self._keys),
            values=tuple(value[:, :, : self.sequence_length, :].clone() for value in self._values),
        )
        self._snapshots[name] = snapshot
        return snapshot

    def restore(self, name: str) -> None:
        self._require_live()
        try:
            snapshot = self._snapshots[name]
        except KeyError as exc:
            raise KeyError(f"unknown static cache snapshot {name!r}") from exc
        self.rollback(0)
        for destination, source in zip(self._keys, snapshot.keys, strict=True):
            destination[:, :, : snapshot.sequence_length, :].copy_(source)
        for destination, source in zip(self._values, snapshot.values, strict=True):
            destination[:, :, : snapshot.sequence_length, :].copy_(source)
        self.sequence_length = snapshot.sequence_length

    def fork(self, *, request_ids: tuple[str, ...]) -> StaticStageKVCache:
        self._require_live()
        if len(request_ids) != self.batch_size:
            raise ValueError("cache fork must preserve batch size")
        forked = StaticStageKVCache(
            torch_module=self.torch,
            request_ids=request_ids,
            model_revision=self.model_revision,
            stage_id=self.stage_id,
            layer_start=self.layer_start,
            layer_end=self.layer_end,
            route_generation=self.route_generation,
            cache_generation=self.cache_generation,
            max_sequence_length=self.max_sequence_length,
            key_value_head_count=self.key_value_head_count,
            head_dimension=self.head_dimension,
            dtype=self.dtype,
            device=self.device,
            fixed_shape=self.fixed_shape,
        )
        for destination, source in zip(forked._keys, self._keys, strict=True):
            destination[:, :, : self.sequence_length, :].copy_(
                source[:, :, : self.sequence_length, :]
            )
        for destination, source in zip(forked._values, self._values, strict=True):
            destination[:, :, : self.sequence_length, :].copy_(
                source[:, :, : self.sequence_length, :]
            )
        forked.sequence_length = self.sequence_length
        return forked

    def rollback(self, sequence_length: int) -> None:
        self._require_live()
        if not 0 <= sequence_length <= self.sequence_length:
            raise ValueError(
                f"rollback position {sequence_length} is outside [0, {self.sequence_length}]"
            )
        if sequence_length < self.sequence_length:
            for tensor in (*self._keys, *self._values):
                tensor[:, :, sequence_length : self.sequence_length, :].zero_()
        self.sequence_length = sequence_length
        self._pending_end = None

    def delete(self) -> int:
        if self.deleted:
            return 0
        released = self.reserved_bytes
        self._keys.clear()
        self._values.clear()
        self.layers.clear()
        self._snapshots.clear()
        self.sequence_length = 0
        self._pending_end = None
        self.deleted = True
        return released

    @property
    def reserved_bytes(self) -> int:
        return sum(
            int(tensor.numel() * tensor.element_size()) for tensor in (*self._keys, *self._values)
        )

    @property
    def used_bytes(self) -> int:
        if self.deleted:
            return 0
        per_token = (
            2
            * (self.layer_end - self.layer_start)
            * self.batch_size
            * self.key_value_head_count
            * self.head_dimension
            * int(self._keys[0].element_size())
        )
        return per_token * self.sequence_length

    @property
    def fragmentation_bytes(self) -> int:
        return self.reserved_bytes - self.used_bytes

    @property
    def fragmentation_fraction(self) -> float:
        reserved = self.reserved_bytes
        return self.fragmentation_bytes / reserved if reserved else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "backend": "static",
            "request_ids": list(self.request_ids),
            "request_slots": dict(self.request_slots),
            "model_revision": self.model_revision,
            "stage_id": self.stage_id,
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "owned_layer_count": self.layer_end - self.layer_start,
            "route_generation": self.route_generation,
            "cache_generation": self.cache_generation,
            "sequence_length": self.sequence_length,
            "max_sequence_length": self.max_sequence_length,
            "batch_size": self.batch_size,
            "dtype": str(self.dtype).removeprefix("torch."),
            "device": str(self.device),
            "fixed_shape": self.fixed_shape,
            "allocation_count": self.allocation_count,
            "append_count": self.append_count,
            "cache_bytes": self.used_bytes,
            "reserved_bytes": self.reserved_bytes,
            "fragmentation_bytes": self.fragmentation_bytes,
            "fragmentation_fraction": self.fragmentation_fraction,
            "snapshot_count": len(self._snapshots),
            "deleted": self.deleted,
        }

    def _require_live(self) -> None:
        if self.deleted:
            raise UnsupportedCacheFormatError("static cache has been deleted")
