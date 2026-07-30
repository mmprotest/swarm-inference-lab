"""Deterministic executable synthetic stages and canaries."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from swarm_inference.config.models import OperationKind, StageDefinition, SyntheticModelConfig
from swarm_inference.exceptions import ConfigurationError
from swarm_inference.protocol.checksums import sha256_bytes


@dataclass(slots=True)
class SyntheticCacheState:
    cache_generation: int
    last_token_position: int
    digest: bytes
    tokens: int


def deterministic_cpu_kernel(
    payload: bytes,
    *,
    seed_material: bytes,
    work_units: int,
    buffer_bytes: int = 16 * 1024,
) -> bytes:
    """Run deterministic, single-threaded CPU work and return its digest.

    Each work unit hashes a fixed-size buffer and the previous digest. The
    dependency chain prevents the interpreter or hashing library from
    precomputing or parallelising units, and unlike a sleep it consumes actual
    CPU time. ``hashlib`` delegates one call at a time to the platform crypto
    implementation and does not create a worker thread pool.
    """

    if work_units < 0:
        raise ValueError("work_units cannot be negative")
    if buffer_bytes <= 0:
        raise ValueError("buffer_bytes must be positive")
    seed = hashlib.blake2b(
        payload + seed_material,
        digest_size=64,
        person=b"swarm-cpu-v1",
    ).digest()
    repeats = math.ceil(buffer_bytes / len(seed))
    block = (seed * repeats)[:buffer_bytes]
    digest = seed
    for unit in range(work_units):
        transform = hashlib.blake2b(digest_size=64, person=b"swarm-cpu-v1")
        transform.update(block)
        transform.update(digest)
        transform.update(unit.to_bytes(8, "little", signed=False))
        digest = transform.digest()
    return digest


def synthetic_activation(
    token_ids: list[int],
    *,
    hidden_size: int,
    dtype: str,
) -> np.ndarray:
    """Create deterministic synthetic embeddings without model-specific prompts."""

    if not token_ids:
        raise ValueError("token_ids cannot be empty")
    rows = np.arange(hidden_size, dtype=np.uint64)[None, :]
    tokens = np.asarray(token_ids, dtype=np.uint64)[:, None]
    values = ((tokens * 1_103_515_245 + rows * 12_345 + 97) % 65_521) / 65_521.0
    return values.astype(np.dtype(dtype))[None, :, :]


class SyntheticStageModule:
    """Exact deterministic stage whose decode path depends on replayable cache."""

    def __init__(
        self,
        *,
        config: SyntheticModelConfig,
        stage: StageDefinition,
        corrupt: bool = False,
    ) -> None:
        self.config = config
        self.stage = stage
        self.stage_id = stage.stage_id
        self.required_memory_bytes = stage.required_memory_bytes
        self.corrupt = corrupt
        self._cache: dict[tuple[str, int], SyntheticCacheState] = {}

    def _layer_transform(
        self,
        values: np.ndarray,
        *,
        layer_id: int,
        token_position: int,
        cache_digest: bytes,
    ) -> np.ndarray:
        cache_term = int.from_bytes(cache_digest[:4], "little") % 1009
        scalar = (
            (self.config.model_seed * 131) + (layer_id * 17) + (token_position * 29) + cache_term
        ) % 4093
        working = values.astype(np.float32, copy=False)
        transformed = working * np.float32(1.0 + (scalar % 31) / 4096.0) + np.float32(
            (scalar % 101) / 8192.0
        )
        if transformed.shape[-1] > 1:
            shift = (layer_id + token_position + self.config.model_seed) % transformed.shape[-1]
            transformed = np.roll(transformed, shift=shift, axis=-1)
        return transformed.astype(values.dtype, copy=False)

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
    ) -> np.ndarray:
        if activation.ndim < 2:
            raise ConfigurationError("synthetic activation must have rank >= 2")
        if activation.shape[-1] != self.config.hidden_size:
            raise ConfigurationError(
                f"activation hidden size {activation.shape[-1]} does not match "
                f"{self.config.hidden_size}"
            )
        key = (request_id, cache_generation)
        existing = self._cache.get(key)
        if operation == OperationKind.PREFILL:
            if token_position != 0:
                raise ConfigurationError("prefill token_position must be zero")
            prior_digest = b"\0" * 32
        else:
            if existing is None:
                raise ConfigurationError(
                    f"missing cache for request={request_id} generation={cache_generation}"
                )
            if token_position != existing.last_token_position + 1:
                raise ConfigurationError(
                    f"non-contiguous token position for request={request_id}: "
                    f"expected {existing.last_token_position + 1}, got {token_position}"
                )
            prior_digest = existing.digest
        cpu_digest = deterministic_cpu_kernel(
            np.ascontiguousarray(activation).tobytes(),
            seed_material=b"|".join(
                (
                    str(self.config.model_seed).encode(),
                    str(self.stage_id).encode(),
                    f"{self.stage.layer_start}:{self.stage.layer_end}".encode(),
                    request_id.encode(),
                    str(token_position).encode(),
                    str(cache_generation).encode(),
                )
            ),
            work_units=self.config.cpu_work_units,
            buffer_bytes=self.config.cpu_kernel_buffer_bytes,
        )
        transform_digest = hashlib.sha256(prior_digest + cpu_digest).digest()
        output = np.ascontiguousarray(activation)
        for layer_id in range(self.stage.layer_start, self.stage.layer_end):
            output = self._layer_transform(
                output,
                layer_id=layer_id,
                token_position=token_position,
                cache_digest=transform_digest,
            )
        if self.corrupt and output.size:
            output = output.copy()
            flat = output.reshape(-1)
            flat[0] = flat[0] + np.asarray(1, dtype=output.dtype)
        digest = hashlib.sha256(
            prior_digest
            + np.ascontiguousarray(output).tobytes()
            + token_position.to_bytes(8, "little", signed=False)
        ).digest()
        consumed_tokens = sequence_length if operation == OperationKind.PREFILL else 1
        self._cache[key] = SyntheticCacheState(
            cache_generation=cache_generation,
            last_token_position=token_position,
            digest=digest,
            tokens=(existing.tokens if existing else 0) + consumed_tokens,
        )
        return output

    def cancel(self, request_id: str) -> None:
        for key in [key for key in self._cache if key[0] == request_id]:
            self._cache.pop(key, None)

    def cache_bytes(self) -> int:
        return sum(
            state.tokens * self.stage.cache_spec.bytes_per_token for state in self._cache.values()
        )

    def state_summary(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "request_states": len(self._cache),
            "cache_bytes": self.cache_bytes(),
            "corrupt": self.corrupt,
            "state_checksums": {
                f"{request_id}:{generation}": state.digest.hex()
                for (request_id, generation), state in sorted(self._cache.items())
            },
        }

    def canary(
        self,
        *,
        request_id: str = "synthetic-canary",
        cache_generation: int = 0,
    ) -> str:
        activation = synthetic_activation(
            [1, 7, 13],
            hidden_size=self.config.hidden_size,
            dtype=self.config.activation_dtype,
        )
        output = self.execute(
            activation,
            request_id=request_id,
            operation=OperationKind.PREFILL,
            token_position=0,
            sequence_length=3,
            cache_generation=cache_generation,
        )
        self.cancel(request_id)
        return sha256_bytes(np.ascontiguousarray(output).tobytes())
