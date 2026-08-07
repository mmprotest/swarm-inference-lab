"""Backend-neutral, native-quantization-preserving expert microshard ABI."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Literal, cast

import numpy as np
from pydantic import Field, model_validator

from swarm_inference.backends.colibri.schemas import NativeQuantizationMetadata
from swarm_inference.config.models import StrictModel
from swarm_inference.model.architecture import ShardReduction


class ExpertProjectionSlice(StrictModel):
    """One adapter-described logical tensor slice; never an arbitrary byte interval."""

    tensor_id: str
    tensor_name: str
    projection: str = Field(min_length=1)
    logical_axis: int = Field(ge=0, le=7)
    reduction: ShardReduction | None = None
    semantic_group: str = "expert-mlp"
    slice_start: int = Field(ge=0)
    slice_end: int = Field(gt=0)
    logical_shape: list[int] = Field(min_length=1, max_length=8)
    storage_file: str
    storage_offset: int = Field(ge=0)
    storage_length: int = Field(gt=0)
    storage_file_size: int = Field(gt=0)
    storage_segments: list[dict[str, int]] = Field(default_factory=list)
    content_hash: str

    @model_validator(mode="after")
    def validate_projection(self) -> ExpertProjectionSlice:
        if self.slice_end <= self.slice_start:
            raise ValueError("projection slice must be non-empty")
        expected_axis = {"up": 0, "gate": 0, "down": 1}.get(self.projection)
        if expected_axis is not None and self.logical_axis != expected_axis:
            raise ValueError("legacy up/gate must slice rows and down must slice columns")
        if any(dimension <= 0 for dimension in self.logical_shape):
            raise ValueError("projection shapes must be positive")
        if self.slice_end > self.logical_shape[self.logical_axis]:
            raise ValueError("projection range exceeds its logical shape")
        if self.storage_offset + self.storage_length > self.storage_file_size:
            raise ValueError("projection storage range is out of bounds")
        if self.storage_segments:
            total = 0
            starts = []
            for segment in self.storage_segments:
                if set(segment) != {"offset", "length"}:
                    raise ValueError("storage segments require only offset and length")
                offset, length = segment["offset"], segment["length"]
                if offset < 0 or length <= 0 or offset + length > self.storage_file_size:
                    raise ValueError("projection storage segment is out of bounds")
                starts.append(offset)
                total += length
            if total != self.storage_length or min(starts) != self.storage_offset:
                raise ValueError("projection storage segments do not reconcile to their summary")
        if not self.content_hash:
            raise ValueError("projection content hash is required")
        return self


class RoutedComputationMicroshardDescriptor(StrictModel):
    """Architecture-neutral matched tensor slices for one routed computation."""

    model_id: str
    architecture_id: str
    layer_id: int = Field(ge=0)
    routed_unit_id: int = Field(ge=0)
    routed_unit_type: Literal["routed", "shared", "always_on", "latent", "grouped"]
    shard_id: str
    shard_start: int = Field(ge=0)
    shard_end: int = Field(gt=0)
    logical_shard_extent: int = Field(gt=0)
    tensor_slices: tuple[ExpertProjectionSlice, ...]
    native_quantization: NativeQuantizationMetadata
    reduction: ShardReduction
    required_accumulator: str
    supported_backends: tuple[str, ...] = ()
    execution_status: Literal["supported", "unsupported"] = "unsupported"
    content_hash: str = ""

    @model_validator(mode="after")
    def validate_adapter_slices(self) -> RoutedComputationMicroshardDescriptor:
        if self.shard_end <= self.shard_start or self.shard_end > self.logical_shard_extent:
            raise ValueError("routed-computation shard range is invalid")
        if not self.tensor_slices:
            raise ValueError("routed-computation microshards require tensor slices")
        identities = {(item.tensor_id, item.tensor_name) for item in self.tensor_slices}
        if len(identities) != len(self.tensor_slices):
            raise ValueError("routed-computation microshard repeats a tensor slice")
        if any(
            item.slice_start != self.shard_start or item.slice_end != self.shard_end
            for item in self.tensor_slices
        ):
            raise ValueError("adapter tensor slices do not preserve the logical shard range")
        group = self.native_quantization.scale_group_size
        if group is not None:
            if self.shard_start % group:
                raise ValueError("microshard start splits a native quantization group")
            if self.shard_end != self.logical_shard_extent and self.shard_end % group:
                raise ValueError("microshard end splits a native quantization group")
        expected = routed_descriptor_content_hash(self)
        if self.content_hash and self.content_hash != expected:
            raise ValueError("routed-computation descriptor content hash is invalid")
        object.__setattr__(self, "content_hash", expected)
        if self.execution_status == "supported" and not self.supported_backends:
            raise ValueError("supported microshards require an executable backend")
        return self


def routed_descriptor_content_hash(
    descriptor: RoutedComputationMicroshardDescriptor,
) -> str:
    payload = descriptor.model_dump(mode="json", exclude={"content_hash"})
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def validate_routed_microshard_set(
    descriptors: Iterable[RoutedComputationMicroshardDescriptor],
) -> dict[str, int | bool | str]:
    """Prove an adapter-described set is a complete, non-overlapping routed unit."""

    shards = sorted(descriptors, key=lambda item: (item.shard_start, item.shard_end))
    if not shards:
        raise ValueError("at least one routed-computation microshard is required")
    identity = {
        (item.model_id, item.architecture_id, item.layer_id, item.routed_unit_id) for item in shards
    }
    if len(identity) != 1:
        raise ValueError("microshard set contains more than one routed-computation unit")
    if len({item.shard_id for item in shards}) != len(shards):
        raise ValueError("microshard IDs must be unique")
    if len({item.logical_shard_extent for item in shards}) != 1:
        raise ValueError("microshards disagree about their logical extent")
    cursor = 0
    for shard in shards:
        if shard.shard_start != cursor:
            kind = "overlap" if shard.shard_start < cursor else "gap"
            raise ValueError(f"routed-computation microshards contain a {kind} at {cursor}")
        if shard.content_hash != routed_descriptor_content_hash(shard):
            raise ValueError("microshard content hash changed after validation")
        cursor = shard.shard_end
    extent = shards[0].logical_shard_extent
    if cursor != extent:
        raise ValueError("microshards do not cover the complete routed computation")
    return {
        "valid": True,
        "architecture_id": shards[0].architecture_id,
        "layer_id": shards[0].layer_id,
        "routed_unit_id": shards[0].routed_unit_id,
        "shard_count": len(shards),
        "covered_units": extent,
    }


class ExpertMicroshardDescriptor(StrictModel):
    """Compatibility ABI for the promoted three-projection SiLU expert kernel."""

    model_id: str
    layer_id: int = Field(ge=0)
    expert_id: int = Field(ge=0)
    shard_id: str
    hidden_start: int = Field(ge=0)
    hidden_end: int = Field(gt=0)
    up_projection: ExpertProjectionSlice
    gate_projection: ExpertProjectionSlice
    down_projection: ExpertProjectionSlice
    native_quantization: NativeQuantizationMetadata
    content_hash: str = ""
    required_accumulator: str
    supported_backends: list[str] = Field(default_factory=list)
    execution_status: Literal["supported", "unsupported"] = "unsupported"

    @model_validator(mode="after")
    def validate_atomic_slice(self) -> ExpertMicroshardDescriptor:
        if self.hidden_end <= self.hidden_start:
            raise ValueError("microshard range must be non-empty")
        slices = (self.up_projection, self.gate_projection, self.down_projection)
        if tuple(item.projection for item in slices) != ("up", "gate", "down"):
            raise ValueError("microshard requires matching up, gate, and down projections")
        if any(
            item.slice_start != self.hidden_start or item.slice_end != self.hidden_end
            for item in slices
        ):
            raise ValueError("all expert projections must preserve the same hidden range")
        intermediate_sizes = (
            self.up_projection.logical_shape[0],
            self.gate_projection.logical_shape[0],
            self.down_projection.logical_shape[1],
        )
        if len(set(intermediate_sizes)) != 1:
            raise ValueError("expert projection intermediate dimensions do not reconcile")
        hidden_sizes = (
            self.up_projection.logical_shape[1],
            self.gate_projection.logical_shape[1],
            self.down_projection.logical_shape[0],
        )
        if len(set(hidden_sizes)) != 1:
            raise ValueError("expert projection input/output hidden dimensions do not reconcile")
        group = self.native_quantization.scale_group_size
        if group is not None:
            intermediate = intermediate_sizes[0]
            if self.hidden_start % group:
                raise ValueError("microshard start splits a native quantization group")
            if self.hidden_end != intermediate and self.hidden_end % group:
                raise ValueError("microshard end splits a native quantization group")
        expected = descriptor_content_hash(self)
        if self.content_hash and self.content_hash != expected:
            raise ValueError("microshard descriptor content hash is unstable or incorrect")
        object.__setattr__(self, "content_hash", expected)
        if self.execution_status == "supported" and not self.supported_backends:
            raise ValueError("supported microshards require at least one executable backend")
        return self

    @property
    def intermediate_size(self) -> int:
        return self.up_projection.logical_shape[0]


def descriptor_content_hash(descriptor: ExpertMicroshardDescriptor) -> str:
    payload = descriptor.model_dump(mode="json", exclude={"content_hash"})
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def validate_expert_microshard_set(
    descriptors: Iterable[ExpertMicroshardDescriptor],
) -> dict[str, int | bool | str]:
    """Prove a set exactly reconstructs one expert without gaps or overlaps."""

    shards = sorted(descriptors, key=lambda item: (item.hidden_start, item.hidden_end))
    if not shards:
        raise ValueError("at least one expert microshard is required")
    identity = {(item.model_id, item.layer_id, item.expert_id) for item in shards}
    if len(identity) != 1:
        raise ValueError("microshard set contains more than one expert")
    if len({item.shard_id for item in shards}) != len(shards):
        raise ValueError("microshard IDs must be unique")
    cursor = 0
    for shard in shards:
        if shard.hidden_start < cursor:
            raise ValueError("expert microshard ranges overlap")
        if shard.hidden_start > cursor:
            raise ValueError("expert microshard ranges contain a gap")
        if shard.content_hash != descriptor_content_hash(shard):
            raise ValueError("microshard content hash changed after validation")
        cursor = shard.hidden_end
    expected = shards[0].intermediate_size
    if cursor != expected:
        raise ValueError(f"microshards cover 0:{cursor}, not the complete 0:{expected} expert")
    return {
        "valid": True,
        "model_id": shards[0].model_id,
        "layer_id": shards[0].layer_id,
        "expert_id": shards[0].expert_id,
        "shard_count": len(shards),
        "covered_hidden_units": cursor,
    }


def executable_microshard_equivalence(
    *,
    inputs: np.ndarray,
    up: np.ndarray,
    gate: np.ndarray,
    down: np.ndarray,
    ranges: Iterable[tuple[int, int]],
) -> dict[str, float | bool]:
    """Execute a small native logical slice fixture and compare reconstruction."""

    def silu(values: np.ndarray) -> np.ndarray:
        # The algebraic definition overflows for large negative float32
        # activations even though SiLU itself is finite.  Evaluate the sigmoid
        # by sign so this correctness fixture tests shard reconstruction, not
        # an avoidable exp() warning.
        sigmoid = np.empty_like(values)
        positive = values >= 0
        sigmoid[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
        negative_exp = np.exp(values[~positive])
        sigmoid[~positive] = negative_exp / (1.0 + negative_exp)
        return cast(np.ndarray, values * sigmoid)

    if up.ndim != 2 or gate.shape != up.shape or down.shape != (up.shape[1], up.shape[0]):
        raise ValueError("expected up/gate [intermediate, hidden] and down [hidden, intermediate]")
    intervals = sorted(ranges)
    cursor = 0
    for start, end in intervals:
        if start != cursor or end <= start:
            raise ValueError("fixture ranges must be complete, ordered, and non-overlapping")
        cursor = end
    if cursor != up.shape[0]:
        raise ValueError("fixture ranges do not reconstruct the intermediate dimension")

    up_full = inputs @ up.T
    gate_full = inputs @ gate.T
    silu_full = silu(gate_full)
    reference = (silu_full * up_full) @ down.T
    partials = []
    for start, end in intervals:
        up_part = inputs @ up[start:end].T
        gate_part = inputs @ gate[start:end].T
        silu_part = silu(gate_part)
        partials.append((silu_part * up_part) @ down[:, start:end].T)
    reconstructed = np.sum(np.stack(partials), axis=0)
    difference = np.abs(reference - reconstructed)
    return {
        "allclose": bool(np.allclose(reference, reconstructed, atol=1e-5, rtol=1e-5)),
        "maximum_absolute_error": float(difference.max(initial=0.0)),
        "mean_absolute_error": float(difference.mean()),
    }
