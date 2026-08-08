"""Open, metadata-driven model architecture contracts.

Architecture identities are derived from checkpoint metadata.  They are not a
closed product enum and they never depend on a repository name.  Concrete
family knowledge is supplied by :mod:`swarm_inference.model.architecture_adapters`
and can be extended through Python entry points without changing the resolver,
coordinator, planner, or execution engines.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from pydantic import ConfigDict, Field, NonNegativeInt, PositiveInt, model_validator

from swarm_inference.config.models import StrictModel

ArchitectureSource = Literal[
    "config.architectures",
    "config.model_type",
    "config.text_config.architectures",
    "config.text_config.model_type",
    "gguf.general.architecture",
    "tensor-layout",
    "explicit-metadata",
    "unknown",
]
DenseOrMoe = Literal["dense", "moe"]


class TensorRole(StrEnum):
    """Canonical tensor roles understood by artifact and execution code."""

    EMBEDDING = "embedding"
    ATTENTION = "attention"
    ATTENTION_NORM = "attention_norm"
    DENSE_MLP = "dense_mlp"
    ROUTER = "router"
    ROUTED_EXPERT = "routed_expert"
    SHARED_EXPERT = "shared_expert"
    ALWAYS_ON_EXPERT = "always_on_expert"
    FINAL_NORM = "final_norm"
    OUTPUT_HEAD = "output_head"
    POSITIONAL = "positional"
    MULTIMODAL = "multimodal"
    AUXILIARY = "auxiliary"
    UNKNOWN = "unknown"


class ShardReduction(StrEnum):
    """Mathematical operation used to reconstruct a sharded result."""

    CONCATENATE = "concatenate"
    SUM = "sum"
    ALL_REDUCE_SUM = "all_reduce_sum"
    GATHER = "gather"
    NONE = "none"


class TensorShardSemantics(StrictModel):
    """Adapter-proved slicing and reduction semantics for one tensor role."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tensor_role: str = Field(min_length=1)
    shard_axis: int = Field(ge=0)
    output_axis: int | None = Field(default=None, ge=0)
    reduction: ShardReduction
    alignment: PositiveInt = 1
    independently_executable: bool = True
    notes: tuple[str, ...] = ()


class TensorInterpretation(StrictModel):
    """Architecture-owned meaning of an immutable checkpoint tensor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tensor_name: str = Field(min_length=1)
    role: TensorRole
    layer_index: int | None = Field(default=None, ge=0)
    expert_index: int | None = Field(default=None, ge=0)
    expert_type: Literal["routed", "shared", "always_on", "latent", "grouped"] | None = None
    projection: str | None = None
    tensor_group: str | None = None
    shape: tuple[PositiveInt, ...] = ()
    dtype: str | None = None
    byte_size: NonNegativeInt = 0
    shard_semantics: tuple[TensorShardSemantics, ...] = ()


class TensorGroupDescriptor(StrictModel):
    """A mathematical tensor group owned by a routed-computation unit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    group_id: str = Field(min_length=1)
    tensor_names: tuple[str, ...]
    tensor_roles: tuple[str, ...]
    tensor_shapes: tuple[tuple[PositiveInt, ...], ...] = ()
    parameter_count: NonNegativeInt | None = None
    memory_bytes: NonNegativeInt = 0
    shard_semantics: tuple[TensorShardSemantics, ...] = ()

    @model_validator(mode="after")
    def names_match_roles(self) -> TensorGroupDescriptor:
        if not self.tensor_names or len(self.tensor_names) != len(self.tensor_roles):
            raise ValueError("tensor groups require matching non-empty names and roles")
        if self.tensor_shapes and len(self.tensor_shapes) != len(self.tensor_names):
            raise ValueError("tensor group shapes must align with tensor names")
        return self


class ExpertDescriptor(StrictModel):
    """Canonical representation of routed, shared, or always-on computation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    layer_index: int = Field(ge=0)
    expert_index: int = Field(ge=0)
    expert_type: Literal["routed", "shared", "always_on", "latent", "grouped"]
    tensor_groups: tuple[TensorGroupDescriptor, ...]
    parameter_count: NonNegativeInt | None = None
    memory_bytes: NonNegativeInt = 0
    input_shape: tuple[PositiveInt, ...] = ()
    output_shape: tuple[PositiveInt, ...] = ()
    routing_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def has_tensor_groups(self) -> ExpertDescriptor:
        if not self.tensor_groups:
            raise ValueError("expert descriptors require at least one tensor group")
        return self


class ModelArchitectureProfile(StrictModel):
    """Execution-neutral facts proven from a model artifact and its metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    architecture_id: str = Field(min_length=1)
    adapter_id: str = Field(min_length=1)
    dense_or_moe: DenseOrMoe
    layer_count: NonNegativeInt
    hidden_size: PositiveInt
    attention_type: str = Field(min_length=1)
    attention_metadata: dict[str, Any] = Field(default_factory=dict)
    expert_count: PositiveInt | None = None
    experts_per_token: PositiveInt | None = None
    shared_expert_count: NonNegativeInt | None = None
    expert_intermediate_size: PositiveInt | None = None
    router_type: str | None = None
    routing_metadata: dict[str, Any] = Field(default_factory=dict)
    tensor_layout: str = Field(min_length=1)
    checkpoint_format: str = Field(min_length=1)
    quantization: str | None = None
    multimodal: bool = False
    modalities: tuple[str, ...] = ("text",)
    capabilities: frozenset[str] = frozenset()
    total_parameters: NonNegativeInt | None = None
    active_parameters: NonNegativeInt | None = None
    vocab_size: PositiveInt | None = None
    configuration_source: ArchitectureSource = "unknown"
    raw_architectures: tuple[str, ...] = ()
    validation_notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def topology_is_coherent(self) -> ModelArchitectureProfile:
        if self.dense_or_moe == "moe":
            if self.expert_count is None or self.experts_per_token is None:
                raise ValueError("MoE profiles require expert_count and experts_per_token")
            if self.experts_per_token > self.expert_count:
                raise ValueError("experts_per_token cannot exceed expert_count")
        elif any(
            value is not None
            for value in (
                self.expert_count,
                self.experts_per_token,
                self.shared_expert_count,
                self.expert_intermediate_size,
            )
        ):
            raise ValueError("dense profiles cannot declare expert topology")
        if (
            self.active_parameters is not None
            and self.total_parameters is not None
            and self.active_parameters > self.total_parameters
        ):
            raise ValueError("active parameters cannot exceed total parameters")
        return self


class ArchitectureSupport(StrictModel):
    """Result of one architecture adapter probing an immutable model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    adapter_id: str
    architecture_id: str
    supported: bool
    confidence: Literal["exact_metadata", "tensor_inferred", "insufficient"]
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ArchitectureIdentity:
    canonical: str | None
    raw: str | None
    source: ArchitectureSource

    @property
    def known(self) -> bool:
        return architecture_is_known(self.canonical)


def _first_config_identity(config: dict[str, Any]) -> tuple[str | None, ArchitectureSource]:
    architectures = config.get("architectures")
    if isinstance(architectures, list) and architectures:
        return str(architectures[0]), "config.architectures"
    model_type = config.get("model_type")
    if isinstance(model_type, str) and model_type.strip():
        return model_type, "config.model_type"
    nested = config.get("text_config")
    if isinstance(nested, dict):
        architectures = nested.get("architectures")
        if isinstance(architectures, list) and architectures:
            return str(architectures[0]), "config.text_config.architectures"
        model_type = nested.get("model_type")
        if isinstance(model_type, str) and model_type.strip():
            return model_type, "config.text_config.model_type"
    return None, "unknown"


def normalize_model_architecture(
    value: str | None,
    *,
    dense_or_moe: DenseOrMoe | None = None,
) -> str | None:
    """Resolve an exact metadata identifier through installed adapters."""

    if value is None or not value.strip():
        return None
    from swarm_inference.model.architecture_adapters import (
        default_architecture_adapter_registry,
    )

    resolved = default_architecture_adapter_registry().architecture_for_identifier(
        value,
        dense_or_moe=dense_or_moe,
    )
    return resolved or value.strip()


def architecture_is_known(value: str | None) -> bool:
    if value is None or not str(value).strip():
        return False
    from swarm_inference.model.architecture_adapters import (
        default_architecture_adapter_registry,
    )

    registry = default_architecture_adapter_registry()
    return (
        registry.has_architecture(str(value))
        or registry.architecture_for_identifier(str(value)) is not None
    )


def architecture_from_config(config: dict[str, Any]) -> ArchitectureIdentity:
    raw, source = _first_config_identity(config)
    from swarm_inference.model.architecture_adapters import (
        default_architecture_adapter_registry,
    )

    adapter = default_architecture_adapter_registry().resolve_config(config)
    canonical = (
        adapter.architecture_id if adapter is not None else normalize_model_architecture(raw)
    )
    return ArchitectureIdentity(canonical, raw, source)


def architecture_from_gguf(
    raw_value: object,
    *,
    fallback: ArchitectureIdentity | None = None,
    dense_or_moe: DenseOrMoe | None = None,
) -> ArchitectureIdentity:
    if raw_value is not None and str(raw_value).strip():
        raw = str(raw_value).strip()
        return ArchitectureIdentity(
            normalize_model_architecture(raw, dense_or_moe=dense_or_moe),
            raw,
            "gguf.general.architecture",
        )
    return fallback or ArchitectureIdentity(None, None, "unknown")


def gguf_identifiers_for(value: str | None) -> tuple[str, ...]:
    if value is None or not str(value).strip():
        return ()
    from swarm_inference.model.architecture_adapters import (
        default_architecture_adapter_registry,
    )

    registry = default_architecture_adapter_registry()
    adapter = registry.get_by_architecture(str(value))
    if adapter is None:
        architecture = registry.architecture_for_identifier(str(value))
        adapter = registry.get_by_architecture(architecture) if architecture else None
    return adapter.gguf_architectures if adapter is not None else (str(value).strip(),)


class _GgufArchitectureMapping(Mapping[str, tuple[str, ...]]):
    """Backward-compatible, registry-generated view of GGUF capabilities."""

    @staticmethod
    def _data() -> dict[str, tuple[str, ...]]:
        from swarm_inference.model.architecture_adapters import (
            default_architecture_adapter_registry,
        )

        return default_architecture_adapter_registry().gguf_architecture_mapping()

    def __getitem__(self, key: str) -> tuple[str, ...]:
        return self._data()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data())

    def __len__(self) -> int:
        return len(self._data())


GGUF_ARCHITECTURE_IDENTIFIERS: Mapping[str, tuple[str, ...]] = _GgufArchitectureMapping()


__all__ = [
    "GGUF_ARCHITECTURE_IDENTIFIERS",
    "ArchitectureIdentity",
    "ArchitectureSource",
    "ArchitectureSupport",
    "DenseOrMoe",
    "ExpertDescriptor",
    "ModelArchitectureProfile",
    "ShardReduction",
    "TensorGroupDescriptor",
    "TensorInterpretation",
    "TensorRole",
    "TensorShardSemantics",
    "architecture_from_config",
    "architecture_from_gguf",
    "architecture_is_known",
    "gguf_identifiers_for",
    "normalize_model_architecture",
]
