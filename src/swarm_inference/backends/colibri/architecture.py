"""Architecture-neutral contracts for the pinned Colibri execution backend.

The coordinator only sees structured engine support and execution plans.  All
checkpoint, router, expert, prompt, and launch semantics live behind the
adapter protocol in this module.
"""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic import ConfigDict, Field, NonNegativeInt, PositiveInt, model_validator

from swarm_inference.config.models import StrictModel
from swarm_inference.model.architecture import ExpertDescriptor
from swarm_inference.model.descriptor import ResolvedModelDescriptor

if TYPE_CHECKING:
    from swarm_inference.engines.interfaces import ClusterCapabilities


class ColibriCompatibilityStatus(StrEnum):
    """Public compatibility classifications used by the support matrix."""

    SUPPORTED = "SUPPORTED"
    SUPPORTED_WITH_LIMITATIONS = "SUPPORTED_WITH_LIMITATIONS"
    UPSTREAM_SUPPORTED_NOT_YET_INTEGRATED = "UPSTREAM_SUPPORTED_NOT_YET_INTEGRATED"
    UNSUPPORTED_BY_PINNED_COLIBRI = "UNSUPPORTED_BY_PINNED_COLIBRI"
    INCOMPATIBLE_FORMAT = "INCOMPATIBLE_FORMAT"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ColibriRuntimeCapabilities(StrictModel):
    """Facts advertised by one hash-verified Colibri worker runtime."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    installed: bool
    runtime_version: str | None = None
    binary_hashes: dict[str, str] = Field(default_factory=dict)
    adapters: tuple[str, ...] = ()
    formats: tuple[str, ...] = ()
    quantizations: tuple[str, ...] = ()
    device_types: tuple[str, ...] = ()
    features: tuple[str, ...] = ()


class ColibriSupportResult(StrictModel):
    """Structured adapter-level support proof; never a model-name guess."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    supported: bool
    engine: Literal["colibri"] = "colibri"
    classification: ColibriCompatibilityStatus
    architecture: str | None = None
    adapter_id: str | None = None
    reasons: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    runtime_version: str | None = None
    model_format_supported: bool
    architecture_supported: bool
    quantization_supported: bool
    tokenizer_supported: bool
    cluster_supported: bool = True
    runtime_supported: bool
    required_features: tuple[str, ...] = ()
    unsupported_features: tuple[str, ...] = ()

    @model_validator(mode="after")
    def support_matches_classification(self) -> ColibriSupportResult:
        compatible = self.classification in {
            ColibriCompatibilityStatus.SUPPORTED,
            ColibriCompatibilityStatus.SUPPORTED_WITH_LIMITATIONS,
        }
        if self.supported != compatible:
            raise ValueError("Colibri support boolean and classification disagree")
        if not self.reasons:
            raise ValueError("Colibri support results require an explicit reason")
        return self


class ColibriTensorMapping(StrictModel):
    """Adapter-owned interpretation of one checkpoint tensor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tensor_name: str = Field(min_length=1)
    tensor_role: str = Field(min_length=1)
    layer_index: int = Field(default=-1, ge=-1)
    expert_index: int | None = Field(default=None, ge=0)
    projection: str | None = None
    shard_axis: int | None = Field(default=None, ge=0)
    logical_shape: tuple[PositiveInt, ...] = ()
    packed_shape: tuple[PositiveInt, ...] = ()
    quantization_format: str
    packing: str
    scale_format: str
    scale_group_size: int | None = Field(default=None, gt=0)
    quantization_aware_trained: bool = False
    reencoding_allowed: bool = False
    byte_size: NonNegativeInt = 0


class ColibriRoutingDescriptor(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    routing_kind: str
    expert_count: PositiveInt
    experts_per_token: PositiveInt
    shared_expert_count: NonNegativeInt = 0
    normalization: str
    score_correction: str | None = None
    routed_weight_semantics: str

    @model_validator(mode="after")
    def top_k_fits(self) -> ColibriRoutingDescriptor:
        if self.experts_per_token > self.expert_count:
            raise ValueError("experts_per_token cannot exceed expert_count")
        return self


class ColibriModelProfile(StrictModel):
    """Validated architecture facts required by a concrete Colibri binary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    adapter_id: str
    architecture: str
    architecture_raw: str | None = None
    checkpoint_format: str
    checkpoint_layout: str
    quantization: str | None = None
    layer_count: PositiveInt | None = None
    hidden_size: PositiveInt | None = None
    expert_count: PositiveInt | None = None
    experts_per_token: PositiveInt | None = None
    shared_expert_count: NonNegativeInt = 0
    expert_intermediate_size: PositiveInt | None = None
    routing_kind: str
    architecture_metadata: dict[str, Any] = Field(default_factory=dict)
    required_config_fields: tuple[str, ...] = ()
    tokenizer_mode: Literal["local-token-ids", "gateway-text"]
    launch_mode: Literal["one-shot", "persistent-gateway"]
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def topology_reconciles(self) -> ColibriModelProfile:
        if (
            self.expert_count is not None
            and self.experts_per_token is not None
            and self.experts_per_token > self.expert_count
        ):
            raise ValueError("experts_per_token cannot exceed expert_count")
        return self


class ColibriExecutionProfile(StrictModel):
    """Adapter-provided execution facts consumed by generic plan generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    adapter_id: str
    engine_basename: str
    complete_model: bool = True
    component_capabilities: tuple[str, ...] = ()
    gateway_architecture: str | None = None
    topology: str
    routing_mode: str
    required_memory_bytes: NonNegativeInt
    expected_expert_movement_bytes: NonNegativeInt | None = None
    persistent_worker: bool
    persistent_expert_residency: bool
    supports_streaming: bool
    supports_exact_replay: bool
    whole_expert_execution: bool
    tensor_microshards: bool
    direct_peer_model_data: bool
    coordinator_activation_relay: Literal[False] = False
    cache_policy: str
    required_features: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()


class ColibriReplayInvocation(StrictModel):
    """Architecture-owned command contract for fixed replay or generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    command: tuple[str, ...]
    environment: dict[str, str] = Field(default_factory=dict)
    exact_replay: bool


@runtime_checkable
class ColibriArchitectureAdapter(Protocol):
    """Mandatory architecture boundary for every executable Colibri family."""

    adapter_id: str
    adapter_version: str
    claimed_architectures: tuple[str, ...]
    engine_basename: str
    gateway_architecture: str | None
    tokenizer_mode: str
    launch_mode: str
    complete_model: bool
    component_capabilities: tuple[str, ...]
    native_quantization: str
    exact_replay: bool
    tensor_microshards: bool
    tuning_settings: tuple[str, ...]
    supports_text_calibration: bool

    def validate_contract(self) -> None: ...

    def matches_config(self, config: dict[str, Any]) -> bool: ...

    def supports(
        self,
        model: ResolvedModelDescriptor,
        runtime: ColibriRuntimeCapabilities,
    ) -> ColibriSupportResult: ...

    def inspect_model(self, model: ResolvedModelDescriptor) -> ColibriModelProfile: ...

    def build_execution_profile(
        self,
        model: ResolvedModelDescriptor,
        cluster: ClusterCapabilities,
    ) -> ColibriExecutionProfile: ...

    def validate_model_identity(
        self,
        model: ResolvedModelDescriptor,
        *,
        tensor_names: tuple[str, ...] = (),
    ) -> None: ...

    def map_tensor_names(
        self,
        tensor_names: tuple[tuple[str, tuple[int, ...], str, int], ...],
        *,
        config: dict[str, Any],
    ) -> tuple[ColibriTensorMapping, ...]: ...

    def describe_experts(
        self,
        tensors: tuple[ColibriTensorMapping, ...],
        profile: ColibriModelProfile,
    ) -> tuple[ExpertDescriptor, ...]: ...

    def describe_routing(self, profile: ColibriModelProfile) -> ColibriRoutingDescriptor: ...

    def validate_execution_result(
        self,
        *,
        output_token_ids: tuple[int, ...],
        requested_tokens: int,
    ) -> None: ...

    def validate_generation_request(
        self,
        *,
        stream: bool,
        temperature: float,
        top_p: float,
        has_token_ids: bool,
    ) -> None: ...

    def calibration_invocation(
        self,
        *,
        engine: Path,
        model_path: Path,
        cap: int,
        prompt: str,
        continuation_tokens: int,
    ) -> ColibriReplayInvocation: ...

    def replay_invocation(
        self,
        *,
        engine: Path,
        model_path: Path,
        cap: int,
        quant_bits: int,
        reference: Path,
        prompt_ids: tuple[int, ...],
        completion_tokens: int,
        teacher_forced: bool,
    ) -> ColibriReplayInvocation: ...


def architecture_metadata_values(config: dict[str, Any]) -> tuple[str, ...]:
    """Return exact architecture metadata values, never repository-name text."""

    values: list[str] = []
    text_config = config.get("text_config")
    owners = (config, text_config) if isinstance(text_config, dict) else (config,)
    for owner in owners:
        architectures = owner.get("architectures")
        if isinstance(architectures, list):
            values.extend(str(item) for item in architectures if str(item).strip())
        model_type = owner.get("model_type")
        if isinstance(model_type, str) and model_type.strip():
            values.append(model_type)
    return tuple(dict.fromkeys(values))


def _owner_metadata_values(owner: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    architectures = owner.get("architectures")
    if isinstance(architectures, list):
        values.extend(str(item) for item in architectures if str(item).strip())
    model_type = owner.get("model_type")
    if isinstance(model_type, str) and model_type.strip():
        values.append(model_type)
    return tuple(dict.fromkeys(values))


def architecture_claim_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


class ColibriArchitectureAdapterRegistry:
    """Deterministic, fail-closed registry for architecture metadata claims."""

    def __init__(self, adapters: tuple[ColibriArchitectureAdapter, ...] = ()) -> None:
        self._adapters: dict[str, ColibriArchitectureAdapter] = {}
        self._claims: dict[str, str] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: ColibriArchitectureAdapter) -> None:
        if not isinstance(adapter, ColibriArchitectureAdapter):
            raise TypeError("Colibri adapter does not satisfy ColibriArchitectureAdapter")
        adapter.validate_contract()
        adapter_id = adapter.adapter_id.strip()
        if not adapter_id:
            raise ValueError("Colibri adapter ID cannot be empty")
        if adapter_id in self._adapters:
            raise ValueError(f"Colibri adapter {adapter_id!r} is already registered")
        claims: list[str] = []
        for architecture in adapter.claimed_architectures:
            key = architecture_claim_key(architecture)
            if not key:
                raise ValueError(f"Colibri adapter {adapter_id!r} has an empty architecture claim")
            owner = self._claims.get(key)
            if owner is not None:
                raise ValueError(
                    f"Colibri architecture claim {architecture!r} is duplicated by "
                    f"{owner!r} and {adapter_id!r}"
                )
            claims.append(key)
        if not claims:
            raise ValueError(f"Colibri adapter {adapter_id!r} claims no architecture")
        self._adapters[adapter_id] = adapter
        self._claims.update(dict.fromkeys(claims, adapter_id))

    def adapters(self) -> tuple[ColibriArchitectureAdapter, ...]:
        return tuple(self._adapters[key] for key in sorted(self._adapters))

    def get(self, adapter_id: str) -> ColibriArchitectureAdapter:
        try:
            return self._adapters[adapter_id]
        except KeyError as exc:
            raise KeyError(f"Colibri adapter {adapter_id!r} is not registered") from exc

    def _metadata_matches(self, values: tuple[str, ...]) -> tuple[ColibriArchitectureAdapter, ...]:
        identifiers = {
            self._claims[key]
            for value in values
            if (key := architecture_claim_key(value)) in self._claims
        }
        return tuple(self._adapters[key] for key in sorted(identifiers))

    def resolve_model(self, model: ResolvedModelDescriptor) -> ColibriArchitectureAdapter | None:
        # The execution-neutral architecture profile has already reconciled
        # wrapper and nested text metadata.  Prefer that canonical identity so
        # a Kimi wrapper whose text core reuses DeepSeek metadata is not
        # misclassified as a standalone DeepSeek checkpoint.
        matches = self._metadata_matches((model.architecture,)) if model.architecture else ()
        if not matches:
            values = tuple(
                value
                for value in (
                    model.architecture_raw,
                    *architecture_metadata_values(model.configuration),
                )
                if value
            )
            matches = self._metadata_matches(values)
        if not matches:
            return None
        if len(matches) > 1:
            raise LookupError(
                "Colibri architecture metadata is ambiguous: "
                + ", ".join(item.adapter_id for item in matches)
            )
        adapter = matches[0]
        # A canonical architecture profile is the execution-neutral resolver's
        # validated result.  GGUF inspection deliberately retains only bounded
        # metadata and may not include every Safetensors layout field required
        # by this adapter.  Re-validating that partial dictionary here would
        # conflate artifact format with architecture identity.
        if (
            model.architecture_profile is None
            and model.configuration
            and not adapter.matches_config(model.configuration)
        ):
            raise LookupError(
                f"Colibri adapter {adapter.adapter_id!r} architecture claim does not satisfy "
                "its configuration validation contract"
            )
        return adapter

    def resolve_config(
        self,
        config: dict[str, Any],
        *,
        explicit_adapter_id: str | None = None,
    ) -> ColibriArchitectureAdapter:
        root_matches = self._metadata_matches(_owner_metadata_values(config))
        nested = config.get("text_config")
        nested_matches = (
            self._metadata_matches(_owner_metadata_values(nested))
            if isinstance(nested, dict)
            else ()
        )
        # Root metadata describes the checkpoint wrapper and has precedence.
        # Nested metadata is consulted only when the wrapper is generic or
        # absent.  This keeps architecture reuse composable without ambiguity.
        metadata_matches = root_matches or nested_matches
        matches = tuple(item for item in metadata_matches if item.matches_config(config))
        if explicit_adapter_id is not None:
            explicit = self.get(explicit_adapter_id)
            if explicit not in matches:
                detected = ", ".join(item.adapter_id for item in matches) or "none"
                raise ValueError(
                    f"requested Colibri adapter {explicit_adapter_id!r} does not match "
                    f"validated architecture metadata; detected={detected}"
                )
            return explicit
        if not matches:
            values = architecture_metadata_values(config)
            detail = ", ".join(values) if values else "missing"
            raise LookupError("no Colibri architecture adapter validates config metadata " + detail)
        if len(matches) > 1:
            raise LookupError(
                "Colibri config matches multiple adapters: "
                + ", ".join(item.adapter_id for item in matches)
            )
        return matches[0]


__all__ = [
    "ColibriArchitectureAdapter",
    "ColibriArchitectureAdapterRegistry",
    "ColibriCompatibilityStatus",
    "ColibriExecutionProfile",
    "ColibriModelProfile",
    "ColibriReplayInvocation",
    "ColibriRoutingDescriptor",
    "ColibriRuntimeCapabilities",
    "ColibriSupportResult",
    "ColibriTensorMapping",
    "ExpertDescriptor",
    "architecture_claim_key",
    "architecture_metadata_values",
]
