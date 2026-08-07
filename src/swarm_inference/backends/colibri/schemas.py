"""Strict schemas for Colibri capabilities, inventories, plans, and telemetry."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from swarm_inference.backends.colibri.constants import (
    BRIDGE_EVENT_TYPES,
    COLIBRI_COMMIT,
    COLIBRI_RELEASE,
)
from swarm_inference.config.models import StrictModel


class ColibriMode(StrEnum):
    STOCK = "stock"
    BRIDGE = "bridge"


class TelemetryLevel(StrEnum):
    OFF = "off"
    SUMMARY = "summary"
    DETAILED = "detailed"
    TRACE = "trace"


class BridgeEvent(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    event_type: str
    timestamp_ns: int = Field(ge=0)
    engine_pid: int = Field(ge=0)
    request_id: str | None = None
    model_id: str
    model_revision: str
    engine_family: str
    sequence_number: int = Field(ge=0)
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_type")
    @classmethod
    def known_event_type(cls, value: str) -> str:
        if value not in BRIDGE_EVENT_TYPES:
            raise ValueError(f"unknown Colibri bridge event type {value!r}")
        return value


class ColibriCapabilityReport(StrictModel):
    backend: Literal["colibri"] = "colibri"
    colibri_version: str = COLIBRI_RELEASE
    colibri_commit: str = COLIBRI_COMMIT
    bridge_version: str
    platform: str
    architecture: str
    model_families: list[str]
    execution_backends: list[str]
    quantization_formats: list[str]
    supports_cpu: bool
    supports_cuda: bool
    supports_vulkan: bool
    supports_metal: bool
    supports_multi_gpu: bool
    supports_expert_residency: bool
    supports_route_trace: bool
    supports_usage_history: bool
    supports_expert_prefetch: bool
    supports_dynamic_reconfiguration: bool
    supports_native_mxfp4: bool
    supports_tensor_microshards: bool
    supports_full_expert_placement: bool
    supports_exact_replay: bool
    supports_prefill_decode_separation: bool
    storage_tiers: list[dict[str, Any]]
    gpu_devices: list[dict[str, Any]]
    cpu: dict[str, Any]
    memory: dict[str, Any]
    storage: dict[str, Any]
    cuda_kernel_proof: dict[str, Any] | None = None

    @model_validator(mode="after")
    def require_executable_backend(self) -> ColibriCapabilityReport:
        if self.supports_cuda and "cuda" not in self.execution_backends:
            raise ValueError("CUDA cannot be supported unless a CUDA execution path was probed")
        if self.supports_cuda:
            proof = self.cuda_kernel_proof or {}
            if not (
                proof.get("dll_loaded")
                and proof.get("device_detected")
                and proof.get("kernel_executed")
                and proof.get("correctness_passed")
            ):
                raise ValueError("CUDA support requires a successful bound kernel proof")
        if self.supports_vulkan and "vulkan" not in self.execution_backends:
            raise ValueError("Vulkan cannot be supported unless a Vulkan execution path was probed")
        if self.supports_metal and "metal" not in self.execution_backends:
            raise ValueError("Metal cannot be supported unless a Metal execution path was probed")
        if self.supports_tensor_microshards:
            raise ValueError("Colibri v1.4.0 does not execute tensor microshards")
        return self


class NativeQuantizationMetadata(StrictModel):
    format_name: str
    packing: str
    scale_format: str
    scale_group_size: int | None = Field(default=None, gt=0)
    quantization_aware_trained: bool
    reencoding_allowed: bool
    backend_requirements: list[str] = Field(default_factory=list)
    logical_shape: list[int] = Field(default_factory=list)
    packed_shape: list[int] = Field(default_factory=list)
    byte_size: int = Field(ge=0)

    @model_validator(mode="after")
    def protect_native_mxfp4(self) -> NativeQuantizationMetadata:
        if self.format_name.lower() == "mxfp4":
            if self.packing != "e2m1_two_nibbles":
                raise ValueError("MXFP4 must preserve its native E2M1 nibble packing")
            if self.scale_format != "ue8m0" or self.scale_group_size != 32:
                raise ValueError("MXFP4 requires UE8M0 scales in groups of 32")
            if self.reencoding_allowed:
                raise ValueError("native MXFP4 expert weights may not be silently reencoded")
        return self


class TensorInventoryEntry(StrictModel):
    tensor_id: str
    model_id: str
    model_revision: str
    engine_family: str
    layer_id: int = Field(ge=-1)
    tensor_name: str
    tensor_role: str
    expert_id: int | None = Field(default=None, ge=0)
    logical_shape: list[int]
    byte_size: int = Field(ge=0)
    storage_file: str
    storage_offset: int = Field(ge=0)
    storage_length: int = Field(ge=0)
    quantization: NativeQuantizationMetadata
    current_tier: str
    permitted_tiers: list[str]
    content_hash: str
    execution_backends: list[str]
    physical_storage_order: int = Field(ge=0)

    @model_validator(mode="after")
    def storage_length_matches(self) -> TensorInventoryEntry:
        if self.storage_length != self.byte_size:
            raise ValueError("storage_length and byte_size must reconcile")
        if not self.logical_shape or any(dimension <= 0 for dimension in self.logical_shape):
            raise ValueError("tensor shape must be non-empty and positive")
        return self


class ExpertInventoryEntry(StrictModel):
    layer_id: int = Field(ge=0)
    expert_id: int = Field(ge=0)
    expert_type: Literal["routed", "shared", "always_on", "latent", "grouped"] = "routed"
    tensor_ids: list[str]
    total_bytes: int = Field(ge=0)
    native_format: str
    storage_location: dict[str, Any]
    current_tier: str
    activation_count: int = Field(default=0, ge=0)
    activation_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    cache_hits: int = Field(default=0, ge=0)
    cache_misses: int = Field(default=0, ge=0)
    physical_storage_order: int = Field(ge=0)
    routing_metadata: dict[str, Any] = Field(default_factory=dict)


class ModelInventory(StrictModel):
    model_id: str
    model_revision: str
    model_family: str
    selected_engine_binary: str
    engine_build_fingerprint: str
    model_config_hash: str
    tokenizer_hash: str
    model_files: list[dict[str, Any]]
    quantization_formats: list[str]
    expert_geometry: dict[str, Any]
    tensor_count: int = Field(ge=0)
    expert_count: int = Field(ge=0)


class TierInventoryEntry(StrictModel):
    tier: Literal["vram", "ram", "nvme"]
    device_id: str
    total_capacity_bytes: int = Field(ge=0)
    available_capacity_bytes: int = Field(ge=0)
    allocated_model_bytes: int = Field(ge=0)
    allocated_expert_bytes: int = Field(ge=0)
    mutable_state_bytes: int = Field(ge=0)
    temporary_working_bytes: int = Field(ge=0)
    measured_bandwidth_bytes_per_second: float | None = Field(default=None, ge=0)
    measured_latency_ms: float | None = Field(default=None, ge=0)
    supported_formats: list[str]
    eviction_policy: str

    @model_validator(mode="after")
    def allocation_fits(self) -> TierInventoryEntry:
        accounted = (
            self.allocated_model_bytes
            + self.allocated_expert_bytes
            + self.mutable_state_bytes
            + self.temporary_working_bytes
        )
        if accounted > self.total_capacity_bytes:
            raise ValueError(f"{self.tier} allocation exceeds reported capacity")
        if self.available_capacity_bytes > self.total_capacity_bytes:
            raise ValueError("available capacity cannot exceed total capacity")
        return self


class SwarmColibriPlan(StrictModel):
    schema_version: str = "experiment-009-colibri-plan-v1"
    backend: Literal["colibri"] = "colibri"
    model: dict[str, Any]
    hardware_fingerprint: str
    dense_residency: dict[str, Any]
    shared_expert_residency: dict[str, Any]
    routed_expert_tiers: dict[str, Any]
    ram_budget_bytes: int = Field(ge=0)
    vram_budget_bytes: int = Field(ge=0)
    storage_policy: dict[str, Any]
    prefetch_policy: dict[str, Any]
    pipeline_policy: dict[str, Any]
    thread_policy: dict[str, Any]
    expected_bottleneck: str
    source: Literal["colibri_plan", "swarm_bounded_adjustment", "swarm_routing_history"]
    semantics_preserved: Literal[True] = True
    unsupported_features: list[str] = Field(default_factory=list)


class ColibriGenerationResult(StrictModel):
    request_id: str
    text: str
    input_token_ids: list[int] | None
    output_token_ids: list[int] | None
    token_identity_observed: bool
    stop_reason: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    elapsed_ms: float = Field(ge=0)
    time_to_first_token_ms: float | None = Field(default=None, ge=0)
    decode_tokens_per_second: float | None = Field(default=None, ge=0)
    raw_response: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def token_counts_reconcile(self) -> ColibriGenerationResult:
        if self.token_identity_observed:
            if self.input_token_ids is None or self.output_token_ids is None:
                raise ValueError("observed token identity requires input and output token IDs")
            if self.prompt_tokens != len(self.input_token_ids):
                raise ValueError("reported prompt token count does not match bridge token IDs")
            if self.completion_tokens != len(self.output_token_ids):
                raise ValueError("reported completion token count does not match bridge token IDs")
        elif self.input_token_ids is not None or self.output_token_ids is not None:
            raise ValueError("stock mode must represent unavailable token IDs as null, not empty")
        return self


class RouteSelection(StrictModel):
    call_index: int = Field(ge=0)
    row_index: int = Field(ge=0)
    layer_id: int = Field(ge=0)
    expert_id: int = Field(ge=0)
    routing_weight: float | None = None
    phase: Literal["prefill", "decode", "unknown"] = "unknown"
    token_index: int | None = Field(default=None, ge=0)
    request_id: str | None = None
    execution_tier: str | None = None


class TuningSample(StrictModel):
    candidate_id: str
    repeat: int = Field(ge=0)
    order: Literal["forward", "reverse"]
    decode_tokens_per_second: float = Field(gt=0)
    latency_ms: float | None = Field(default=None, ge=0)
    time_to_first_token_ms: float | None = Field(default=None, ge=0)
    p95_latency_ms: float | None = Field(default=None, ge=0)
    input_token_ids: list[int]
    output_token_ids: list[int]
    settings_applied: dict[str, Any]
    settings_ignored: list[str] = Field(default_factory=list)
