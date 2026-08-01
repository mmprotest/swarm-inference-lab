"""Versioned backend-neutral Universal Worker ABI.

The ABI deliberately describes work rather than a Python implementation.  Its
wire representation is canonical JSON plus the existing ``SWARMT01`` binary
tensor envelope.  Arbitrary Python objects and pickle are never accepted.
"""

from __future__ import annotations

import base64
import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from swarm_inference.config.models import StrictModel
from swarm_inference.protocol.tensor_codec import ActivationTensor, decode_tensor, encode_tensor

UNIVERSAL_WORKER_ABI_MAJOR = 1
UNIVERSAL_WORKER_ABI_MINOR = 0


class ResultClassification(StrEnum):
    MEASURED_CUDA = "measured_cuda"
    MEASURED_X86_CPU = "measured_x86_cpu"
    MEASURED_MIXED_BACKEND = "measured_mixed_backend"
    ARM64_COMPATIBILITY = "arm64_compatibility"
    EMULATED_NETWORK = "emulated_network"
    PROJECTED_DEVICE_PROFILE = "projected_device_profile"


class WorkerProtocolVersion(StrictModel):
    major: int = Field(default=UNIVERSAL_WORKER_ABI_MAJOR, ge=0)
    minor: int = Field(default=UNIVERSAL_WORKER_ABI_MINOR, ge=0)
    capabilities: set[str] = Field(default_factory=set)

    def negotiate(self, peer: WorkerProtocolVersion) -> WorkerProtocolVersion | None:
        """Return the mutually usable version, or ``None`` for a major mismatch."""

        if self.major != peer.major:
            return None
        return WorkerProtocolVersion(
            major=self.major,
            minor=min(self.minor, peer.minor),
            capabilities=self.capabilities & peer.capabilities,
        )


class WorkerIdentity(StrictModel):
    worker_id: str
    node_id: str
    public_key: str
    backend_id: str
    protocol_version: WorkerProtocolVersion


class WorkerCapabilities(StrictModel):
    architecture: str
    operating_system: str
    cpu_model: str
    physical_cpu_cores: int = Field(gt=0)
    logical_cpu_cores: int = Field(gt=0)
    cpu_features: list[str] = Field(default_factory=list)
    accelerator_type: str | None = None
    accelerator_model: str | None = None
    accelerator_memory_bytes: int = Field(default=0, ge=0)
    system_memory_bytes: int = Field(gt=0)
    supported_weight_formats: list[str] = Field(default_factory=list)
    supported_activation_dtypes: list[str] = Field(default_factory=list)
    supported_cache_dtypes: list[str] = Field(default_factory=list)
    supported_collectives: list[str] = Field(default_factory=list)
    maximum_weight_bytes: int = Field(ge=0)
    maximum_cache_bytes: int = Field(ge=0)
    maximum_batch_size: int = Field(gt=0)
    maximum_context_length: int = Field(gt=0)
    measured_network_upload_bps: float = Field(ge=0)
    measured_network_download_bps: float = Field(ge=0)
    coordinator_latency_ms: float = Field(ge=0)
    backend_features: list[str] = Field(default_factory=list)


class WorkerBenchmarkProfile(StrictModel):
    model_revision: str | None = None
    shard_hash: str | None = None
    prefill_tokens_per_second: float | None = Field(default=None, ge=0)
    decode_tokens_per_second: float | None = Field(default=None, ge=0)
    matrix_kernel_results: dict[str, float] = Field(default_factory=dict)
    attention_kernel_results: dict[str, float] = Field(default_factory=dict)
    expert_kernel_results: dict[str, float] = Field(default_factory=dict)
    draft_tokens_per_second: float | None = Field(default=None, ge=0)
    expert_calls_per_second: float | None = Field(default=None, ge=0)
    background_tokens_per_second: float | None = Field(default=None, ge=0)
    model_load_seconds: float = Field(ge=0)
    warmup_seconds: float = Field(ge=0)
    measured_at_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class WorkerJobType(StrEnum):
    TARGET_PREFILL = "target_prefill"
    TARGET_DECODE = "target_decode"
    PIPELINE_STAGE_PREFILL = "pipeline_stage_prefill"
    PIPELINE_STAGE_DECODE = "pipeline_stage_decode"
    TENSOR_RANK = "tensor_rank"
    MOE_EXPERT = "moe_expert"
    SPECULATIVE_DRAFT = "speculative_draft"
    BACKGROUND_GENERATE = "background_generate"
    INTEGRITY_AUDIT = "integrity_audit"
    SHARD_CACHE = "shard_cache"


class WorkerJobStatus(StrEnum):
    ACCEPTED = "accepted"
    UNSUPPORTED = "unsupported"
    INSUFFICIENT_MEMORY = "insufficient_memory"
    INCOMPATIBLE_DTYPE = "incompatible_dtype"
    DEADLINE_IMPOSSIBLE = "deadline_impossible"
    BACKEND_FAILURE = "backend_failure"
    CANCELLED = "cancelled"


class TensorPayload(StrictModel):
    payload_kind: Literal["tensor"] = "tensor"
    encoding: Literal["SWARMT01"] = "SWARMT01"
    data_base64: str

    @field_validator("data_base64")
    @classmethod
    def validate_base64(cls, value: str) -> str:
        try:
            base64.b64decode(value, validate=True)
        except ValueError as exc:
            raise ValueError("tensor payload is not valid base64") from exc
        return value

    @classmethod
    def from_tensor(cls, tensor: ActivationTensor) -> TensorPayload:
        return cls(data_base64=base64.b64encode(encode_tensor(tensor)).decode("ascii"))

    def to_tensor(self) -> ActivationTensor:
        return decode_tensor(base64.b64decode(self.data_base64, validate=True))


class TokenPayload(StrictModel):
    payload_kind: Literal["tokens"] = "tokens"
    token_ids: list[int] = Field(default_factory=list)
    text: str | None = None
    tokenizer_hash: str | None = None


InputPayload = Annotated[TensorPayload | TokenPayload, Field(discriminator="payload_kind")]


class CacheReference(StrictModel):
    cache_id: str
    owner_worker_id: str
    generation: int = Field(ge=0)
    token_count: int = Field(ge=0)
    checksum: str | None = None


class GenerationParameters(StrictModel):
    max_new_tokens: int = Field(default=1, gt=0)
    temperature: float = Field(default=0.0, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: int = Field(default=0, ge=0)
    ignore_eos: bool = False
    seed: int = 7


class WorkerJob(StrictModel):
    job_id: str
    request_id: str
    role: WorkerJobType
    model_id: str
    model_revision: str
    partition_manifest_hash: str | None = None
    shard_hash: str | None = None
    input_payload: InputPayload
    cache_reference: CacheReference | None = None
    generation_parameters: GenerationParameters | None = None
    deadline_ms: int = Field(gt=0)
    priority: int = Field(default=0, ge=0)
    route_generation: int = Field(default=0, ge=0)
    created_at_unix_ms: int = Field(default_factory=lambda: time.time_ns() // 1_000_000)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def remaining_deadline_ms(self) -> float:
        return self.deadline_ms - (time.time_ns() // 1_000_000 - self.created_at_unix_ms)


class WorkerJobResult(StrictModel):
    job_id: str
    request_id: str
    status: WorkerJobStatus
    output_payload: InputPayload | None = None
    detail: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    classification: ResultClassification | None = None


class BackendArtifactMapping(StrictModel):
    canonical_model_id: str
    canonical_revision: str
    canonical_partition_hash: str
    backend_id: str
    backend_artifact_path: str
    backend_artifact_hash: str
    conversion_tool: str
    conversion_version: str
    conversion_parameters: dict[str, object] = Field(default_factory=dict)
    canonical_tensor_mapping: dict[str, str] = Field(default_factory=dict)
    weight_format: str
    conversion_loss: str
    tokenizer_hash: str | None = None
    vocabulary_hash: str | None = None
    special_tokens_hash: str | None = None

    @model_validator(mode="after")
    def validate_identity(self) -> BackendArtifactMapping:
        if not self.canonical_revision:
            raise ValueError("canonical revision must be immutable and non-empty")
        if not self.canonical_partition_hash:
            raise ValueError("canonical partition hash is required")
        if not self.backend_artifact_hash:
            raise ValueError("backend artifact hash is required")
        return self


class BackendInterfaceEvidence(StrictModel):
    backend_id: Literal["mlx", "executorch", "vulkan", "rocm"]
    implementation_status: Literal["interface_defined"] = "interface_defined"
    physical_execution_status: Literal["physical_execution_unproven"] = (
        "physical_execution_unproven"
    )


class BackendAdapter(ABC):
    """Strict adapter interface.  Implementations may not delegate to another backend."""

    backend_id: str
    supported_jobs: frozenset[WorkerJobType]

    @abstractmethod
    def capabilities(self) -> WorkerCapabilities:
        raise NotImplementedError

    @abstractmethod
    def benchmark_profile(self) -> WorkerBenchmarkProfile:
        raise NotImplementedError

    def admission_result(self, job: WorkerJob) -> WorkerJobResult | None:
        if job.role not in self.supported_jobs:
            return WorkerJobResult(
                job_id=job.job_id,
                request_id=job.request_id,
                status=WorkerJobStatus.UNSUPPORTED,
                detail=f"{self.backend_id} does not support {job.role.value}",
            )
        if job.remaining_deadline_ms <= 0:
            return WorkerJobResult(
                job_id=job.job_id,
                request_id=job.request_id,
                status=WorkerJobStatus.DEADLINE_IMPOSSIBLE,
                detail="job deadline elapsed before admission",
            )
        return None

    @abstractmethod
    async def execute(self, job: WorkerJob) -> WorkerJobResult:
        raise NotImplementedError

    @abstractmethod
    async def cancel(self, request_id: str) -> bool:
        raise NotImplementedError

    async def shutdown(self) -> None:
        return None


def tensor_payload_from_array(
    array: np.ndarray,
    *,
    tensor_id: str,
    request_id: str,
    stage_id: int,
    token_position: int,
    sequence_length: int,
    model_revision: str,
    partition_hash: str,
    route_generation: int,
    logical_dtype: str | None = None,
) -> TensorPayload:
    return TensorPayload.from_tensor(
        ActivationTensor(
            tensor_id=tensor_id,
            request_id=request_id,
            stage_id=stage_id,
            token_position=token_position,
            sequence_length=sequence_length,
            array=array,
            logical_dtype=logical_dtype,
            model_revision=model_revision,
            partition_hash=partition_hash,
            route_generation=route_generation,
        )
    )
