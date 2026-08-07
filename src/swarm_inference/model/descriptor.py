"""Execution-neutral immutable model facts."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, NonNegativeInt

from swarm_inference.config.models import StrictModel


class ModelFileDescriptor(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str = Field(min_length=1)
    size_bytes: NonNegativeInt
    sha256: str | None = None
    etag: str | None = None
    multipart_group: str | None = None
    multipart_index: int | None = Field(default=None, ge=1)
    multipart_count: int | None = Field(default=None, ge=1)


class ResolvedModelDescriptor(StrictModel):
    """Facts about one immutable model artifact, with no execution decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    content_fingerprint: str = Field(min_length=1)
    source_type: Literal["huggingface", "local"]
    format: Literal["safetensors", "gguf", "pytorch", "unknown"]
    architecture: str | None = None
    architecture_raw: str | None = None
    architecture_source: Literal[
        "config.architectures",
        "config.model_type",
        "gguf.general.architecture",
        "unknown",
    ] = "unknown"
    files: tuple[ModelFileDescriptor, ...]
    variant: str | None = None
    quantization: str | None = None
    weight_bytes: NonNegativeInt
    layer_count: int | None = Field(default=None, ge=0)
    hidden_size: int | None = Field(default=None, gt=0)
    activation_dtype_bytes: int | None = Field(default=None, gt=0)
    parameter_count: int | None = Field(default=None, ge=0)
    tokenizer_identity: str | None = None
    modalities: tuple[str, ...] = ("text",)
    features: tuple[str, ...] = ()
    local_paths: tuple[str, ...] = ()

    @property
    def is_local(self) -> bool:
        return self.source_type == "local"


__all__ = ["ModelFileDescriptor", "ResolvedModelDescriptor"]
