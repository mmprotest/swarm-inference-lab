"""Immutable product model identity and metadata records.

These records describe what may be deployed without naming a host-local model
path.  Workers resolve the exact identity independently from their configured
cache and the coordinator compares the resulting metadata hashes.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import ConfigDict, Field, PositiveInt, model_validator

from swarm_inference.config.models import StrictModel
from swarm_inference.model.partition import LayerCost, ModelPartitionMetadata


class ModelResolutionPolicy(StrEnum):
    LOCAL_ONLY = "local-only"
    ALLOW_DOWNLOAD = "allow-download"


class ProductModelReference(StrictModel):
    """User-supplied exact model identity before metadata is resolved."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    tokenizer_revision: str = Field(min_length=1)
    adapter_id: str = Field(default="olmoe", min_length=1)
    dtype: str = Field(default="bfloat16", min_length=1)
    resolution_policy: ModelResolutionPolicy = ModelResolutionPolicy.LOCAL_ONLY


class ProductLayerCost(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    layer_id: int = Field(ge=0)
    execution_ns: int = Field(ge=0)
    weight_bytes: int = Field(ge=0)
    kv_bytes_per_token: int = Field(ge=0)
    peak_temporary_bytes: int = Field(ge=0)
    activation_bytes: int = Field(ge=0)
    measured: bool

    def to_partition_cost(self) -> LayerCost:
        return LayerCost(**self.model_dump())


class ProductModelMetadata(StrictModel):
    """Header/config-only metadata sufficient for contiguous planning."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    layer_costs: tuple[ProductLayerCost, ...]
    embedding_weight_bytes: int = Field(ge=0)
    final_weight_bytes: int = Field(ge=0)
    dtype_bytes: PositiveInt
    hidden_size: PositiveInt
    metadata_hash: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_layers(self) -> ProductModelMetadata:
        if [cost.layer_id for cost in self.layer_costs] != list(range(len(self.layer_costs))):
            raise ValueError("product layer costs must be contiguous from layer zero")
        if not self.layer_costs:
            raise ValueError("product metadata must describe at least one layer")
        return self

    def to_partition_metadata(
        self,
        *,
        model_revision: str,
        tokenizer_revision: str,
    ) -> ModelPartitionMetadata:
        return ModelPartitionMetadata(
            layer_costs=tuple(cost.to_partition_cost() for cost in self.layer_costs),
            embedding_weight_bytes=self.embedding_weight_bytes,
            final_weight_bytes=self.final_weight_bytes,
            dtype_bytes=self.dtype_bytes,
            hidden_size=self.hidden_size,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            metadata_hash=self.metadata_hash,
        )


class ProductModelSpec(StrictModel):
    """Canonical immutable model specification used by plans and deployments."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    tokenizer_revision: str = Field(min_length=1)
    adapter_id: str = Field(min_length=1)
    dtype: str = Field(min_length=1)
    layer_count: PositiveInt
    hidden_size: PositiveInt
    metadata_hash: str = Field(min_length=1)
    resolution_policy: ModelResolutionPolicy

    @classmethod
    def resolved(
        cls,
        reference: ProductModelReference,
        metadata: ProductModelMetadata,
    ) -> ProductModelSpec:
        return cls(
            model_id=reference.model_id,
            model_revision=reference.model_revision,
            tokenizer_revision=reference.tokenizer_revision,
            adapter_id=reference.adapter_id,
            dtype=reference.dtype,
            layer_count=len(metadata.layer_costs),
            hidden_size=metadata.hidden_size,
            metadata_hash=metadata.metadata_hash,
            resolution_policy=reference.resolution_policy,
        )


__all__ = [
    "ModelResolutionPolicy",
    "ProductLayerCost",
    "ProductModelMetadata",
    "ProductModelReference",
    "ProductModelSpec",
]
