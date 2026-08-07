"""Bounded, metadata-only discovery for newly published model checkpoints."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

from pydantic import ConfigDict, PositiveInt

from swarm_inference.config.models import StrictModel
from swarm_inference.model.descriptor import ResolvedModelDescriptor
from swarm_inference.model.resolver import ModelResolution, ModelSourceResolver


class ModelInspector(Protocol):
    def inspect(self, source: str, **kwargs: Any) -> ModelResolution: ...


class ModelDiscoveryRecord(StrictModel):
    """One immutable checkpoint inspection, never an execution claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requested_reference: str
    status: str
    model_id: str | None = None
    revision: str | None = None
    model_fingerprint: str | None = None
    architecture_id: str | None = None
    architecture_adapter: str | None = None
    dense_or_moe: str | None = None
    format: str | None = None
    quantization: str | None = None
    modalities: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    inspection_error: str | None = None


class ModelDiscoveryReport(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0"
    inspected_count: PositiveInt
    records: tuple[ModelDiscoveryRecord, ...]


def _record(reference: str, descriptor: ResolvedModelDescriptor) -> ModelDiscoveryRecord:
    profile = descriptor.architecture_profile
    return ModelDiscoveryRecord(
        requested_reference=reference,
        status="PROFILED" if profile is not None else "NO_ARCHITECTURE_ADAPTER",
        model_id=descriptor.model_id,
        revision=descriptor.revision,
        model_fingerprint=descriptor.content_fingerprint,
        architecture_id=(
            profile.architecture_id if profile is not None else descriptor.architecture
        ),
        architecture_adapter=profile.adapter_id if profile is not None else None,
        dense_or_moe=profile.dense_or_moe if profile is not None else None,
        format=descriptor.format,
        quantization=descriptor.quantization,
        modalities=descriptor.modalities,
        capabilities=tuple(sorted(profile.capabilities)) if profile is not None else (),
        inspection_error=(
            str(descriptor.artifact_metadata.get("architecture_inspection_error"))
            if descriptor.artifact_metadata.get("architecture_inspection_error")
            else None
        ),
    )


def discover_models(
    references: Iterable[str],
    *,
    inspector: ModelInspector | None = None,
    maximum_models: int = 32,
) -> ModelDiscoveryReport:
    """Inspect a small explicit set without downloading model weights."""

    values = tuple(dict.fromkeys(item.strip() for item in references if item.strip()))
    if not values:
        raise ValueError("model discovery requires at least one explicit reference")
    if maximum_models <= 0 or maximum_models > 64:
        raise ValueError("maximum_models must be between 1 and 64")
    if len(values) > maximum_models:
        raise ValueError(
            f"model discovery received {len(values)} references; bounded limit is {maximum_models}"
        )
    resolver = inspector or ModelSourceResolver()
    records: list[ModelDiscoveryRecord] = []
    for reference in values:
        try:
            records.append(_record(reference, resolver.inspect(reference).descriptor))
        except (OSError, RuntimeError, ValueError) as exc:
            records.append(
                ModelDiscoveryRecord(
                    requested_reference=reference,
                    status="INSPECTION_FAILED",
                    inspection_error=f"{type(exc).__name__}: {exc}",
                )
            )
    return ModelDiscoveryReport(inspected_count=len(records), records=tuple(records))


__all__ = ["ModelDiscoveryRecord", "ModelDiscoveryReport", "discover_models"]
