"""Machine-readable compatibility registry generated from executable probes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import ConfigDict, Field

from swarm_inference.config.models import StrictModel
from swarm_inference.engines.interfaces import (
    ClusterCapabilities,
    CompatibilityStatus,
    EngineSupportReport,
    EngineSupportStatus,
    ExecutionEngine,
)
from swarm_inference.model.descriptor import ResolvedModelDescriptor


class ValidationEvidence(StrictModel):
    """Content-addressed evidence; claims never outlive their exact artifact/runtime."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_fingerprint: str
    engine_id: str
    runtime_fingerprint: str
    status: CompatibilityStatus
    evidence_fingerprint: str
    metrics: dict[str, float | int | None] = Field(default_factory=dict)


class EngineCompatibilityRecord(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    engine_id: str
    support_status: CompatibilityStatus
    validation_status: CompatibilityStatus
    supported: bool
    architecture_supported: bool
    format_supported: bool
    quantization_supported: bool
    hardware_supported: bool
    adapter_id: str | None = None
    capabilities: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    expected_compute_cost: float | None = None
    expected_network_cost: float | None = None
    expected_memory_cost: int | None = None
    confidence: float = 0.0
    runtime_identity: dict[str, Any] = Field(default_factory=dict)
    evidence_fingerprint: str | None = None
    measurements: dict[str, float | int | None] = Field(default_factory=dict)


class ArtifactCompatibilityRecord(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    revision: str
    model_fingerprint: str
    architecture_id: str
    dense_or_moe: str
    format: str
    quantization: str | None = None
    engines: dict[str, EngineCompatibilityRecord]


class CompatibilityRegistry(StrictModel):
    """Probe-derived registry grouped by architecture and artifact format."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0"
    architectures: dict[str, dict[str, tuple[ArtifactCompatibilityRecord, ...]]]


def _support_status(report: EngineSupportReport) -> CompatibilityStatus:
    if report.supported:
        return (
            CompatibilityStatus.SUPPORTED_WITH_LIMITATIONS
            if report.limitations
            else CompatibilityStatus.SUPPORTED
        )
    if report.status == EngineSupportStatus.COMPONENT_SUPPORTED:
        return CompatibilityStatus.SUPPORTED_WITH_LIMITATIONS
    if report.status == EngineSupportStatus.UNSUPPORTED_FORMAT:
        return CompatibilityStatus.UNSUPPORTED_FORMAT
    if report.status == EngineSupportStatus.UNSUPPORTED_ARCHITECTURE:
        return CompatibilityStatus.UNSUPPORTED_ARCHITECTURE
    if (
        report.status == EngineSupportStatus.UNSUPPORTED_QUANTIZATION
        or report.quantization_supported is False
    ):
        return CompatibilityStatus.UNSUPPORTED_QUANTIZATION
    if report.status in {
        EngineSupportStatus.MISSING_RUNTIME,
        EngineSupportStatus.BROKEN_RUNTIME,
        EngineSupportStatus.MISSING_DEVICE_CAPABILITY,
        EngineSupportStatus.INSUFFICIENT_MEMORY,
    }:
        return CompatibilityStatus.UNAVAILABLE_RUNTIME
    return CompatibilityStatus.NOT_TESTED


def _validation_for(
    model: ResolvedModelDescriptor,
    report: EngineSupportReport,
    evidence: Mapping[tuple[str, str], ValidationEvidence],
) -> ValidationEvidence | None:
    item = evidence.get((model.content_fingerprint, report.engine_id))
    if item is None:
        return None
    fingerprint = engine_runtime_fingerprint(report)
    if fingerprint is None or fingerprint != item.runtime_fingerprint:
        return None
    return item


def engine_runtime_fingerprint(report: EngineSupportReport) -> str | None:
    """Hash every probe-owned runtime fact used to bind validation evidence."""

    if not report.runtime_identity:
        return None
    payload = json.dumps(
        report.runtime_identity,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def build_compatibility_registry(
    models: Iterable[ResolvedModelDescriptor],
    engines: Iterable[ExecutionEngine],
    cluster: ClusterCapabilities,
    *,
    validation_evidence: Iterable[ValidationEvidence] = (),
) -> CompatibilityRegistry:
    """Probe every supplied artifact; never infer validation from a family label."""

    engine_values = tuple(engines)
    evidence = {(item.model_fingerprint, item.engine_id): item for item in validation_evidence}
    grouped: dict[str, dict[str, list[ArtifactCompatibilityRecord]]] = {}
    for model in models:
        profile = model.architecture_profile
        architecture_id = (
            profile.architecture_id if profile is not None else model.architecture or "unknown"
        )
        dense_or_moe = profile.dense_or_moe if profile is not None else "unknown"
        records: dict[str, EngineCompatibilityRecord] = {}
        for engine in engine_values:
            report = engine.probe(model, cluster)
            validated = _validation_for(model, report, evidence)
            records[engine.engine_id] = EngineCompatibilityRecord(
                engine_id=engine.engine_id,
                support_status=_support_status(report),
                validation_status=(
                    validated.status if validated is not None else CompatibilityStatus.NOT_TESTED
                ),
                supported=report.supported,
                architecture_supported=bool(report.architecture_supported),
                format_supported=bool(report.format_supported),
                quantization_supported=bool(report.quantization_supported),
                hardware_supported=bool(report.hardware_supported),
                adapter_id=report.adapter_id,
                capabilities=report.capabilities,
                limitations=report.limitations,
                rejection_reasons=report.rejection_reasons,
                expected_compute_cost=report.expected_compute_cost,
                expected_network_cost=report.expected_network_cost,
                expected_memory_cost=report.expected_memory_cost,
                confidence=report.confidence,
                runtime_identity=report.runtime_identity,
                evidence_fingerprint=(
                    validated.evidence_fingerprint if validated is not None else None
                ),
                measurements=validated.metrics if validated is not None else {},
            )
        artifact = ArtifactCompatibilityRecord(
            model_id=model.model_id,
            revision=model.revision,
            model_fingerprint=model.content_fingerprint,
            architecture_id=architecture_id,
            dense_or_moe=dense_or_moe,
            format=model.format,
            quantization=model.quantization,
            engines=records,
        )
        grouped.setdefault(architecture_id, {}).setdefault(model.format, []).append(artifact)
    frozen = {
        architecture: {
            model_format: tuple(
                sorted(
                    items, key=lambda item: (item.model_id, item.revision, item.quantization or "")
                )
            )
            for model_format, items in sorted(formats.items())
        }
        for architecture, formats in sorted(grouped.items())
    }
    return CompatibilityRegistry(architectures=frozen)


__all__ = [
    "ArtifactCompatibilityRecord",
    "CompatibilityRegistry",
    "EngineCompatibilityRecord",
    "ValidationEvidence",
    "build_compatibility_registry",
    "engine_runtime_fingerprint",
]
