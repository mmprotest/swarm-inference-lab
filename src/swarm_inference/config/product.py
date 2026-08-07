"""Configuration for the persistent product coordinator path."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import Field, PositiveInt, field_validator, model_validator

from swarm_inference.config.models import StrictModel


class ProductCoordinatorConfig(StrictModel):
    kind: Literal["product-stage-ring"] = "product-stage-ring"
    schema_version: str = "1"
    # Adapter discovery is model-driven. This remains an override for private
    # deployments that intentionally restrict their native adapter registry.
    default_adapter_id: str | None = None
    default_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    local_only_by_default: bool = True
    worker_heartbeat_timeout_s: float = Field(default=15.0, gt=0)
    deployment_lease_seconds: float = Field(default=30 * 24 * 60 * 60, gt=0)
    control_timeout_s: float = Field(default=120.0, gt=0)
    request_timeout_s: float = Field(default=300.0, gt=0)
    engine_action_lease_seconds: PositiveInt = Field(default=300, le=3600)
    event_queue_capacity: PositiveInt = 256
    token_ingress_capacity: PositiveInt = 256
    planning_max_sequence_tokens: PositiveInt = 2048
    maximum_candidate_workers: PositiveInt = Field(default=64, le=256)
    maximum_stage_count: PositiveInt = Field(default=32, le=128)
    planning_beam_width: PositiveInt = Field(default=512, le=8192)
    network_measurement_ttl_seconds: PositiveInt = Field(default=900, le=86_400)
    network_probe_max_bytes: PositiveInt = Field(
        default=16 * 1024 * 1024,
        ge=1024,
        le=256 * 1024 * 1024,
    )
    network_probe_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    allow_unmeasured_links_for_explicit_plans: bool = True
    balanced_throughput_weight: float = Field(default=0.45, ge=0)
    balanced_memory_headroom_weight: float = Field(default=0.25, ge=0)
    balanced_reliability_weight: float = Field(default=0.20, ge=0)
    balanced_participation_weight: float = Field(default=0.10, ge=0)
    maximum_active_sessions_per_worker: PositiveInt = 256
    coordinator_id: str = "coordinator"
    route_future_tolerance_s: float = Field(default=30.0, ge=0)
    route_nonce_cache_capacity: PositiveInt = 4096
    cleanup_timeout_s: float = Field(default=10.0, gt=0)
    recovery_timeout_s: float = Field(default=120.0, gt=0)
    maximum_recovery_attempts: PositiveInt = 2
    trusted_worker_fingerprints: list[str] = Field(default_factory=list)
    require_trusted_workers: bool = True
    trust_store_path: Path | None = None

    @field_validator("trusted_worker_fingerprints")
    @classmethod
    def validate_trusted_worker_fingerprints(cls, values: list[str]) -> list[str]:
        from swarm_inference.security.trust_store import normalize_fingerprint

        return sorted({normalize_fingerprint(value) for value in values})

    @model_validator(mode="after")
    def validate_balanced_weights(self) -> Self:
        total = (
            self.balanced_throughput_weight
            + self.balanced_memory_headroom_weight
            + self.balanced_reliability_weight
            + self.balanced_participation_weight
        )
        if total <= 0:
            raise ValueError("at least one balanced-planning objective weight must be positive")
        return self


def load_product_config(path: Path) -> ProductCoordinatorConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("product configuration must contain a YAML object")
    return ProductCoordinatorConfig.model_validate(raw)


__all__ = ["ProductCoordinatorConfig", "load_product_config"]
