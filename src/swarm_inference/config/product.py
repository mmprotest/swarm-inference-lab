"""Configuration for the persistent product coordinator path."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, PositiveInt

from swarm_inference.config.models import StrictModel


class ProductCoordinatorConfig(StrictModel):
    kind: Literal["product-stage-ring"] = "product-stage-ring"
    schema_version: str = "1"
    default_adapter_id: Literal["olmoe"] = "olmoe"
    default_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    local_only_by_default: bool = True
    worker_heartbeat_timeout_s: float = Field(default=15.0, gt=0)
    deployment_lease_seconds: float = Field(default=30 * 24 * 60 * 60, gt=0)
    control_timeout_s: float = Field(default=120.0, gt=0)
    request_timeout_s: float = Field(default=300.0, gt=0)
    event_queue_capacity: PositiveInt = 256
    token_ingress_capacity: PositiveInt = 256
    planning_max_sequence_tokens: PositiveInt = 2048
    maximum_active_sessions_per_worker: PositiveInt = 256


def load_product_config(path: Path) -> ProductCoordinatorConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("product configuration must contain a YAML object")
    return ProductCoordinatorConfig.model_validate(raw)


__all__ = ["ProductCoordinatorConfig", "load_product_config"]
