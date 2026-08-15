"""Validated resident service curves used by both E022 planner arms."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import LayerSpec, LayerType, PartitionKind


@dataclass(frozen=True, slots=True)
class ServiceKey:
    layer_type: LayerType
    operation: str
    degree: int
    rows: int


class ResidentServiceModel:
    """Feature-conditioned physical service estimates with no global multiplier."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        calibration: dict[ServiceKey, list[float]] = defaultdict(list)
        layer_calibration: dict[tuple[int, str, int, int], list[float]] = defaultdict(list)
        exact: dict[tuple[int, str, int, int], list[float]] = defaultdict(list)
        self.sources: set[str] = set()
        for row in rows:
            if str(row.get("valid_for_service", "True")).lower() not in (
                "true",
                "1",
                "yes",
            ):
                continue
            layer_type = LayerType(str(row["layer_type"]))
            operation = str(row["operation"])
            degree = int(row["degree"])
            chunk_rows = int(row["rows"])
            value = float(row["p50_ms"])
            if value <= 0:
                raise ValueError("service values must be positive")
            key = ServiceKey(layer_type, operation, degree, chunk_rows)
            if str(row.get("split", "calibration")) == "calibration":
                calibration[key].append(value)
                layer_calibration[
                    (int(row["layer"]), operation, degree, chunk_rows)
                ].append(value)
            exact[(int(row["layer"]), operation, degree, chunk_rows)].append(value)
            if row.get("source"):
                self.sources.add(str(row["source"]))
        self._service = {
            key: statistics.median(values) for key, values in calibration.items()
        }
        self._layer_service = {
            key: statistics.median(values) for key, values in layer_calibration.items()
        }
        self._exact = {
            key: statistics.median(values) for key, values in exact.items()
        }
        if not self._service:
            raise ValueError("resident service model has no calibration rows")

    @classmethod
    def from_csv(cls, path: Path) -> ResidentServiceModel:
        with path.open(encoding="utf-8", newline="") as handle:
            return cls([dict(row) for row in csv.DictReader(handle)])

    def has(self, layer: LayerSpec, operation: str, degree: int, rows: int) -> bool:
        return (
            (layer.layer_id, operation, degree, rows) in self._layer_service
            or ServiceKey(layer.layer_type, operation, degree, rows) in self._service
        )

    def service_ms(
        self,
        layer: LayerSpec,
        operation: str,
        degree: int,
        rows: int,
    ) -> float:
        layer_key = (layer.layer_id, operation, degree, rows)
        if layer_key in self._layer_service:
            return self._layer_service[layer_key]
        key = ServiceKey(layer.layer_type, operation, degree, rows)
        try:
            return self._service[key]
        except KeyError as exc:
            raise KeyError(
                f"no validated resident service for {layer.layer_type.value}/"
                f"{operation}/p{degree}/r{rows}"
            ) from exc

    def exact_physical_ms(
        self,
        layer: int,
        operation: str,
        degree: int,
        rows: int,
    ) -> float | None:
        return self._exact.get((layer, operation, degree, rows))

    def supported_candidate(
        self,
        layer: LayerSpec,
        kind: PartitionKind,
        degree: int,
        rows: int,
    ) -> bool:
        if kind is PartitionKind.WHOLE_LAYER:
            return self.has(layer, "whole_layer", 1, rows) and self.has(
                layer, "worker_protocol", 1, rows
            )
        common = {
            ("router", 1),
            ("worker_protocol", 1),
            ("attention_preprocess", 1),
            ("post_attention_preprocess", 1),
            ("routed_shared_reduction", 2),
            ("routed_norm", 1),
            ("output_state_commit", 1),
        }
        if kind is PartitionKind.WHOLE_EXPERT:
            required = common | {
                ("attention_whole", 1),
                ("latent_down_whole", 1),
                ("expert_whole_group", degree),
                ("expert_whole_group_remote", degree),
                ("shared_expert_whole", 1),
                ("latent_up_whole", 1),
                ("expert_reduction", degree),
            }
        elif kind is PartitionKind.EXPERT_SHARD:
            required = common | {
                ("attention_whole", 1),
                ("latent_down_whole", 1),
                ("expert_stripe", degree),
                ("expert_stripe_remote", degree),
                ("shared_expert_whole", 1),
                ("latent_up_whole", 1),
                ("expert_reduction", degree),
            }
        elif kind is PartitionKind.ATTENTION_PROJECTION_SHARD:
            required = common | {
                ("attention_common", 1),
                ("attention_shard", degree),
                ("attention_shard_remote", degree),
                ("latent_down_whole", 1),
                ("expert_whole", 1),
                ("shared_expert_whole", 1),
                ("latent_up_whole", 1),
                ("attention_reduction", degree),
            }
        elif kind is PartitionKind.FULL_MIXED_STRIPE:
            required = common | {
                ("attention_common", 1),
                ("attention_shard", degree),
                ("attention_shard_remote", degree),
                ("latent_down", degree),
                ("latent_down_remote", degree),
                ("latent_up", degree),
                ("latent_up_remote", degree),
                ("expert_stripe", degree),
                ("expert_stripe_remote", degree),
                ("shared_expert", degree),
                ("shared_expert_remote", degree),
                ("attention_reduction", degree),
                ("expert_reduction", degree),
                ("shared_reduction", degree),
                ("latent_up_reduction", degree),
            }
        else:
            return False
        return all(self.has(layer, operation, operation_degree, rows) for operation, operation_degree in required)

    def describe(self) -> dict[str, Any]:
        return {
            "schema_version": "experiment-022-resident-service-model-v1",
            "condition_count": len(self._service),
            "feature_keys": ["layer_type", "operation", "degree", "rows"],
            "normalization_applied": False,
            "global_multiplier": None,
            "sources": sorted(self.sources),
        }


__all__ = ["ResidentServiceModel", "ServiceKey"]
