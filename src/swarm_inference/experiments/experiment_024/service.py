"""Frozen calibrated services consumed by the E024 event model."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

from .correctness import ModelInvalidError
from .freeze import LAYER_ZERO_WHOLE_CANDIDATE_ID

SERVICE_RELATIVE_PATH = Path("artifacts/experiment-024/calibration/e024-service.csv")
CATALOG_RELATIVE_PATH = Path(
    "artifacts/experiment-022/completion/rerun/candidate-catalog.json"
)


@dataclass(frozen=True, slots=True)
class ServiceKey:
    layer: int
    operation: str
    degree: int
    rows: int


class E024ServiceTable:
    """Exact p50 lookup with no interpolation or post-hoc correction."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.path = self.repo_root / SERVICE_RELATIVE_PATH
        if not self.path.is_file():
            raise ModelInvalidError(f"missing calibrated service table: {self.path}")
        with self.path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self._values: dict[ServiceKey, float] = {}
        for row in rows:
            key = ServiceKey(
                int(row["layer"]),
                str(row["operation"]),
                int(row["degree"]),
                int(row["rows"]),
            )
            value = float(row["p50_ms"])
            if not math.isfinite(value) or value <= 0:
                raise ModelInvalidError(f"invalid calibrated service for {key}")
            if key in self._values:
                raise ModelInvalidError(f"duplicate calibrated service for {key}")
            self._values[key] = value
        catalog = json.loads(
            (self.repo_root / CATALOG_RELATIVE_PATH).read_text(encoding="utf-8")
        )
        self.layer_type_by_id = {
            int(row["layer"]): str(row["layer_type"]).upper()
            for row in catalog["candidates"]
            if row["candidate_type"] == "WHOLE_LAYER"
        }
        self.whole_resident_bytes_by_layer = {
            int(row["layer"]): int(row["resident_memory_bytes"][0])
            for row in catalog["candidates"]
            if row["candidate_type"] == "WHOLE_LAYER" and int(row["degree"]) == 1
        }
        self._validate_complete()

    def _validate_complete(self) -> None:
        required_p8 = (
            "attention_preprocess",
            "attention_common",
            "attention_shard",
            "attention_reduction",
            "post_attention_preprocess",
            "router",
            "latent_down",
            "expert_stripe",
            "expert_reduction",
            "routed_norm",
            "latent_up",
            "latent_up_reduction",
            "shared_expert",
            "shared_reduction",
            "routed_shared_reduction",
            "output_state_commit",
            "worker_protocol",
        )
        p8_degree = {
            "attention_shard",
            "attention_reduction",
            "latent_down",
            "expert_stripe",
            "expert_reduction",
            "latent_up",
            "latent_up_reduction",
            "shared_expert",
            "shared_reduction",
        }
        for rows in (1, 2, 4):
            self.service_ms(0, "whole_layer", 1, rows)
            self.service_ms(0, "worker_protocol", 1, rows)
            self.service_ms(-1, "d_worker_local_fusion", 1, rows)
            for layer in range(1, 93):
                self.service_ms(layer, "whole_layer", 1, rows)
                for operation in required_p8:
                    degree = (
                        2
                        if operation == "routed_shared_reduction"
                        else 8
                        if operation in p8_degree
                        else 1
                    )
                    self.service_ms(layer, operation, degree, rows)
        if self.layer_type_by_id.get(0) != "DENSE":
            raise ModelInvalidError("service catalog changed dense layer 0")
        if len(self.layer_type_by_id) != 93:
            raise ModelInvalidError("service catalog does not cover 93 layers")

    def service_ms(
        self,
        layer: int,
        operation: str,
        degree: int,
        rows: int,
    ) -> float:
        if rows not in (1, 2, 4):
            raise ValueError("E024 services exist only for rows 1, 2, and 4")
        key = ServiceKey(layer, operation, degree, rows)
        try:
            return self._values[key]
        except KeyError as exc:
            raise ModelInvalidError(f"missing calibrated service for {key}") from exc

    def dense_layer_zero_ms(self, rows: int, compute_multiplier: float) -> float:
        if not 0 < compute_multiplier <= 1.0:
            raise ValueError("compute multiplier must be in (0, 1]")
        return self.service_ms(0, "whole_layer", 1, rows) / compute_multiplier

    @property
    def layer_zero_candidate_id(self) -> str:
        return LAYER_ZERO_WHOLE_CANDIDATE_ID


__all__ = ["SERVICE_RELATIVE_PATH", "E024ServiceTable", "ServiceKey"]
