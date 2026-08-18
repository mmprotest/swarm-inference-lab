"""Corrected deterministic Kimi K3 commodity candidate selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .freeze import (
    CANDIDATE_CATALOG_RELATIVE_PATH,
    DEGREE,
    LAYER_ZERO_WHOLE_CANDIDATE_ID,
)

PARTITION_PRIORITY = (
    "ATTENTION_PROJECTION_SHARD",
    "EXPERT_SHARD",
    "WHOLE_EXPERT",
)


@dataclass(frozen=True, slots=True)
class CandidateAdmission:
    layer: int
    candidate_id: str | None
    candidate_type: str | None
    degree: int | None
    resident_memory_bytes: tuple[int, ...]
    checkpoint_bytes: tuple[int, ...]
    status: str
    reason: str | None


def _physically_admitted(candidate: dict[str, Any]) -> bool:
    return (
        candidate.get("headline_eligible") is True
        and candidate.get("production_native_binding") is True
        and candidate.get("correctness_status") == "PASS"
        and list(candidate.get("chunk_sizes_physically_validated", [])) == [1, 2, 4]
    )


def _admitted_p8(candidate: dict[str, Any]) -> bool:
    return (
        int(candidate.get("degree", -1)) == DEGREE
        and candidate.get("candidate_type") != "WHOLE_LAYER"
        and _physically_admitted(candidate)
    )


def _admitted_layer_zero(candidate: dict[str, Any]) -> bool:
    return (
        str(candidate.get("candidate_id")) == LAYER_ZERO_WHOLE_CANDIDATE_ID
        and candidate.get("candidate_type") == "WHOLE_LAYER"
        and int(candidate.get("degree", -1)) == 1
        and _physically_admitted(candidate)
    )


def _admission(layer: int, candidate: dict[str, Any]) -> CandidateAdmission:
    return CandidateAdmission(
        layer=layer,
        candidate_id=str(candidate["candidate_id"]),
        candidate_type=str(candidate["candidate_type"]),
        degree=int(candidate["degree"]),
        resident_memory_bytes=tuple(
            int(value) for value in candidate["resident_memory_bytes"]
        ),
        checkpoint_bytes=tuple(int(value) for value in candidate["checkpoint_bytes"]),
        status="PASS",
        reason=None,
    )


class CommodityK3Planner:
    """Choose one admitted whole dense layer and P8 for all MoE layers."""

    def __init__(self, repo_root: Path) -> None:
        path = repo_root.resolve() / CANDIDATE_CATALOG_RELATIVE_PATH
        self.catalog = json.loads(path.read_text(encoding="utf-8"))

    def choose_candidate(self, layer: int) -> CandidateAdmission:
        if layer not in range(93):
            raise ValueError("Kimi K3 transformer layer must be in 0..92")
        layer_candidates = [
            row
            for row in self.catalog["candidates"]
            if int(row.get("layer", -1)) == layer
        ]
        if layer == 0:
            candidates = [row for row in layer_candidates if _admitted_layer_zero(row)]
            if len(candidates) == 1:
                return _admission(layer, candidates[0])
            return CandidateAdmission(
                layer,
                None,
                None,
                None,
                (),
                (),
                "PLACEMENT_INFEASIBLE",
                "no exact admitted layer-00:WHOLE_LAYER:p1 candidate",
            )

        candidates = [row for row in layer_candidates if _admitted_p8(row)]
        full_mixed = [
            row for row in candidates if row["candidate_type"] == "FULL_MIXED_STRIPE"
        ]
        if full_mixed:
            chosen = min(full_mixed, key=lambda row: str(row["candidate_id"]))
            return _admission(layer, chosen)
        priority = {name: index for index, name in enumerate(PARTITION_PRIORITY)}
        candidates.sort(
            key=lambda row: (
                max(int(value) for value in row["resident_memory_bytes"]),
                sum(int(value) for value in row["resident_memory_bytes"]),
                priority.get(str(row["candidate_type"]), len(priority)),
                str(row["candidate_id"]),
            )
        )
        if candidates:
            return _admission(layer, candidates[0])
        return CandidateAdmission(
            layer,
            None,
            None,
            None,
            (),
            (),
            "PLACEMENT_INFEASIBLE",
            "no physically admitted production-native P8 candidate",
        )

    def admitted_candidates(self, layer: int) -> tuple[CandidateAdmission, ...]:
        """Return candidates in the frozen placement preference order."""

        if layer == 0:
            selected = self.choose_candidate(0)
            return (selected,) if selected.status == "PASS" else ()
        if layer not in range(1, 93):
            raise ValueError("Kimi K3 transformer layer must be in 0..92")
        candidates = [
            row
            for row in self.catalog["candidates"]
            if int(row.get("layer", -1)) == layer and _admitted_p8(row)
        ]
        priority = {
            "FULL_MIXED_STRIPE": -1,
            **{name: index for index, name in enumerate(PARTITION_PRIORITY)},
        }
        candidates.sort(
            key=lambda row: (
                priority.get(str(row["candidate_type"]), len(priority)),
                max(int(value) for value in row["resident_memory_bytes"]),
                sum(int(value) for value in row["resident_memory_bytes"]),
                str(row["candidate_id"]),
            )
        )
        return tuple(_admission(layer, row) for row in candidates)

    def candidate_preflight(self) -> tuple[CandidateAdmission, ...]:
        return tuple(self.choose_candidate(layer) for layer in range(93))


__all__ = ["CandidateAdmission", "CommodityK3Planner"]
