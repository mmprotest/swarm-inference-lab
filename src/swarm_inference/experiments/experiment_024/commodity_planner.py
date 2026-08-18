"""Deterministic P8-only commodity-placement admission checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .freeze import CANDIDATE_CATALOG_RELATIVE_PATH, DEGREE

PARTITION_PRIORITY = (
    "ATTENTION_PROJECTION_SHARD",
    "EXPERT_SHARD",
    "WHOLE_EXPERT",
)


@dataclass(frozen=True, slots=True)
class CandidateAdmission:
    layer: int
    candidate_id: str | None
    status: str
    reason: str | None


def _admitted(candidate: dict[str, Any]) -> bool:
    return (
        int(candidate.get("degree", -1)) == DEGREE
        and candidate.get("headline_eligible") is True
        and candidate.get("production_native_binding") is True
        and candidate.get("correctness_status") == "PASS"
        and list(candidate.get("chunk_sizes_physically_validated", [])) == [1, 2, 4]
        and candidate.get("candidate_type") != "WHOLE_LAYER"
    )


class P8OnlyCommodityPlanner:
    """Apply the frozen candidate rule before any group search."""

    def __init__(self, repo_root: Path) -> None:
        path = repo_root.resolve() / CANDIDATE_CATALOG_RELATIVE_PATH
        self.catalog = json.loads(path.read_text(encoding="utf-8"))

    def choose_candidate(self, layer: int) -> CandidateAdmission:
        candidates = [
            row
            for row in self.catalog["candidates"]
            if int(row.get("layer", -1)) == layer and _admitted(row)
        ]
        full_mixed = [
            row for row in candidates if row["candidate_type"] == "FULL_MIXED_STRIPE"
        ]
        if full_mixed:
            chosen = min(full_mixed, key=lambda row: str(row["candidate_id"]))
            return CandidateAdmission(layer, str(chosen["candidate_id"]), "PASS", None)
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
            return CandidateAdmission(
                layer, str(candidates[0]["candidate_id"]), "PASS", None
            )
        return CandidateAdmission(
            layer,
            None,
            "PLACEMENT_INFEASIBLE",
            "no physically admitted production-native P8 candidate",
        )

    def candidate_preflight(self) -> tuple[CandidateAdmission, ...]:
        return tuple(self.choose_candidate(layer) for layer in range(93))


__all__ = ["CandidateAdmission", "P8OnlyCommodityPlanner"]
