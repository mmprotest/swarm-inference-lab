"""Validity gates and the performance-independent E024 closure anchor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_022.io import atomic_write_json

from .freeze import Phase0Audit, audit_immutable_inputs
from .models import CommodityScenario
from .placement import CommodityPlacement

FIXED_CORRECTNESS_SCENARIO = CommodityScenario.COMMODITY_REGIONAL
FIXED_CORRECTNESS_PLACEMENT_KIND = "D_PLACEMENT"
FIXED_CORRECTNESS_BUDGET = 192
FIXED_CORRECTNESS_PLACEMENT_SHA256 = (
    "67703454689a63e5ff7541adb91610448d2d2fe1115e3a36bc5d86a04245ac6a"
)
FIXED_CORRECTNESS_FEASIBILITY = {
    96: False,
    128: False,
    160: False,
    192: True,
}


class ModelInvalidError(RuntimeError):
    """Raised when mandatory evidence cannot support an E024 conclusion."""


def require_phase0(repo_root: Path) -> Phase0Audit:
    audit = audit_immutable_inputs(repo_root)
    if audit.status != "PASS":
        raise ModelInvalidError(audit.reason or "Experiment 024 Phase 0 failed")
    return audit


def select_fixed_correctness_anchor(
    repo_root: Path,
) -> tuple[CommodityPlacement, dict[str, Any]]:
    """Select the fixed closure anchor without consulting performance evidence."""

    repo_root = repo_root.resolve()
    placement_path = repo_root / "artifacts/experiment-024/freeze/d-placements.json"
    audit_path = (
        repo_root
        / "artifacts/experiment-024/closure/correctness-anchor-audit.json"
    )
    payload = json.loads(placement_path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in payload["placements"]
        if row["scenario"] == FIXED_CORRECTNESS_SCENARIO.value
        and row["placement_kind"] == FIXED_CORRECTNESS_PLACEMENT_KIND
        and int(row["available_node_budget"]) in FIXED_CORRECTNESS_FEASIBILITY
    ]
    by_budget = {int(row["available_node_budget"]): row for row in rows}
    observed = [
        {
            "available_node_budget": budget,
            "expected_feasible": expected,
            "observed_feasible": (
                bool(by_budget[budget]["feasible"]) if budget in by_budget else None
            ),
            "placement_sha256": (
                str(by_budget[budget]["placement_sha256"])
                if budget in by_budget
                else None
            ),
        }
        for budget, expected in FIXED_CORRECTNESS_FEASIBILITY.items()
    ]
    unique_budgets = len(rows) == len(FIXED_CORRECTNESS_FEASIBILITY)
    feasibility_matches = unique_budgets and all(
        bool(by_budget[budget]["feasible"]) is expected
        for budget, expected in FIXED_CORRECTNESS_FEASIBILITY.items()
    )
    target_hash = (
        str(by_budget[FIXED_CORRECTNESS_BUDGET]["placement_sha256"])
        if FIXED_CORRECTNESS_BUDGET in by_budget
        else None
    )
    hash_matches = target_hash == FIXED_CORRECTNESS_PLACEMENT_SHA256
    audit: dict[str, Any] = {
        "schema_version": "experiment-024-fixed-correctness-anchor-v1",
        "status": "PASS" if feasibility_matches and hash_matches else "MODEL_INVALID",
        "selection_basis": "STRUCTURAL_PLACEMENT_FEASIBILITY_ONLY",
        "performance_metrics_consulted": False,
        "commercial_slo_consulted": False,
        "canonical_commercial_selection_consulted": False,
        "scenario": FIXED_CORRECTNESS_SCENARIO.value,
        "architecture": "SWARM_D_OPT",
        "placement_kind": FIXED_CORRECTNESS_PLACEMENT_KIND,
        "available_node_budget": FIXED_CORRECTNESS_BUDGET,
        "expected_placement_sha256": FIXED_CORRECTNESS_PLACEMENT_SHA256,
        "observed_placement_sha256": target_hash,
        "placement_hash_matches": hash_matches,
        "feasibility_sequence_matches": feasibility_matches,
        "feasibility_sequence": observed,
        "source": str(placement_path.relative_to(repo_root)).replace("\\", "/"),
    }
    atomic_write_json(audit_path, audit)
    if not feasibility_matches or not hash_matches:
        raise ModelInvalidError(
            "STOP E024 CLOSURE: MODEL_INVALID: fixed correctness anchor mismatch"
        )
    placement = CommodityPlacement.from_dict(by_budget[FIXED_CORRECTNESS_BUDGET])
    if not placement.feasible:
        raise ModelInvalidError(
            "STOP E024 CLOSURE: MODEL_INVALID: fixed correctness anchor is infeasible"
        )
    audit.update(
        {
            "active_node_count": placement.active_node_count,
            "active_node_ids": list(placement.active_node_ids),
            "whole_layer_transformer_layer_ids": list(
                placement.whole_layer_layer_ids
            ),
            "p8_transformer_layer_ids": list(placement.p8_layer_ids),
        }
    )
    atomic_write_json(audit_path, audit)
    return placement, audit


def run_fixed_anchor_two_token_correctness(repo_root: Path) -> dict[str, Any]:
    """Run unchanged E024 full correctness on the frozen structural anchor."""

    require_phase0(repo_root)
    placement, anchor_audit = select_fixed_correctness_anchor(repo_root)
    from .full_correctness import run_two_token_correctness

    result = run_two_token_correctness(repo_root, placement)
    result.update(
        {
            "schema_version": "experiment-024-two-token-correctness-closure-v1",
            "correctness_anchor": anchor_audit,
            "anchor_selection_basis": "STRUCTURAL_PLACEMENT_FEASIBILITY_ONLY",
            "performance_metrics_consulted_for_anchor": False,
            "commercial_slo_consulted_for_anchor": False,
            "canonical_commercial_selection_consulted_for_anchor": False,
        }
    )
    atomic_write_json(
        repo_root.resolve()
        / "artifacts/experiment-024/physical/two-token-full-correctness.json",
        result,
    )
    return result


__all__ = [
    "FIXED_CORRECTNESS_BUDGET",
    "FIXED_CORRECTNESS_FEASIBILITY",
    "FIXED_CORRECTNESS_PLACEMENT_KIND",
    "FIXED_CORRECTNESS_PLACEMENT_SHA256",
    "FIXED_CORRECTNESS_SCENARIO",
    "ModelInvalidError",
    "require_phase0",
    "run_fixed_anchor_two_token_correctness",
    "select_fixed_correctness_anchor",
]
