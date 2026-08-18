"""Fail-closed completion checks for the immutable E024 input set."""

from __future__ import annotations

import json
from pathlib import Path

from swarm_inference.experiments.experiment_024.commodity_planner import (
    P8OnlyCommodityPlanner,
)
from swarm_inference.experiments.experiment_024.freeze import audit_immutable_inputs

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_phase0_detects_missing_layer_zero_p8_admission() -> None:
    audit = audit_immutable_inputs(REPO_ROOT)
    assert audit.status == "MODEL_INVALID"
    assert audit.mandatory_failure_id == (
        "NO_PRODUCTION_NATIVE_P8_CANDIDATE_FOR_LAYER_0"
    )
    assert audit.e022_inventory_count == 27
    assert audit.e022_inventory_hashes_valid
    assert audit.e023_final_verdict == "NO_WEDGE"
    assert audit.transformer_layer_count == 93
    assert audit.layer_zero_p8_candidate_count == 2
    assert audit.layer_zero_admitted_p8_candidate_count == 0
    assert not audit.complete_p8_only_placement_possible


def test_p8_planner_preflight_is_infeasible_at_layer_zero() -> None:
    preflight = P8OnlyCommodityPlanner(REPO_ROOT).candidate_preflight()
    assert len(preflight) == 93
    assert preflight[0].status == "PLACEMENT_INFEASIBLE"
    assert preflight[0].candidate_id is None
    assert all(row.status == "PASS" for row in preflight[1:])


def test_invalid_summary_and_truth_table_are_consistent() -> None:
    summary_path = REPO_ROOT / "artifacts/experiment-024/summary.json"
    truth_path = REPO_ROOT / "artifacts/experiment-024/truth-table.json"
    if not summary_path.exists() or not truth_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    assert summary["final_verdict"] == "MODEL_INVALID"
    assert truth["final verdict"] == "MODEL_INVALID"
    assert summary["scenario_wedge_count"] == 0
    assert truth["complete P8-only candidate coverage"] is False


def test_historical_hash_audit_records_every_frozen_input() -> None:
    path = REPO_ROOT / "artifacts/experiment-024/freeze/e022-input-hashes.json"
    if not path.exists():
        return
    audit = json.loads(path.read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    assert audit["inventory_count"] == 27
    assert audit["all_inventory_hashes_valid"]
    assert len(audit["inventories"]) == 27
    assert all(row["match"] for row in audit["inventories"])
    assert audit["placements"]["file_count"] == 135
    assert len(audit["placements"]["files"]) == 135
    assert len(audit["e023_context"]) == 3
