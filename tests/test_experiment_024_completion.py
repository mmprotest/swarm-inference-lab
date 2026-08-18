"""Completion checks for the corrected Experiment 024 architecture."""

from __future__ import annotations

import json
from dataclasses import replace
from functools import cache
from pathlib import Path

import pytest

from swarm_inference.experiments.experiment_024.commodity_planner import (
    CommodityK3Planner,
)
from swarm_inference.experiments.experiment_024.economics import cost_per_million
from swarm_inference.experiments.experiment_024.freeze import (
    COMMODITY_WORKER_MEMORY_BYTES,
    LAYER_ZERO_WHOLE_CANDIDATE_ID,
    LAYER_ZERO_WHOLE_RESIDENT_BYTES,
    P8_REQUIRED_LAYER_IDS,
    audit_immutable_inputs,
)
from swarm_inference.experiments.experiment_024.models import CommodityScenario
from swarm_inference.experiments.experiment_024.placement import (
    CommodityPlacement,
    CommodityPlacementBuilder,
    validate_commodity_architecture,
)
from swarm_inference.experiments.experiment_024.service import (
    E024ServiceTable,
    ServiceKey,
)
from swarm_inference.experiments.experiment_024.service_calibration import (
    CUDA_LIBRARY_RELATIVE_PATH,
    _isolated_request,
    validate_dense_layer0_service,
)
from swarm_inference.experiments.experiment_024.stage_b import (
    StageBResult,
    apply_global_correctness,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@cache
def _placement(kind: str) -> CommodityPlacement:
    return CommodityPlacementBuilder(REPO_ROOT).build(
        scenario=CommodityScenario.COMMODITY_REGIONAL,
        available_node_budget=192,
        placement_kind=kind,
    )


def test_layer_zero_whole_admitted_and_fits_commodity() -> None:
    audit = audit_immutable_inputs(REPO_ROOT)
    assert audit.layer_zero_whole_candidate_id == LAYER_ZERO_WHOLE_CANDIDATE_ID
    assert audit.layer_zero_whole_candidate_admitted
    assert audit.layer_zero_whole_resident_bytes == LAYER_ZERO_WHOLE_RESIDENT_BYTES
    assert LAYER_ZERO_WHOLE_RESIDENT_BYTES < COMMODITY_WORKER_MEMORY_BYTES
    assert audit.layer_zero_whole_fits_commodity


def test_all_non_dense_layers_are_whole_infeasible() -> None:
    audit = audit_immutable_inputs(REPO_ROOT)
    assert audit.whole_layer_resident_bytes_by_layer[1] > COMMODITY_WORKER_MEMORY_BYTES
    assert audit.whole_layer_infeasible_layer_ids_on_commodity == P8_REQUIRED_LAYER_IDS
    assert len(audit.whole_layer_infeasible_layer_ids_on_commodity) == 92


def test_all_non_dense_layers_have_four_admitted_p8_candidates() -> None:
    audit = audit_immutable_inputs(REPO_ROOT)
    assert set(audit.p8_admitted_candidate_counts_by_layer) == set(
        P8_REQUIRED_LAYER_IDS
    )
    assert set(audit.p8_admitted_candidate_counts_by_layer.values()) == {4}


def test_repaired_phase0_passes_complete_candidate_coverage() -> None:
    audit = audit_immutable_inputs(REPO_ROOT)
    assert audit.status == "PASS"
    assert audit.mandatory_failure_id is None
    assert audit.complete_commodity_candidate_coverage
    assert audit.complete_commodity_architecture_candidate_coverage
    assert audit.whole_layer_feasible_layer_ids_on_commodity == (0,)
    assert not audit.whole_layer_only_commodity_model_feasible


def test_planner_chooses_layer_zero_whole_and_layer_one_p8() -> None:
    planner = CommodityK3Planner(REPO_ROOT)
    layer_zero = planner.choose_candidate(0)
    layer_one = planner.choose_candidate(1)
    assert layer_zero.candidate_id == LAYER_ZERO_WHOLE_CANDIDATE_ID
    assert layer_zero.candidate_type == "WHOLE_LAYER"
    assert layer_zero.degree == 1
    assert layer_one.candidate_type != "WHOLE_LAYER"
    assert layer_one.degree == 8
    assert layer_one.candidate_id == "layer-01:FULL_MIXED_STRIPE:p8"


def test_commodity_architecture_has_exactly_one_whole_and_92_p8_layers() -> None:
    placement = _placement("D_PLACEMENT")
    assert placement.feasible, placement.infeasible_reason
    assert placement.whole_layer_layer_ids == (0,)
    assert placement.p8_layer_ids == P8_REQUIRED_LAYER_IDS
    assert len(placement.assignments) == 93


def test_whole_layer_only_commodity_model_is_infeasible() -> None:
    audit = audit_immutable_inputs(REPO_ROOT)
    assert not audit.whole_layer_only_commodity_model_feasible
    path = (
        REPO_ROOT
        / "artifacts/experiment-024/validation/commodity-whole-layer-feasibility.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["whole_layer_feasible_transformer_layer_ids"] == [0]
    assert value["whole_layer_only_complete_k3_placement"] == "INFEASIBLE"


def test_layer_zero_worker_is_active_and_costed_once() -> None:
    placement = _placement("D_PLACEMENT")
    layer_zero_owner = placement.assignments[0].node_ids[0]
    assert layer_zero_owner in placement.active_node_ids
    expected_hourly_cost = placement.active_node_count * 0.15
    assert expected_hourly_cost > 0
    assert cost_per_million(expected_hourly_cost, 100.0) == pytest.approx(
        expected_hourly_cost * 1_000_000 / (100.0 * 3600)
    )
    assert len(set(placement.active_node_ids)) == placement.active_node_count


def test_d_does_not_transform_layer_zero() -> None:
    current = _placement("CURRENT_PLACEMENT")
    transformed = _placement("D_PLACEMENT")
    assert current.assignments[0] == transformed.assignments[0]


def test_placement_is_deterministic_before_freeze() -> None:
    first = _placement("D_PLACEMENT")
    second = CommodityPlacementBuilder(REPO_ROOT).build(
        scenario=CommodityScenario.COMMODITY_REGIONAL,
        available_node_budget=192,
        placement_kind="D_PLACEMENT",
    )
    assert first.placement_sha256 == second.placement_sha256
    assert first.as_dict() == second.as_dict()


def test_dense_layer_zero_service_receipt_and_samples() -> None:
    services = {
        str(rows): {
            "physical_service": {"p50_ms": float(rows)},
            "sample_count": 200,
            "all_outputs_finite": True,
            "timed_checkpoint_reads": 0,
        }
        for rows in (1, 2, 4)
    }
    payload = {
        "status": "PASS",
        "candidate_id": LAYER_ZERO_WHOLE_CANDIDATE_ID,
        "candidate_type": "WHOLE_LAYER",
        "degree": 1,
        "production_native_dispatch": True,
        "timed_checkpoint_reads": 0,
        "resident_weights": True,
        "resident_memory_bytes": LAYER_ZERO_WHOLE_RESIDENT_BYTES,
        "physical_resident_device_bytes": 1_000_000_000,
        "maximum_observed_device_bytes": 1_100_000_000,
        "commodity_worker_memory_bytes": COMMODITY_WORKER_MEMORY_BYTES,
        "memory_feasible": True,
        "service_by_rows": services,
    }
    samples = [
        {
            "candidate_id": LAYER_ZERO_WHOLE_CANDIDATE_ID,
            "rows": rows,
            "wall_ms": float(rows),
            "finite_output": True,
            "production_native_dispatch": True,
            "timed_checkpoint_reads": 0,
            "resident_memory_bytes": LAYER_ZERO_WHOLE_RESIDENT_BYTES,
            "physical_resident_device_bytes": 1_000_000_000,
            "maximum_observed_device_bytes": 1_100_000_000,
            "commodity_worker_memory_bytes": COMMODITY_WORKER_MEMORY_BYTES,
        }
        for rows in (1, 2, 4)
        for _ in range(200)
    ]
    validate_dense_layer0_service(payload, samples)


def test_dense_calibration_builds_a_non_endpoint_layer_zero_request() -> None:
    request = _isolated_request(
        Path(r"F:\models\Kimi-K3"),
        REPO_ROOT / CUDA_LIBRARY_RELATIVE_PATH,
        layer=0,
        maximum_context=228,
    )
    assert request.assignment.layer_ids == (0,)
    assert request.assignment.weight_bytes == 2_341_213_184
    assert not request.assignment.owns_embeddings
    assert not request.assignment.owns_final_norm
    assert not request.assignment.owns_output_projection
    assert request.fast_path_mode == "verification-major"
    assert request.fast_path_batch_bucket == 17


def test_dense_layer_zero_compute_multiplier_scaling_is_exact() -> None:
    table = E024ServiceTable.__new__(E024ServiceTable)
    table._values = {ServiceKey(0, "whole_layer", 1, 4): 12.0}
    assert table.dense_layer_zero_ms(4, 1.0) == pytest.approx(12.0)
    assert table.dense_layer_zero_ms(4, 0.4) == pytest.approx(30.0)


def test_csv_boolean_round_trip_does_not_change_wedge_logic() -> None:
    row = {
        "cost_per_M_at_0_15": "10.0",
        "slo_feasible": "True",
        "layer_zero_candidate_id": LAYER_ZERO_WHOLE_CANDIDATE_ID,
        "p8_layer_count": "92",
        "whole_layer_layer_count": "1",
        "whole_layer_only_commodity_model_feasible": "False",
        "p8_required_whole_layer_incapable_compute_share": "1.0",
        "canonical_scenario_point": "True",
    }
    result = StageBResult(
        reference=None,  # type: ignore[arg-type]
        reference_rows=(),
        reference_c1_p95_token_latency_ms=1.0,
        primary_token_latency_budget_ms=4.0,
        reference_slo_output_tps=1.0,
        placements=(),
        screening_rows=(),
        selected_budget_rows=(),
        decode_rows=(),
        frontier_rows=(row,),
        canonical_rows=(),
    )
    final = apply_global_correctness(result, global_correctness_pass=True)
    assert len(final.canonical_rows) == 1
    assert final.canonical_rows[0]["scenario_wedge_pass"] is True


def test_csv_false_canonical_flag_remains_false() -> None:
    row = {
        "cost_per_M_at_0_15": "10.0",
        "slo_feasible": "True",
        "layer_zero_candidate_id": LAYER_ZERO_WHOLE_CANDIDATE_ID,
        "p8_layer_count": "92",
        "whole_layer_layer_count": "1",
        "whole_layer_only_commodity_model_feasible": "False",
        "p8_required_whole_layer_incapable_compute_share": "1.0",
        "canonical_scenario_point": "False",
    }
    result = StageBResult(
        reference=None,  # type: ignore[arg-type]
        reference_rows=(),
        reference_c1_p95_token_latency_ms=1.0,
        primary_token_latency_budget_ms=4.0,
        reference_slo_output_tps=1.0,
        placements=(),
        screening_rows=(),
        selected_budget_rows=(),
        decode_rows=(),
        frontier_rows=(row,),
        canonical_rows=(),
    )
    final = apply_global_correctness(result, global_correctness_pass=True)
    assert final.canonical_rows == ()


def test_architecture_gate_rejects_whole_non_dense_layer() -> None:
    placement = _placement("D_PLACEMENT")
    bad_layer = replace(
        placement.assignments[1],
        candidate_type="WHOLE_LAYER",
        degree=1,
        node_ids=(placement.assignments[1].node_ids[0],),
        resident_memory_bytes=(placement.assignments[1].resident_memory_bytes[0],),
        checkpoint_bytes=(placement.assignments[1].checkpoint_bytes[0],),
    )
    bad = replace(
        placement,
        assignments=(placement.assignments[0], bad_layer, *placement.assignments[2:]),
    )
    with pytest.raises(RuntimeError, match="only layer 0 whole"):
        validate_commodity_architecture(bad)


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
