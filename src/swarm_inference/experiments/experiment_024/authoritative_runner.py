"""Resumable authoritative repaired E024 execution phases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_022.io import atomic_write_json, write_csv

from .correctness import ModelInvalidError, require_phase0
from .freeze import (
    COMMODITY_AVAILABLE_NODE_BUDGETS,
    COMMODITY_WORKER_MEMORY_BYTES,
)
from .full_correctness import run_two_token_correctness
from .models import Architecture, CommodityScenario
from .physical_d import validate_physical_d
from .placement import CommodityPlacement
from .reproducibility import (
    stage_a_reproducibility,
    stage_b_reproducibility,
)
from .service import E024ServiceTable
from .service_calibration import (
    CUDA_LIBRARY_RELATIVE_PATH,
    GROUPED_LIBRARY_RELATIVE_PATH,
    ORACLE_ROOT_RELATIVE_PATH,
    SHARD_LIBRARY_RELATIVE_PATH,
)
from .stage_a import run_stage_a
from .stage_b import (
    StageBResult,
    apply_global_correctness,
    canonical_medians,
    run_current_on_d_causal_checks,
    run_stage_b,
)
from .validation import (
    communication_reconciliation,
    cost_reconciliation_rows,
    create_code_freeze,
    memory_reconciliation_rows,
    placement_reconciliation_rows,
    token_semantics_audit,
    verify_code_freeze,
)

ARTIFACT_ROOT = Path("artifacts/experiment-024")


def _root(repo_root: Path) -> Path:
    return repo_root.resolve() / ARTIFACT_ROOT


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _code_freeze(repo_root: Path) -> dict[str, Any]:
    path = _root(repo_root) / "freeze/code-freeze.json"
    if not path.is_file():
        raise ModelInvalidError("repaired code freeze does not exist")
    payload = _read_json(path)
    if payload.get("status") != "PASS" or not verify_code_freeze(repo_root, payload):
        raise ModelInvalidError("repaired code freeze hash verification failed")
    return payload


def run_code_freeze(repo_root: Path) -> dict[str, Any]:
    require_phase0(repo_root)
    calibration = _read_json(_root(repo_root) / "calibration/calibration-summary.json")
    if calibration.get("status") != "PASS":
        raise ModelInvalidError("calibration must pass before repaired code freeze")
    payload = create_code_freeze(repo_root)
    atomic_write_json(_root(repo_root) / "freeze/code-freeze.json", payload)
    return payload


def run_physical_d_phase(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    _code_freeze(repo_root)
    payload = validate_physical_d(
        checkpoint=Path(r"F:\models\Kimi-K3"),
        cuda_library=(repo_root / CUDA_LIBRARY_RELATIVE_PATH).resolve(),
        shard_library=(repo_root / SHARD_LIBRARY_RELATIVE_PATH).resolve(),
        grouped_library=(repo_root / GROUPED_LIBRARY_RELATIVE_PATH).resolve(),
        oracle_root=(repo_root / ORACLE_ROOT_RELATIVE_PATH).resolve(),
    )
    atomic_write_json(
        _root(repo_root) / "physical/composed-block-correctness.json",
        payload,
    )
    token_audit = token_semantics_audit()
    atomic_write_json(
        _root(repo_root) / "validation/token-semantics.json",
        token_audit,
    )
    return {"physical_d": payload, "token_semantics": token_audit}


def run_stage_a_phase(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    _code_freeze(repo_root)
    service = E024ServiceTable(repo_root)
    result = run_stage_a(service)
    root = _root(repo_root)
    write_csv(root / "stage-a/block-results.csv", result.rows)
    primary = tuple(
        row for row in result.rows if int(row["concurrency"]) == 32
    )
    write_csv(root / "stage-a/c32-primary-results.csv", primary)
    write_csv(root / "stage-a/gap-closure.csv", result.gap_closure_rows)
    family_rows = []
    for scenario in CommodityScenario:
        values = [
            row
            for row in result.gap_closure_rows
            if row["scenario"] == scenario.value
        ]
        family_rows.append(
            {
                "scenario": scenario.value,
                "cell_count": len(values),
                "median_gap_closure_percent": sorted(
                    float(row["latency_gap_closure_percent"]) for row in values
                )[len(values) // 2],
                "maximum_gap_closure_percent": max(
                    float(row["latency_gap_closure_percent"]) for row in values
                ),
                "maximum_d_regression_percent": max(
                    float(row["d_regression_percent"]) for row in values
                ),
            }
        )
    write_csv(root / "stage-a/family-summary.csv", family_rows)
    summary = {
        "schema_version": "experiment-024-stage-a-summary-v1",
        "status": "PASS",
        "cell_count": len(result.rows),
        "gap_cell_count": len(result.gap_closure_rows),
        "median_gap_closure_percent": result.median_gap_closure_percent,
        "maximum_gap_closure_percent": result.maximum_gap_closure_percent,
        "maximum_d_regression_percent": result.d_max_regression_percent,
        "authoritative_rows": list(result.rows),
    }
    atomic_write_json(root / "stage-a/stage-a-summary.json", summary)
    reconciliation = communication_reconciliation()
    atomic_write_json(
        root / "validation/communication-reconciliation.json",
        reconciliation,
    )
    write_csv(
        root / "communication/task-graph-accounting.csv",
        tuple(
            {
                "scenario": row["scenario"],
                "layer": row["layer"],
                "rows": row["rows"],
                "concurrency": row["concurrency"],
                "arm": row["arm"],
                "measured_blocks": row["measured_blocks"],
                "moe_network_bytes": row["moe_network_bytes"],
                "moe_network_bytes_per_row": row["moe_network_bytes_per_row"],
                "moe_network_messages": row["moe_network_messages"],
                "status": row["status"],
            }
            for row in result.rows
        ),
    )
    return summary


def _write_placements(root: Path, result: StageBResult) -> None:
    for placement_kind, filename in (
        ("CURRENT_PLACEMENT", "current-placements.json"),
        ("D_PLACEMENT", "d-placements.json"),
    ):
        placements = [
            placement.as_dict()
            for placement in result.placements
            if placement.placement_kind == placement_kind
        ]
        atomic_write_json(
            root / "freeze" / filename,
            {
                "schema_version": "experiment-024-frozen-placements-v2",
                "status": "PASS",
                "placement_kind": placement_kind,
                "placement_count": len(placements),
                "placements": placements,
            },
        )


def run_stage_b_phase(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    _code_freeze(repo_root)
    service = E024ServiceTable(repo_root)
    result = run_stage_b(repo_root, service)
    root = _root(repo_root)
    atomic_write_json(
        root / "freeze/reference-architecture.json",
        result.reference.as_dict(),
    )
    _write_placements(root, result)
    atomic_write_json(
        root / "freeze/commodity-pool-definition.json",
        {
            "schema_version": "experiment-024-commodity-pool-v2",
            "status": "PASS",
            "available_node_budgets": list(COMMODITY_AVAILABLE_NODE_BUDGETS),
            "memory_bytes_per_node": COMMODITY_WORKER_MEMORY_BYTES,
            "memory_gib_per_node": COMMODITY_WORKER_MEMORY_BYTES / 2**30,
            "compute_multipliers_cycled": [1.0, 0.8, 0.6, 0.4],
            "reliability": 1.0,
            "ordinary_layer_zero_worker": True,
            "special_dense_host": False,
        },
    )
    write_csv(root / "stage-b/reference-serving-results.csv", result.reference_rows)
    write_csv(root / "stage-b/screening-results.csv", result.screening_rows)
    write_csv(root / "stage-b/candidate-budget-selection.csv", result.selected_budget_rows)
    write_csv(root / "stage-b/decode-serving-results.csv", result.decode_rows)
    write_csv(root / "stage-b/performance-cost-frontier.csv", result.frontier_rows)
    write_csv(
        root / "stage-b/current-frontier.csv",
        tuple(
            row
            for row in result.frontier_rows
            if row["architecture"] == Architecture.SWARM_CURRENT_OPT.value
        ),
    )
    write_csv(
        root / "stage-b/d-frontier.csv",
        tuple(
            row
            for row in result.frontier_rows
            if row["architecture"] == Architecture.SWARM_D_OPT.value
        ),
    )
    write_csv(root / "stage-b/combined-frontier.csv", result.frontier_rows)
    atomic_write_json(
        root / "stage-b/canonical-points.json",
        {
            "status": "PENDING_GLOBAL_CORRECTNESS",
            "points": list(result.canonical_rows),
        },
    )
    write_csv(
        root / "validation/placement-reconciliation.csv",
        placement_reconciliation_rows(result.placements),
    )
    write_csv(
        root / "validation/memory-reconciliation.csv",
        memory_reconciliation_rows(result.placements),
    )
    causal = run_current_on_d_causal_checks(repo_root, service, result)
    write_csv(root / "stage-b/current-vs-d.csv", causal)
    stage_b_summary = {
        "schema_version": "experiment-024-stage-b-summary-v2",
        "status": "PENDING_GLOBAL_CORRECTNESS",
        "reference_node_count": result.reference.node_count,
        "reference_c1_p95_token_latency_ms": (
            result.reference_c1_p95_token_latency_ms
        ),
        "primary_token_latency_budget_ms": result.primary_token_latency_budget_ms,
        "reference_slo_output_tps": result.reference_slo_output_tps,
        "placement_count": len(result.placements),
        "feasible_placement_count": sum(
            placement.feasible for placement in result.placements
        ),
        "screening_row_count": len(result.screening_rows),
        "decode_row_count": len(result.decode_rows),
        "frontier_row_count": len(result.frontier_rows),
        "canonical_points": list(result.canonical_rows),
        "causal_checks": list(causal),
    }
    atomic_write_json(root / "stage-b/stage-b-summary.json", stage_b_summary)
    return stage_b_summary


def load_frozen_placements(repo_root: Path) -> tuple[CommodityPlacement, ...]:
    root = _root(repo_root)
    values = []
    for filename in ("current-placements.json", "d-placements.json"):
        payload = _read_json(root / "freeze" / filename)
        values.extend(CommodityPlacement.from_dict(row) for row in payload["placements"])
    return tuple(values)


def _canonical_rows(repo_root: Path) -> tuple[dict[str, Any], ...]:
    payload = _read_json(_root(repo_root) / "stage-b/canonical-points.json")
    return tuple(payload["points"])


def run_two_token_phase(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    _code_freeze(repo_root)
    canonical = next(
        row
        for row in _canonical_rows(repo_root)
        if row["scenario"] == CommodityScenario.COMMODITY_REGIONAL.value
    )
    if canonical["architecture"] != Architecture.SWARM_D_OPT.value:
        raise ModelInvalidError(
            "canonical REGIONAL point is not D; canonical D correctness cannot be substituted"
        )
    placements = load_frozen_placements(repo_root)
    placement = next(
        value
        for value in placements
        if value.scenario is CommodityScenario.COMMODITY_REGIONAL
        and value.placement_kind == "D_PLACEMENT"
        and value.available_node_budget == int(canonical["available_node_budget"])
    )
    payload = run_two_token_correctness(repo_root, placement)
    atomic_write_json(
        _root(repo_root) / "physical/two-token-full-correctness.json",
        payload,
    )
    return payload


def _restore_stage_b_result(repo_root: Path) -> StageBResult:
    root = _root(repo_root)
    stage_b = _read_json(root / "stage-b/stage-b-summary.json")
    reference_payload = _read_json(root / "freeze/reference-architecture.json")
    # Only the fields used by correctness finalization/reproducibility are
    # restored; reference construction and placement optimization are not rerun.
    from .placement import LayerPlacement
    from .reference_architecture import ReferenceArchitecture
    from .task_graph import RuntimeNode

    reference = ReferenceArchitecture(
        node_count=int(reference_payload["node_count"]),
        nodes=tuple(
            RuntimeNode(f"reference-{index:03d}", index, 1.0)
            for index in range(int(reference_payload["node_count"]))
        ),
        endpoint_node_ids=tuple(reference_payload["endpoint_node_ids"]),
        endpoint_memory_by_node={
            str(key): int(value)
            for key, value in reference_payload["endpoint_memory_by_node"].items()
        },
        assignments=tuple(
            LayerPlacement.from_dict(row) for row in reference_payload["assignments"]
        ),
        memory_used_by_node={
            str(key): int(value)
            for key, value in reference_payload["memory_used_by_node"].items()
        },
    )
    def csv_rows(path: Path) -> tuple[dict[str, Any], ...]:
        import csv

        with path.open(encoding="utf-8", newline="") as handle:
            return tuple(csv.DictReader(handle))

    return StageBResult(
        reference=reference,
        reference_rows=csv_rows(root / "stage-b/reference-serving-results.csv"),
        reference_c1_p95_token_latency_ms=float(
            stage_b["reference_c1_p95_token_latency_ms"]
        ),
        primary_token_latency_budget_ms=float(
            stage_b["primary_token_latency_budget_ms"]
        ),
        reference_slo_output_tps=float(stage_b["reference_slo_output_tps"]),
        placements=load_frozen_placements(repo_root),
        screening_rows=csv_rows(root / "stage-b/screening-results.csv"),
        selected_budget_rows=csv_rows(
            root / "stage-b/candidate-budget-selection.csv"
        ),
        decode_rows=csv_rows(root / "stage-b/decode-serving-results.csv"),
        frontier_rows=csv_rows(root / "stage-b/performance-cost-frontier.csv"),
        canonical_rows=_canonical_rows(repo_root),
    )


def finalize_commercial_gates(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    correctness = _read_json(
        _root(repo_root) / "physical/two-token-full-correctness.json"
    )
    result = apply_global_correctness(
        _restore_stage_b_result(repo_root),
        global_correctness_pass=correctness.get("status") == "PASS",
    )
    root = _root(repo_root)
    write_csv(root / "stage-b/performance-cost-frontier.csv", result.frontier_rows)
    write_csv(root / "stage-b/combined-frontier.csv", result.frontier_rows)
    write_csv(
        root / "stage-b/current-frontier.csv",
        tuple(
            row
            for row in result.frontier_rows
            if row["architecture"] == Architecture.SWARM_CURRENT_OPT.value
        ),
    )
    write_csv(
        root / "stage-b/d-frontier.csv",
        tuple(
            row
            for row in result.frontier_rows
            if row["architecture"] == Architecture.SWARM_D_OPT.value
        ),
    )
    atomic_write_json(
        root / "stage-b/canonical-points.json",
        {"status": "PASS", "points": list(result.canonical_rows)},
    )
    write_csv(
        root / "validation/cost-reconciliation.csv",
        cost_reconciliation_rows(result.frontier_rows),
    )
    medians = canonical_medians(result)
    scenario_rows = []
    for row in result.canonical_rows:
        scenario_rows.append(
            {
                **row,
                "commercially_below_kimi": (
                    float(row["cost_per_M_at_0_15"]) < 15.0
                ),
            }
        )
    write_csv(root / "stage-b/scenario-summary.csv", scenario_rows)
    write_csv(root / "analysis/performance-cost-summary.csv", scenario_rows)
    write_csv(
        root / "analysis/scenario-gates.csv",
        tuple(
            {
                "scenario": row["scenario"],
                "slo_feasible": row["slo_feasible"],
                "cost_per_M_at_0_15": row["cost_per_M_at_0_15"],
                "layer_zero_candidate_id": row["layer_zero_candidate_id"],
                "p8_layer_count": row["p8_layer_count"],
                "whole_layer_only_commodity_model_feasible": row[
                    "whole_layer_only_commodity_model_feasible"
                ],
                "p8_required_whole_layer_incapable_compute_share": row[
                    "p8_required_whole_layer_incapable_compute_share"
                ],
                "global_correctness_pass": correctness["status"] == "PASS",
                "scenario_wedge_pass": row["scenario_wedge_pass"],
            }
            for row in scenario_rows
        ),
    )
    write_csv(
        root / "analysis/contributor-payout-frontier.csv",
        tuple(
            {
                "scenario": row["scenario"],
                "architecture": row["architecture"],
                "active_node_count": row["active_node_count"],
                "output_tokens_per_second": row[
                    "aggregate_output_tokens_per_second"
                ],
                "target_cost_per_M": target,
                "max_uniform_payout_per_active_node_hour": row[
                    f"max_uniform_payout_at_{str(target).rstrip('0').rstrip('.').replace('.', '_')}"
                ],
            }
            for row in scenario_rows
            for target in (15.0, 12.0, 9.0, 7.5, 5.0, 3.0)
        ),
    )
    write_csv(
        root / "analysis/cost-sensitivity.csv",
        tuple(
            {
                "scenario": row["scenario"],
                "architecture": row["architecture"],
                "payout_per_active_node_hour": payout,
                "cost_per_M": row[
                    {
                        0.05: "cost_per_M_at_0_05",
                        0.10: "cost_per_M_at_0_10",
                        0.15: "cost_per_M_at_0_15",
                        0.25: "cost_per_M_at_0_25",
                        0.50: "cost_per_M_at_0_50",
                    }[payout]
                ],
            }
            for row in scenario_rows
            for payout in (0.05, 0.10, 0.15, 0.25, 0.50)
        ),
    )
    payload = {
        "status": "PASS",
        "scenario_wedge_count": sum(
            bool(row["scenario_wedge_pass"]) for row in scenario_rows
        ),
        "canonical_points": scenario_rows,
        **medians,
    }
    atomic_write_json(root / "stage-b/economics-summary.json", payload)
    return payload


def run_reproducibility_phase(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    _code_freeze(repo_root)
    root = _root(repo_root)
    service = E024ServiceTable(repo_root)
    stage_a_summary = _read_json(root / "stage-a/stage-a-summary.json")
    stage_a_result = stage_a_reproducibility(
        service,
        tuple(stage_a_summary["authoritative_rows"]),
    )
    canonical = _canonical_rows(repo_root)
    stage_b_result = stage_b_reproducibility(
        repo_root,
        service,
        canonical,
        load_frozen_placements(repo_root),
    )
    atomic_write_json(
        root / "validation/stage-a-reproducibility.json",
        stage_a_result,
    )
    atomic_write_json(
        root / "validation/stage-b-reproducibility.json",
        stage_b_result,
    )
    return {"stage_a": stage_a_result, "stage_b": stage_b_result}


__all__ = [
    "ARTIFACT_ROOT",
    "finalize_commercial_gates",
    "load_frozen_placements",
    "run_code_freeze",
    "run_physical_d_phase",
    "run_reproducibility_phase",
    "run_stage_a_phase",
    "run_stage_b_phase",
    "run_two_token_phase",
]
