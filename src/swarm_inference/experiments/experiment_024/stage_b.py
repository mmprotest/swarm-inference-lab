"""Stage B reference, commodity serving, frontier, and economics pipeline."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .decode_serving_engine import DecodeServingEngine
from .economics import (
    commercial_metrics,
    cost_per_million,
    max_compute_weighted_payout_per_hour,
    max_uniform_payout_per_active_node_hour,
    scenario_wedge_pass,
)
from .freeze import (
    COMMODITY_AVAILABLE_NODE_BUDGETS,
    DECODE_CONCURRENCY_LEVELS,
    LAYER_ZERO_WHOLE_CANDIDATE_ID,
    PAYOUT_SENSITIVITY_USD_PER_ACTIVE_NODE_HOUR,
    PRIMARY_CONTRIBUTOR_PAYOUT_USD_PER_ACTIVE_NODE_HOUR,
    PRIMARY_DECODE_SLO_MULTIPLIER,
    TARGET_COST_LEVELS_USD_PER_M,
)
from .models import Architecture, CommodityScenario, StageAArm
from .placement import CommodityPlacement, CommodityPlacementBuilder
from .reference_architecture import (
    REFERENCE_TRANSIENT_BYTES,
    ReferenceArchitecture,
    build_reference_architecture,
)
from .service import E024ServiceTable
from .task_graph import commodity_topology

SCREENING_CONCURRENCY = (4, 16, 64)


@dataclass(frozen=True, slots=True)
class StageBResult:
    reference: ReferenceArchitecture
    reference_rows: tuple[dict[str, Any], ...]
    reference_c1_p95_token_latency_ms: float
    primary_token_latency_budget_ms: float
    reference_slo_output_tps: float
    placements: tuple[CommodityPlacement, ...]
    screening_rows: tuple[dict[str, Any], ...]
    selected_budget_rows: tuple[dict[str, Any], ...]
    decode_rows: tuple[dict[str, Any], ...]
    frontier_rows: tuple[dict[str, Any], ...]
    canonical_rows: tuple[dict[str, Any], ...]


def build_commodity_placements(repo_root: Path) -> tuple[CommodityPlacement, ...]:
    builder = CommodityPlacementBuilder(repo_root)
    return tuple(
        builder.build(
            scenario=scenario,
            available_node_budget=budget,
            placement_kind=placement_kind,
        )
        for scenario in CommodityScenario
        for placement_kind in ("CURRENT_PLACEMENT", "D_PLACEMENT")
        for budget in COMMODITY_AVAILABLE_NODE_BUDGETS
    )


def _reference_rows(
    repo_root: Path,
    service: E024ServiceTable,
    reference: ReferenceArchitecture,
) -> tuple[dict[str, Any], ...]:
    engine = DecodeServingEngine()
    active = tuple(
        node.node_id
        for node in reference.nodes
        if reference.memory_used_by_node[node.node_id] > 0
    )
    resident = sum(reference.memory_used_by_node.values())
    transient = len(active) * REFERENCE_TRANSIENT_BYTES
    return tuple(
        engine.execute(
            repo_root=repo_root,
            topology=reference.topology(),
            service=service,
            active_sequence_count=concurrency,
            arm=StageAArm.A_CURRENT,
            available_node_budget=None,
            placement_sha256=reference.architecture_sha256,
            active_node_ids=active,
            resident_model_bytes=resident,
            peak_transient_bytes=transient,
        )
        for concurrency in DECODE_CONCURRENCY_LEVELS
    )


def _execute_commodity(
    repo_root: Path,
    service: E024ServiceTable,
    placement: CommodityPlacement,
    concurrency: int,
    *,
    execution_architecture: Architecture | None = None,
) -> dict[str, Any]:
    if not placement.feasible:
        raise ValueError("cannot execute an infeasible commodity placement")
    if execution_architecture is None:
        execution_architecture = (
            Architecture.SWARM_CURRENT_OPT
            if placement.placement_kind == "CURRENT_PLACEMENT"
            else Architecture.SWARM_D_OPT
        )
    arm = (
        StageAArm.A_CURRENT
        if execution_architecture
        in {Architecture.SWARM_CURRENT_OPT, Architecture.CURRENT_ON_D_PLACEMENT}
        else StageAArm.D_FUSE_OUTPUT
    )
    topology = commodity_topology(
        placement,
        architecture=execution_architecture.value,
    )
    row = DecodeServingEngine().execute(
        repo_root=repo_root,
        topology=topology,
        service=service,
        active_sequence_count=concurrency,
        arm=arm,
        available_node_budget=placement.available_node_budget,
        placement_sha256=placement.placement_sha256,
        active_node_ids=placement.active_node_ids,
        resident_model_bytes=sum(placement.memory_used_by_node.values()),
        peak_transient_bytes=sum(placement.transient_bytes_by_node.values()),
    )
    row.update(
        {
            "placement_kind": placement.placement_kind,
            "layer_zero_candidate_id": placement.layer_zero_candidate_id,
            "whole_layer_layer_ids": "|".join(
                str(value) for value in placement.whole_layer_layer_ids
            ),
            "p8_layer_ids": "|".join(str(value) for value in placement.p8_layer_ids),
            "whole_layer_layer_count": len(placement.whole_layer_layer_ids),
            "p8_layer_count": len(placement.p8_layer_ids),
            "whole_layer_only_commodity_model_feasible": False,
            "canonical_commodity_architecture": "1-WHOLE-L0+92-P8",
        }
    )
    return row


def _select_candidate_budgets(
    screening_rows: tuple[dict[str, Any], ...],
    *,
    latency_budget_ms: float,
) -> tuple[dict[str, Any], ...]:
    selections: list[dict[str, Any]] = []
    for scenario in CommodityScenario:
        for architecture in (
            Architecture.SWARM_CURRENT_OPT,
            Architecture.SWARM_D_OPT,
        ):
            subset = [
                row
                for row in screening_rows
                if row["scenario"] == scenario.value
                and row["architecture"] == architecture.value
            ]
            by_budget: dict[int, list[dict[str, Any]]] = {}
            for row in subset:
                by_budget.setdefault(int(row["available_node_budget"]), []).append(row)
            summaries = []
            for budget, rows in sorted(by_budget.items()):
                slo_rows = [
                    row
                    for row in rows
                    if float(row["p95_token_latency_ms"]) <= latency_budget_ms
                ]
                eligible = slo_rows or rows
                best = max(
                    eligible,
                    key=lambda row: (
                        float(row["aggregate_output_tokens_per_second"]),
                        -float(row["p95_token_latency_ms"]),
                    ),
                )
                summaries.append(
                    {
                        "scenario": scenario.value,
                        "architecture": architecture.value,
                        "available_node_budget": budget,
                        "screening_slo_feasible": bool(slo_rows),
                        "best_screening_concurrency": int(best["concurrency"]),
                        "best_screening_output_tps": float(
                            best["aggregate_output_tokens_per_second"]
                        ),
                        "active_node_count": int(best["active_node_count"]),
                    }
                )
            for candidate in summaries:
                dominated = any(
                    other["active_node_count"] <= candidate["active_node_count"]
                    and other["best_screening_output_tps"]
                    >= candidate["best_screening_output_tps"]
                    and (
                        other["active_node_count"] < candidate["active_node_count"]
                        or other["best_screening_output_tps"]
                        > candidate["best_screening_output_tps"]
                    )
                    for other in summaries
                )
                candidate["screening_pareto_selected"] = not dominated
                candidate["selection_rule"] = (
                    "non-dominated active-nodes versus best frozen-screening throughput"
                )
                selections.append(candidate)
    return tuple(selections)


def _economics_row(
    selected: dict[str, Any],
    *,
    reference_slo_output_tps: float,
) -> dict[str, Any]:
    throughput = float(selected["aggregate_output_tokens_per_second"])
    active_nodes = int(selected["active_node_count"])
    active_compute = float(selected["active_compute_equivalents"])
    hourly_cost = active_nodes * PRIMARY_CONTRIBUTOR_PAYOUT_USD_PER_ACTIVE_NODE_HOUR
    metrics = commercial_metrics(
        swarm_slo_output_tokens_per_second=throughput,
        concentrated_fast_slo_output_tokens_per_second=reference_slo_output_tps,
        swarm_hourly_cost_usd=hourly_cost,
    )
    result = dict(selected)
    result.update(
        {
            "selected_slo_concurrency": int(selected["concurrency"]),
            "performance_retention": metrics.performance_retention,
            "swarm_hourly_cost_at_0_15": hourly_cost,
            "api_cost_ratio_at_0_15": metrics.api_cost_ratio,
            "api_discount_percent_at_0_15": metrics.api_discount_percent,
            "performance_cost_leverage_at_0_15": metrics.performance_cost_leverage,
            "gross_profit_per_M_if_sold_at_15": metrics.gross_profit_per_m_if_sold_at_15,
            "slo_feasible": True,
        }
    )
    for payout in PAYOUT_SENSITIVITY_USD_PER_ACTIVE_NODE_HOUR:
        result[f"cost_per_M_at_{payout:g}".replace(".", "_")] = cost_per_million(
            active_nodes * payout,
            throughput,
        )
    for target in TARGET_COST_LEVELS_USD_PER_M:
        suffix = f"{target:g}".replace(".", "_")
        result[f"max_uniform_payout_at_{suffix}"] = (
            max_uniform_payout_per_active_node_hour(target, throughput, active_nodes)
        )
        result[f"max_compute_weighted_payout_at_{suffix}"] = (
            max_compute_weighted_payout_per_hour(target, throughput, active_compute)
        )
    # Stable legacy field names retained by the original E024 artifact schema.
    result["cost_per_M_at_0_05"] = result["cost_per_M_at_0_05"]
    result["cost_per_M_at_0_10"] = result["cost_per_M_at_0_1"]
    result["cost_per_M_at_0_15"] = metrics.swarm_cost_per_m
    result["cost_per_M_at_0_25"] = result["cost_per_M_at_0_25"]
    result["cost_per_M_at_0_50"] = result["cost_per_M_at_0_5"]
    result["max_uniform_payout_at_15"] = result["max_uniform_payout_at_15"]
    result["max_uniform_payout_at_12"] = result["max_uniform_payout_at_12"]
    result["max_uniform_payout_at_9"] = result["max_uniform_payout_at_9"]
    result["max_uniform_payout_at_7_5"] = result["max_uniform_payout_at_7_5"]
    result["max_uniform_payout_at_5"] = result["max_uniform_payout_at_5"]
    result["max_uniform_payout_at_3"] = result["max_uniform_payout_at_3"]
    result["max_compute_weighted_payout_at_15"] = result[
        "max_compute_weighted_payout_at_15"
    ]
    result["max_compute_weighted_payout_at_7_5"] = result[
        "max_compute_weighted_payout_at_7_5"
    ]
    result["max_compute_weighted_payout_at_3"] = result[
        "max_compute_weighted_payout_at_3"
    ]
    return result


def _mark_pareto(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        same_architecture = [
            other
            for other in rows
            if other["scenario"] == row["scenario"]
            and other["architecture"] == row["architecture"]
        ]
        same_scenario = [
            other for other in rows if other["scenario"] == row["scenario"]
        ]

        def dominated(
            pool: list[dict[str, Any]], target: dict[str, Any] = row
        ) -> bool:
            return any(
                float(other["cost_per_M_at_0_15"])
                <= float(target["cost_per_M_at_0_15"])
                and float(other["performance_retention"])
                >= float(target["performance_retention"])
                and (
                    float(other["cost_per_M_at_0_15"])
                    < float(target["cost_per_M_at_0_15"])
                    or float(other["performance_retention"])
                    > float(target["performance_retention"])
                )
                for other in pool
                if other is not target
            )

        row["architecture_pareto_optimal"] = not dominated(same_architecture)
        row["combined_pareto_optimal"] = not dominated(same_scenario)


def run_stage_b(repo_root: Path, service: E024ServiceTable) -> StageBResult:
    repo_root = repo_root.resolve()
    reference = build_reference_architecture(repo_root)
    reference_rows = _reference_rows(repo_root, service, reference)
    reference_c1 = next(row for row in reference_rows if int(row["concurrency"]) == 1)
    c1_p95 = float(reference_c1["p95_token_latency_ms"])
    latency_budget = c1_p95 * PRIMARY_DECODE_SLO_MULTIPLIER
    reference_slo_rows = [
        row
        for row in reference_rows
        if float(row["p95_token_latency_ms"]) <= latency_budget
    ]
    reference_slo = max(
        float(row["aggregate_output_tokens_per_second"])
        for row in reference_slo_rows
    )

    placements = build_commodity_placements(repo_root)
    feasible = [placement for placement in placements if placement.feasible]
    screening_rows = tuple(
        _execute_commodity(repo_root, service, placement, concurrency)
        for placement in feasible
        for concurrency in SCREENING_CONCURRENCY
    )
    selected_budgets = _select_candidate_budgets(
        screening_rows,
        latency_budget_ms=latency_budget,
    )
    selected_keys = {
        (
            row["scenario"],
            row["architecture"],
            int(row["available_node_budget"]),
        )
        for row in selected_budgets
        if row["screening_pareto_selected"]
    }
    screening_lookup = {
        (
            row["scenario"],
            row["architecture"],
            int(row["available_node_budget"]),
            int(row["concurrency"]),
        ): row
        for row in screening_rows
    }
    decode_rows: list[dict[str, Any]] = []
    for placement in feasible:
        architecture = (
            Architecture.SWARM_CURRENT_OPT
            if placement.placement_kind == "CURRENT_PLACEMENT"
            else Architecture.SWARM_D_OPT
        )
        selection_key = (
            placement.scenario.value,
            architecture.value,
            placement.available_node_budget,
        )
        if selection_key not in selected_keys:
            continue
        for concurrency in DECODE_CONCURRENCY_LEVELS:
            lookup_key = (*selection_key, concurrency)
            row = screening_lookup.get(lookup_key)
            if row is None:
                row = _execute_commodity(
                    repo_root,
                    service,
                    placement,
                    concurrency,
                )
            decode_rows.append(dict(row))

    frontier: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in decode_rows:
        key = (
            str(row["scenario"]),
            str(row["architecture"]),
            int(row["available_node_budget"]),
        )
        groups.setdefault(key, []).append(row)
    for group in groups.values():
        slo_rows = [
            row
            for row in group
            if float(row["p95_token_latency_ms"]) <= latency_budget
        ]
        if not slo_rows:
            continue
        selected = max(
            slo_rows,
            key=lambda row: (
                float(row["aggregate_output_tokens_per_second"]),
                -float(row["p95_token_latency_ms"]),
            ),
        )
        frontier.append(
            _economics_row(
                selected,
                reference_slo_output_tps=reference_slo,
            )
        )
    _mark_pareto(frontier)

    canonical: list[dict[str, Any]] = []
    for scenario in CommodityScenario:
        candidates = [
            row
            for row in frontier
            if row["scenario"] == scenario.value and row["combined_pareto_optimal"]
        ]
        if not candidates:
            continue
        selected = max(
            candidates,
            key=lambda row: (
                float(row["performance_cost_leverage_at_0_15"]),
                float(row["performance_retention"]),
                -float(row["cost_per_M_at_0_15"]),
                -int(row["active_node_count"]),
            ),
        )
        selected["canonical_scenario_point"] = True
        canonical.append(selected)
    for row in frontier:
        row.setdefault("canonical_scenario_point", False)
        row["scenario_wedge_pass"] = False
        row["scenario_wedge_status"] = "PENDING_GLOBAL_CORRECTNESS"
    return StageBResult(
        reference=reference,
        reference_rows=reference_rows,
        reference_c1_p95_token_latency_ms=c1_p95,
        primary_token_latency_budget_ms=latency_budget,
        reference_slo_output_tps=reference_slo,
        placements=placements,
        screening_rows=screening_rows,
        selected_budget_rows=selected_budgets,
        decode_rows=tuple(decode_rows),
        frontier_rows=tuple(frontier),
        canonical_rows=tuple(canonical),
    )


def apply_global_correctness(
    result: StageBResult,
    *,
    global_correctness_pass: bool,
) -> StageBResult:
    def truth(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1"}:
                return True
            if normalized in {"false", "0", ""}:
                return False
        raise ValueError(f"not a serialized boolean: {value!r}")

    updated: list[dict[str, Any]] = []
    for source in result.frontier_rows:
        row = dict(source)
        row["scenario_wedge_pass"] = scenario_wedge_pass(
            cost_per_m=float(row["cost_per_M_at_0_15"]),
            slo_feasible=truth(row["slo_feasible"]),
            layer_zero_uses_exact_whole_candidate=(
                row["layer_zero_candidate_id"] == LAYER_ZERO_WHOLE_CANDIDATE_ID
            ),
            p8_layer_count=int(row["p8_layer_count"]),
            no_whole_layer_execution_on_layers_1_92=(
                int(row["whole_layer_layer_count"]) == 1
            ),
            whole_layer_only_commodity_model_feasible=truth(
                row["whole_layer_only_commodity_model_feasible"]
            ),
            p8_required_whole_layer_incapable_compute_share=float(
                row["p8_required_whole_layer_incapable_compute_share"]
            ),
            global_correctness_pass=global_correctness_pass,
        )
        row["scenario_wedge_status"] = "FINAL"
        updated.append(row)
    canonical = tuple(
        row for row in updated if truth(row["canonical_scenario_point"])
    )
    return replace(
        result,
        frontier_rows=tuple(updated),
        canonical_rows=canonical,
    )


def canonical_medians(result: StageBResult) -> dict[str, float]:
    rows = result.canonical_rows
    if not rows:
        raise ValueError("canonical scenario rows are required")
    return {
        "median_performance_retention": statistics.median(
            float(row["performance_retention"]) for row in rows
        ),
        "median_cost_per_M": statistics.median(
            float(row["cost_per_M_at_0_15"]) for row in rows
        ),
        "median_api_cost_ratio": statistics.median(
            float(row["api_cost_ratio_at_0_15"]) for row in rows
        ),
        "median_performance_cost_leverage": statistics.median(
            float(row["performance_cost_leverage_at_0_15"]) for row in rows
        ),
        "median_overall_whole_layer_incapable_compute_share": statistics.median(
            float(row["overall_whole_layer_incapable_compute_share"]) for row in rows
        ),
        "median_p8_required_whole_layer_incapable_compute_share": statistics.median(
            float(row["p8_required_whole_layer_incapable_compute_share"])
            for row in rows
        ),
        "median_max_uniform_payout_at_15": statistics.median(
            float(row["max_uniform_payout_at_15"]) for row in rows
        ),
        "median_max_uniform_payout_at_7_5": statistics.median(
            float(row["max_uniform_payout_at_7_5"]) for row in rows
        ),
        "median_max_uniform_payout_at_3": statistics.median(
            float(row["max_uniform_payout_at_3"]) for row in rows
        ),
    }


def run_current_on_d_causal_checks(
    repo_root: Path,
    service: E024ServiceTable,
    result: StageBResult,
) -> tuple[dict[str, Any], ...]:
    """Change execution semantics only while retaining each canonical D placement."""

    placement_lookup = {
        (
            placement.scenario.value,
            placement.placement_kind,
            placement.available_node_budget,
        ): placement
        for placement in result.placements
    }
    rows: list[dict[str, Any]] = []
    for canonical in result.canonical_rows:
        if canonical["architecture"] != Architecture.SWARM_D_OPT.value:
            continue
        scenario = str(canonical["scenario"])
        budget = int(canonical["available_node_budget"])
        concurrency = int(canonical["selected_slo_concurrency"])
        placement = placement_lookup[(scenario, "D_PLACEMENT", budget)]
        current = _execute_commodity(
            repo_root,
            service,
            placement,
            concurrency,
            execution_architecture=Architecture.CURRENT_ON_D_PLACEMENT,
        )
        d_tps = float(canonical["aggregate_output_tokens_per_second"])
        current_tps = float(current["aggregate_output_tokens_per_second"])
        hourly_cost = (
            placement.active_node_count
            * PRIMARY_CONTRIBUTOR_PAYOUT_USD_PER_ACTIVE_NODE_HOUR
        )
        d_cost = cost_per_million(hourly_cost, d_tps)
        current_cost = cost_per_million(hourly_cost, current_tps)
        rows.append(
            {
                "scenario": scenario,
                "available_node_budget": budget,
                "placement_sha256": placement.placement_sha256,
                "concurrency": concurrency,
                "active_node_count": placement.active_node_count,
                "d_output_tps": d_tps,
                "current_on_d_placement_output_tps": current_tps,
                "execution_only_throughput_improvement_percent": (
                    (d_tps - current_tps) / current_tps * 100
                ),
                "d_cost_per_M": d_cost,
                "current_on_d_placement_cost_per_M": current_cost,
                "execution_only_cost_reduction_percent": (
                    (current_cost - d_cost) / current_cost * 100
                ),
                "placement_identical": True,
                "service_table_identical": (
                    current["service_table_sha256"]
                    == canonical["service_table_sha256"]
                ),
                "only_execution_semantics_changed": True,
                "status": "PASS",
            }
        )
    return tuple(rows)


__all__ = [
    "SCREENING_CONCURRENCY",
    "StageBResult",
    "apply_global_correctness",
    "build_commodity_placements",
    "canonical_medians",
    "run_current_on_d_causal_checks",
    "run_stage_b",
]
