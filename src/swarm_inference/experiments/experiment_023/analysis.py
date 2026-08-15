"""Mechanical analysis for Experiment 023 deterministic artifacts.

The analysis is intentionally able to summarize an invalid attempt.  Diagnostic
numbers remain useful for locating the failed evidence path, but the verdict
evaluator always short-circuits on mandatory validity failures.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .freeze import (
    CAPACITY_INVENTORIES,
    CONTROL_INVENTORIES,
    HEADLINE_INVENTORIES,
)

PRIMARY_MODE = "SHARED_NIC"
LEGACY_MODE = "LEGACY_DIRECTED_LINK"
PRIMARY_SLO_PREFIX = "slo_2.0x"
FAMILIES = (
    "memory-fragmented",
    "compute-heterogeneous",
    "network-heterogeneous",
    "full-mixed",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite analytical input: {value!r}")
    return result


def _integer(value: Any) -> int:
    return int(value)


def _median(values: Iterable[float]) -> float:
    rows = tuple(float(value) for value in values)
    if not rows:
        raise ValueError("median requires at least one value")
    return float(np.quantile(rows, 0.50, method="linear"))


def _quantile(values: Iterable[float], probability: float) -> float:
    rows = tuple(float(value) for value in values)
    if not rows:
        raise ValueError("quantile requires at least one value")
    return float(np.quantile(rows, probability, method="linear"))


def _percent_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        raise ValueError("uplift denominator must be positive")
    return 100.0 * (numerator / denominator - 1.0)


def _slo(row: Mapping[str, str], metric: str) -> float:
    return _number(row[f"{PRIMARY_SLO_PREFIX}_{metric}"])


@dataclass(frozen=True, slots=True)
class AttemptAnalysis:
    uplift_rows: tuple[dict[str, Any], ...]
    family_rows: tuple[dict[str, Any], ...]
    optionality_rows: tuple[dict[str, Any], ...]
    replica_efficiency_rows: tuple[dict[str, Any], ...]
    capacity_rows: tuple[dict[str, Any], ...]
    control_failures: tuple[dict[str, Any], ...]
    completeness: dict[str, Any]
    diagnostic_summary: dict[str, Any]


def _index_saturation(
    rows: Sequence[Mapping[str, str]],
) -> dict[tuple[str, str, str], Mapping[str, str]]:
    result: dict[tuple[str, str, str], Mapping[str, str]] = {}
    for row in rows:
        key = (row["inventory_id"], row["arm"], row["network_mode"])
        if key in result:
            raise ValueError(f"duplicate saturation row {key}")
        result[key] = row
    return result


def _constant_arm_fields(
    rows: Sequence[Mapping[str, str]], inventory_id: str, arm: str
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["inventory_id"] == inventory_id
        and row["arm"] == arm
        and row["network_mode"] == PRIMARY_MODE
    ]
    if not matches:
        raise ValueError(f"missing arm rows for {inventory_id}/{arm}")
    fields = (
        "replica_count",
        "replica_checkpoint_bytes",
        "replica_resident_bytes",
        "new_nodes_activated",
        "abstract_node_cost",
        "nodes_used",
        "plan_sha256",
    )
    values = {field: {row[field] for row in matches} for field in fields}
    if any(len(value) != 1 for value in values.values()):
        raise ValueError(f"non-constant plan accounting for {inventory_id}/{arm}")
    return {
        "replica_count": _integer(next(iter(values["replica_count"]))),
        "replica_checkpoint_bytes": _integer(
            next(iter(values["replica_checkpoint_bytes"]))
        ),
        "replica_resident_bytes": _integer(
            next(iter(values["replica_resident_bytes"]))
        ),
        "new_nodes_activated": _integer(
            next(iter(values["new_nodes_activated"]))
        ),
        "abstract_node_cost": _number(next(iter(values["abstract_node_cost"]))),
        "nodes_used": _integer(next(iter(values["nodes_used"]))),
        "plan_sha256": next(iter(values["plan_sha256"])),
    }


def _routing_usage(
    rows: Sequence[Mapping[str, str]], inventory_id: str, arm: str
) -> tuple[int, int, float]:
    matches = [
        row
        for row in rows
        if row["inventory_id"] == inventory_id
        and row["arm"] == arm
        and row["network_mode"] == PRIMARY_MODE
    ]
    primary = sum(_integer(row["selected_primary_count"]) for row in matches)
    alternate = sum(_integer(row["selected_alternate_count"]) for row in matches)
    total = primary + alternate
    return primary, alternate, alternate / total if total else 0.0


def _inventory_analysis_row(
    inventory_id: str,
    saturation: Mapping[tuple[str, str, str], Mapping[str, str]],
    arm_rows: Sequence[Mapping[str, str]],
    routing_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    required = (
        "U_STRONG",
        "FLEX_FREE_NO_ALT",
        "FLEX_FREE",
        "FLEX_POOL_NO_ALT",
        "FLEX_POOL",
    )
    shared = {
        arm: saturation[(inventory_id, arm, PRIMARY_MODE)] for arm in required
    }
    legacy_u = saturation[(inventory_id, "U_STRONG", LEGACY_MODE)]
    legacy_flex = saturation[(inventory_id, "FLEX_POOL", LEGACY_MODE)]
    u_eff = _slo(shared["U_STRONG"], "rows_per_second_per_abstract_cost")
    u_tps = _slo(shared["U_STRONG"], "target_rows_per_second")
    pool_eff = _slo(shared["FLEX_POOL"], "rows_per_second_per_abstract_cost")
    pool_tps = _slo(shared["FLEX_POOL"], "target_rows_per_second")
    pool_no_alt_eff = _slo(
        shared["FLEX_POOL_NO_ALT"], "rows_per_second_per_abstract_cost"
    )
    free_eff = _slo(shared["FLEX_FREE"], "rows_per_second_per_abstract_cost")
    free_tps = _slo(shared["FLEX_FREE"], "target_rows_per_second")
    free_no_alt_eff = _slo(
        shared["FLEX_FREE_NO_ALT"], "rows_per_second_per_abstract_cost"
    )
    legacy_u_eff = _slo(legacy_u, "rows_per_second_per_abstract_cost")
    legacy_flex_eff = _slo(legacy_flex, "rows_per_second_per_abstract_cost")
    pool_plan = _constant_arm_fields(arm_rows, inventory_id, "FLEX_POOL")
    free_plan = _constant_arm_fields(arm_rows, inventory_id, "FLEX_FREE")
    primary_count, alternate_count, selection_rate = _routing_usage(
        routing_rows, inventory_id, "FLEX_POOL"
    )
    free_primary, free_alternate, free_selection_rate = _routing_usage(
        routing_rows, inventory_id, "FLEX_FREE"
    )
    source = shared["U_STRONG"]
    return {
        "inventory_id": inventory_id,
        "family": source["family"],
        "cohort": source["cohort"],
        "u_strong_slo_target_rows_per_second": u_tps,
        "flex_pool_slo_target_rows_per_second": pool_tps,
        "u_strong_slo_rows_per_second_per_abstract_cost": u_eff,
        "flex_pool_slo_rows_per_second_per_abstract_cost": pool_eff,
        "efficiency_uplift_percent": _percent_ratio(pool_eff, u_eff),
        "throughput_uplift_percent": _percent_ratio(pool_tps, u_tps),
        "optionality_only_percent": _percent_ratio(pool_eff, pool_no_alt_eff),
        "flex_free_efficiency_uplift_percent": _percent_ratio(free_eff, u_eff),
        "flex_free_throughput_uplift_percent": _percent_ratio(free_tps, u_tps),
        "flex_free_optionality_only_percent": _percent_ratio(
            free_eff, free_no_alt_eff
        ),
        "legacy_efficiency_uplift_percent": _percent_ratio(
            legacy_flex_eff, legacy_u_eff
        ),
        "u_strong_abstract_node_cost": _number(shared["U_STRONG"]["abstract_node_cost"]),
        "flex_pool_abstract_node_cost": pool_plan["abstract_node_cost"],
        "flex_free_abstract_node_cost": free_plan["abstract_node_cost"],
        "flex_pool_replica_count": pool_plan["replica_count"],
        "flex_pool_replica_checkpoint_bytes": pool_plan["replica_checkpoint_bytes"],
        "flex_pool_replica_resident_bytes": pool_plan["replica_resident_bytes"],
        "flex_pool_replica_resident_gib": pool_plan["replica_resident_bytes"]
        / (1024**3),
        "flex_pool_new_nodes_activated": pool_plan["new_nodes_activated"],
        "flex_pool_selected_primary_count": primary_count,
        "flex_pool_selected_alternate_count": alternate_count,
        "flex_pool_replica_selection_rate": selection_rate,
        "flex_pool_actual_replica_used": alternate_count > 0,
        "flex_free_replica_count": free_plan["replica_count"],
        "flex_free_selected_primary_count": free_primary,
        "flex_free_selected_alternate_count": free_alternate,
        "flex_free_replica_selection_rate": free_selection_rate,
        "flex_free_actual_replica_used": free_alternate > 0,
        "u_strong_plan_sha256": shared["U_STRONG"]["plan_sha256"],
        "flex_pool_plan_sha256": shared["FLEX_POOL"]["plan_sha256"],
    }


def _family_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for family in FAMILIES:
        values = [row for row in rows if row["family"] == family]
        if not values:
            continue
        result.append(
            {
                "family": family,
                "inventory_count": len(values),
                "median_efficiency_uplift_percent": _median(
                    row["efficiency_uplift_percent"] for row in values
                ),
                "mean_efficiency_uplift_percent": float(
                    np.mean([row["efficiency_uplift_percent"] for row in values])
                ),
                "cases_ge_20_percent": sum(
                    row["efficiency_uplift_percent"] >= 20.0 for row in values
                ),
                "median_raw_throughput_uplift_percent": _median(
                    row["throughput_uplift_percent"] for row in values
                ),
                "worst_efficiency_uplift_percent": min(
                    row["efficiency_uplift_percent"] for row in values
                ),
                "worst_throughput_uplift_percent": min(
                    row["throughput_uplift_percent"] for row in values
                ),
                "replica_using_inventories": sum(
                    bool(row["flex_pool_actual_replica_used"]) for row in values
                ),
                "median_legacy_efficiency_uplift_percent": _median(
                    row["legacy_efficiency_uplift_percent"] for row in values
                ),
                "legacy_nonnegative_inventories": sum(
                    row["legacy_efficiency_uplift_percent"] >= 0.0 for row in values
                ),
            }
        )
    return result


def analyze_attempt(attempt_root: Path) -> AttemptAnalysis:
    """Read one complete deterministic attempt and derive diagnostic metrics."""

    saturation_rows = _read_csv(attempt_root / "serving/saturation-summary.csv")
    arm_rows = _read_csv(attempt_root / "serving/arm-results.csv")
    routing_rows = _read_csv(attempt_root / "serving/replica-routing-summary.csv")
    saturation = _index_saturation(saturation_rows)
    inventory_ids = tuple(
        HEADLINE_INVENTORIES + CONTROL_INVENTORIES + CAPACITY_INVENTORIES
    )
    uplift = [
        _inventory_analysis_row(value, saturation, arm_rows, routing_rows)
        for value in inventory_ids
    ]
    headline = [row for row in uplift if row["cohort"] == "headline"]
    controls = [row for row in uplift if row["cohort"] == "negative_control"]
    capacity = [row for row in uplift if row["cohort"] == "capacity_exploratory"]

    control_failures: list[dict[str, Any]] = []
    for row in controls:
        for arm, efficiency_field, throughput_field in (
            (
                "FLEX_FREE",
                "flex_free_efficiency_uplift_percent",
                "flex_free_throughput_uplift_percent",
            ),
            ("FLEX_POOL", "efficiency_uplift_percent", "throughput_uplift_percent"),
        ):
            efficiency = row[efficiency_field]
            throughput = row[throughput_field]
            if efficiency < -5.0 or throughput < -5.0:
                control_failures.append(
                    {
                        "inventory_id": row["inventory_id"],
                        "arm": arm,
                        "efficiency_uplift_percent": efficiency,
                        "throughput_uplift_percent": throughput,
                        "failure": "CONTROL_REGRESSION_WORSE_THAN_5_PERCENT",
                    }
                )

    expected_arm_rows = len(inventory_ids) * (5 * 5 + 2 * 5)
    expected_saturation_rows = len(inventory_ids) * 7
    complete_status = all(row["status"] == "PASS" for row in arm_rows)
    complete_keys = {
        (
            row["inventory_id"],
            row["arm"],
            row["network_mode"],
            _integer(row["concurrency"]),
        )
        for row in arm_rows
    }
    completeness = {
        "inventory_count": len({row["inventory_id"] for row in arm_rows}),
        "arm_result_row_count": len(arm_rows),
        "expected_arm_result_row_count": expected_arm_rows,
        "saturation_row_count": len(saturation_rows),
        "expected_saturation_row_count": expected_saturation_rows,
        "all_arm_rows_pass": complete_status,
        "unique_arm_result_key_count": len(complete_keys),
        "headline_inventory_count": len(headline),
        "control_inventory_count": len(controls),
        "capacity_inventory_count": len(capacity),
        "all_required_rows_complete": (
            len(arm_rows) == expected_arm_rows
            and len(complete_keys) == expected_arm_rows
            and len(saturation_rows) == expected_saturation_rows
            and complete_status
        ),
    }
    diagnostic_summary = summarize_headline(headline)
    optionality = [
        {
            "inventory_id": row["inventory_id"],
            "family": row["family"],
            "cohort": row["cohort"],
            "flex_pool_vs_no_alt_efficiency_percent": row[
                "optionality_only_percent"
            ],
            "flex_pool_vs_u_strong_efficiency_percent": row[
                "efficiency_uplift_percent"
            ],
            "flex_free_vs_no_alt_efficiency_percent": row[
                "flex_free_optionality_only_percent"
            ],
            "flex_pool_actual_replica_used": row["flex_pool_actual_replica_used"],
        }
        for row in uplift
    ]
    replica_efficiency = [
        {
            "inventory_id": row["inventory_id"],
            "family": row["family"],
            "cohort": row["cohort"],
            "replica_count": row["flex_pool_replica_count"],
            "replica_checkpoint_bytes": row["flex_pool_replica_checkpoint_bytes"],
            "replica_resident_bytes": row["flex_pool_replica_resident_bytes"],
            "replica_resident_gib": row["flex_pool_replica_resident_gib"],
            "new_nodes_activated": row["flex_pool_new_nodes_activated"],
            "replica_selection_rate": row["flex_pool_replica_selection_rate"],
            "efficiency_uplift_percent": row["efficiency_uplift_percent"],
            "throughput_uplift_percent": row["throughput_uplift_percent"],
        }
        for row in uplift
    ]
    return AttemptAnalysis(
        uplift_rows=tuple(uplift),
        family_rows=tuple(_family_summary(headline)),
        optionality_rows=tuple(optionality),
        replica_efficiency_rows=tuple(replica_efficiency),
        capacity_rows=tuple(capacity),
        control_failures=tuple(control_failures),
        completeness=completeness,
        diagnostic_summary=diagnostic_summary,
    )


def summarize_headline(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 18:
        raise ValueError("headline summary requires exactly 18 inventories")
    efficiency = [float(row["efficiency_uplift_percent"]) for row in rows]
    throughput = [float(row["throughput_uplift_percent"]) for row in rows]
    optionality = [float(row["optionality_only_percent"]) for row in rows]
    flex_free = [float(row["flex_free_efficiency_uplift_percent"]) for row in rows]
    legacy = [float(row["legacy_efficiency_uplift_percent"]) for row in rows]
    return {
        "headline_inventory_count": len(rows),
        "median_efficiency_uplift_percent": _median(efficiency),
        "mean_efficiency_uplift_percent": float(np.mean(efficiency)),
        "p90_efficiency_uplift_percent": _quantile(efficiency, 0.90),
        "maximum_efficiency_uplift_percent": max(efficiency),
        "minimum_efficiency_uplift_percent": min(efficiency),
        "headline_cases_ge_20_percent": sum(value >= 20.0 for value in efficiency),
        "median_raw_throughput_uplift_percent": _median(throughput),
        "worst_raw_throughput_regression_percent": min(throughput),
        "worst_efficiency_regression_percent": min(efficiency),
        "replica_using_headline_inventory_count": sum(
            bool(row["flex_pool_actual_replica_used"]) for row in rows
        ),
        "median_optionality_only_percent": _median(optionality),
        "median_flex_free_efficiency_uplift_percent": _median(flex_free),
        "median_legacy_efficiency_uplift_percent": _median(legacy),
        "legacy_nonnegative_inventory_count": sum(value >= 0.0 for value in legacy),
    }


def _general_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    efficiency = [float(row["efficiency_uplift_percent"]) for row in rows]
    throughput = [float(row["throughput_uplift_percent"]) for row in rows]
    legacy = [float(row["legacy_efficiency_uplift_percent"]) for row in rows]
    checks = {
        "exactly_18_headline_inventories": len(rows) == 18,
        "median_efficiency_at_least_20": len(rows) == 18
        and _median(efficiency) >= 20.0,
        "at_least_12_cases_ge_20": sum(value >= 20.0 for value in efficiency) >= 12,
        "median_throughput_nonnegative": _median(throughput) >= 0.0,
        "no_efficiency_regression_worse_than_5": min(efficiency) >= -5.0,
        "no_throughput_regression_worse_than_5": min(throughput) >= -5.0,
        "at_least_9_actual_replica_users": sum(
            bool(row["flex_pool_actual_replica_used"]) for row in rows
        )
        >= 9,
        "legacy_median_at_least_10": _median(legacy) >= 10.0,
        "legacy_at_least_9_nonnegative": sum(value >= 0.0 for value in legacy) >= 9,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _family_gates(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["family"])].append(row)
    result: dict[str, dict[str, Any]] = {}
    for family in FAMILIES:
        members = grouped.get(family, [])
        count = len(members)
        efficiency = [float(row["efficiency_uplift_percent"]) for row in members]
        throughput = [float(row["throughput_uplift_percent"]) for row in members]
        legacy = [float(row["legacy_efficiency_uplift_percent"]) for row in members]
        expected = 3 if family in {"memory-fragmented", "full-mixed"} else 6
        wins_required = 2 if expected == 3 else 4
        checks = {
            "expected_family_size": count == expected,
            "median_efficiency_at_least_20": count == expected
            and _median(efficiency) >= 20.0,
            "required_cases_ge_20": sum(value >= 20.0 for value in efficiency)
            >= wins_required,
            "median_throughput_nonnegative": _median(throughput) >= 0.0,
            "no_efficiency_regression_worse_than_5": min(efficiency) >= -5.0,
            "no_throughput_regression_worse_than_5": min(throughput) >= -5.0,
            "at_least_half_use_replicas": sum(
                bool(row["flex_pool_actual_replica_used"]) for row in members
            )
            >= math.ceil(count / 2),
            "legacy_median_at_least_5": _median(legacy) >= 5.0,
            "legacy_majority_nonnegative": sum(value >= 0.0 for value in legacy)
            > count / 2,
        }
        result[family] = {"passed": all(checks.values()), "checks": checks}
    return result


def evaluate_verdict(
    headline_rows: Sequence[Mapping[str, Any]],
    *,
    validity_failures: Sequence[str] = (),
) -> dict[str, Any]:
    """Apply the frozen E023 verdict tree without prose overrides."""

    general = _general_gate(headline_rows)
    families = _family_gates(headline_rows)
    qualifying = [name for name, value in families.items() if value["passed"]]
    if validity_failures:
        verdict = "MODEL_INVALID"
    elif general["passed"]:
        verdict = "YES_GENERAL_WEDGE"
    elif qualifying:
        verdict = "YES_CONDITIONAL_WEDGE"
    elif any(
        float(row["efficiency_uplift_percent"]) >= 20.0 for row in headline_rows
    ):
        verdict = "CAPABILITY_SIGNAL_ONLY"
    else:
        verdict = "NO_WEDGE"
    return {
        "final_verdict": verdict,
        "validity_failures": list(validity_failures),
        "general_wedge_gate": general,
        "family_wedge_gates": families,
        "qualifying_families": qualifying if not validity_failures else [],
        "diagnostic_qualifying_families_if_valid": qualifying,
    }


def _zero_new_node_family_gate(
    members: Sequence[Mapping[str, Any]], expected: int
) -> bool:
    efficiency = [float(row["flex_free_efficiency_uplift_percent"]) for row in members]
    throughput = [float(row["flex_free_throughput_uplift_percent"]) for row in members]
    wins = 2 if expected == 3 else 4
    return (
        len(members) == expected
        and _median(efficiency) >= 20.0
        and sum(value >= 20.0 for value in efficiency) >= wins
        and _median(throughput) >= 0.0
        and min(efficiency) >= -5.0
        and min(throughput) >= -5.0
        and sum(bool(row["flex_free_actual_replica_used"]) for row in members)
        >= math.ceil(expected / 2)
    )


def evaluate_zero_new_node_wedge(
    headline_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    efficiency = [
        float(row["flex_free_efficiency_uplift_percent"]) for row in headline_rows
    ]
    throughput = [
        float(row["flex_free_throughput_uplift_percent"]) for row in headline_rows
    ]
    general = (
        len(headline_rows) == 18
        and _median(efficiency) >= 20.0
        and sum(value >= 20.0 for value in efficiency) >= 12
        and _median(throughput) >= 0.0
        and min(efficiency) >= -5.0
        and min(throughput) >= -5.0
        and sum(
            bool(row["flex_free_actual_replica_used"]) for row in headline_rows
        )
        >= 9
    )
    family_results = {}
    for family in FAMILIES:
        members = [row for row in headline_rows if row["family"] == family]
        expected = 3 if family in {"memory-fragmented", "full-mixed"} else 6
        family_results[family] = _zero_new_node_family_gate(members, expected)
    return {
        "zero_new_node_wedge": general or any(family_results.values()),
        "general_gate": general,
        "family_gates": family_results,
        "median_efficiency_uplift_percent": _median(efficiency),
        "cases_ge_20_percent": sum(value >= 20.0 for value in efficiency),
        "median_raw_throughput_uplift_percent": _median(throughput),
    }


__all__ = [
    "AttemptAnalysis",
    "analyze_attempt",
    "evaluate_verdict",
    "evaluate_zero_new_node_wedge",
    "summarize_headline",
]
