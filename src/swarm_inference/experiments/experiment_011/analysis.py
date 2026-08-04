"""Machine-readable network analysis and fixed Experiment 011 gate statistics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

PROFILE_ORDER = (
    "loopback_unshaped",
    "fabric_100g",
    "lan_10g",
    "lan_2_5g",
    "lan_1g",
    "wifi",
    "regional_wan",
    "global_wan",
)
PROFILE_LABELS = {
    "loopback_unshaped": "Loopback",
    "fabric_100g": "100G fabric",
    "lan_10g": "10 GbE",
    "lan_2_5g": "2.5 GbE",
    "lan_1g": "1 GbE",
    "wifi": "Wi-Fi",
    "regional_wan": "Regional WAN",
    "global_wan": "Global WAN",
}
ARCHIVED_010_TPS = {
    "loopback_unshaped": 2.55,
    "fabric_100g": 2.21,
    "lan_10g": 2.16,
    "lan_2_5g": 2.03,
    "lan_1g": 1.77,
    "wifi": 0.84,
    "regional_wan": 0.30,
    "global_wan": 0.08,
}

RESULT_COLUMNS = (
    "profile_order",
    "profile_name",
    "profile_parameters_hash",
    "strategy",
    "stage_count",
    "partition_method",
    "compression_mode",
    "speculation_provider",
    "speculation_depth",
    "run_index",
    "prompt_id",
    "generated_tokens",
    "exact_tokens",
    "token_match",
    "throughput_tps",
    "ttft_seconds",
    "mean_itl_seconds",
    "p95_itl_seconds",
    "messages_per_token",
    "serial_waits_per_token",
    "payload_bytes_per_token",
    "wire_bytes_per_token",
    "compression_ratio",
    "accepted_tokens_per_round",
    "gpu_utilisation",
    "gpu_memory_bytes",
    "host_memory_bytes",
    "fallback_used",
    "valid_for_claims",
)

SUMMARY_COLUMNS = (
    "profile_order",
    "profile_name",
    "archived_010_tps",
    "same_run_baseline_median_tps",
    "stage_exact_median_tps",
    "stage_exact_ci_low",
    "stage_exact_ci_high",
    "difference_median_tps",
    "difference_ci_low",
    "difference_ci_high",
    "archived_difference_tps",
    "archived_percentage_difference",
    "percentage_improvement",
    "throughput_multiple",
    "classification",
    "selected_method",
    "selected_stage_count",
    "compression_selected",
    "speculation_selected",
    "serial_wait_reduction",
    "message_reduction",
    "payload_reduction",
)


def profile_parameters_hash(profile: dict[str, Any]) -> str:
    payload = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def quantile(values: Sequence[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability)) if values else 0.0


def descriptive_statistics(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "median": None,
            "mean": None,
            "standard_deviation": None,
            "p25": None,
            "p75": None,
            "p95": None,
        }
    return {
        "count": len(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p25": quantile(values, 0.25),
        "p75": quantile(values, 0.75),
        "p95": quantile(values, 0.95),
    }


def bootstrap_median_ci(
    values: Sequence[float], *, seed: int = 11011, samples: int = 20_000
) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        return (float(values[0]), float(values[0]))
    generator = np.random.default_rng(seed)
    source = np.asarray(values, dtype=np.float64)
    indices = generator.integers(0, len(source), size=(samples, len(source)))
    medians = np.median(source[indices], axis=1)
    return (float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975)))


def bootstrap_difference_ci(
    stage_values: Sequence[float],
    baseline_values: Sequence[float],
    *,
    seed: int = 11012,
    samples: int = 20_000,
) -> tuple[float, float]:
    if not stage_values or not baseline_values:
        return (float("nan"), float("nan"))
    if len(stage_values) == 1 and len(baseline_values) == 1:
        difference = float(stage_values[0] - baseline_values[0])
        return (difference, difference)
    generator = np.random.default_rng(seed)
    stage = np.asarray(stage_values, dtype=np.float64)
    baseline = np.asarray(baseline_values, dtype=np.float64)
    stage_indices = generator.integers(0, len(stage), size=(samples, len(stage)))
    baseline_indices = generator.integers(0, len(baseline), size=(samples, len(baseline)))
    differences = np.median(stage[stage_indices], axis=1) - np.median(
        baseline[baseline_indices], axis=1
    )
    return (float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975)))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, Any], name: str) -> float:
    value = row.get(name)
    return float(value) if value not in {None, ""} else 0.0


def build_network_summary(
    rows: list[dict[str, Any]], *, output_directory: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid = [
        row
        for row in rows
        if str(row.get("valid_for_claims")).lower() in {"true", "1"}
        and str(row.get("token_match")).lower() in {"true", "1"}
        and str(row.get("evidence_category")) == "REAL_MODEL_MEASURED"
    ]
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        grouped[(str(row["profile_name"]), str(row["strategy"]))].append(row)
    summary: list[dict[str, Any]] = []
    inferential_notes: list[dict[str, Any]] = []
    for profile_index, profile in enumerate(PROFILE_ORDER, start=1):
        baseline_rows = grouped[(profile, "experiment_011_same_run_expert_rpc")]
        stage_candidates = [
            row
            for row in valid
            if row["profile_name"] == profile
            and str(row["strategy"]).startswith("stage_ring_exact")
            and str(row.get("planner_eligible", "true")).lower() in {"true", "1"}
        ]
        planner_rows = [
            row for row in stage_candidates if row["strategy"] == "stage_ring_exact_best_planner"
        ]
        if planner_rows:
            stage_candidates = planner_rows
        by_strategy: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in stage_candidates:
            by_strategy[str(row["strategy"])].append(row)
        selected_strategy = max(
            by_strategy,
            key=lambda strategy: statistics.median(
                _float(row, "throughput_tps") for row in by_strategy[strategy]
            ),
            default="",
        )
        selected_rows = by_strategy[selected_strategy]
        baseline_values = [_float(row, "throughput_tps") for row in baseline_rows]
        stage_values = [_float(row, "throughput_tps") for row in selected_rows]
        baseline_median = statistics.median(baseline_values) if baseline_values else 0.0
        stage_median = statistics.median(stage_values) if stage_values else 0.0
        stage_ci = bootstrap_median_ci(stage_values, seed=11011 + profile_index)
        difference = stage_median - baseline_median
        difference_ci = bootstrap_difference_ci(
            stage_values, baseline_values, seed=12011 + profile_index
        )
        independent_replication_sufficient = len(stage_values) >= 2 and len(baseline_values) >= 2
        if not stage_values or not baseline_values:
            classification = "INCONCLUSIVE"
            reason = "one or both compared series have no valid measured row"
        elif difference_ci[0] > 0:
            classification = "IMPROVED"
            reason = "complete bootstrap 95% interval is above zero"
        elif difference_ci[1] < 0:
            classification = "REGRESSED"
            reason = "complete bootstrap 95% interval is below zero"
        else:
            classification = "INCONCLUSIVE"
            reason = "bootstrap 95% interval includes zero"
        selected = selected_rows[0] if selected_rows else {}
        baseline = baseline_rows[0] if baseline_rows else {}
        baseline_waits = _float(baseline, "serial_waits_per_token")
        baseline_messages = _float(baseline, "messages_per_token")
        baseline_payload = _float(baseline, "payload_bytes_per_token")
        selected_waits = _float(selected, "serial_waits_per_token")
        selected_messages = _float(selected, "messages_per_token")
        selected_payload = _float(selected, "payload_bytes_per_token")
        row = {
            "profile_order": profile_index,
            "profile_name": profile,
            "archived_010_tps": ARCHIVED_010_TPS[profile],
            "same_run_baseline_median_tps": baseline_median,
            "stage_exact_median_tps": stage_median,
            "stage_exact_ci_low": stage_ci[0],
            "stage_exact_ci_high": stage_ci[1],
            "difference_median_tps": difference,
            "difference_ci_low": difference_ci[0],
            "difference_ci_high": difference_ci[1],
            "archived_difference_tps": stage_median - ARCHIVED_010_TPS[profile],
            "archived_percentage_difference": (
                (stage_median - ARCHIVED_010_TPS[profile]) / ARCHIVED_010_TPS[profile] * 100.0
            ),
            "percentage_improvement": (
                difference / baseline_median * 100.0 if baseline_median else 0.0
            ),
            "throughput_multiple": stage_median / baseline_median if baseline_median else 0.0,
            "classification": classification,
            "selected_method": selected.get("partition_method", ""),
            "selected_stage_count": selected.get("stage_count", ""),
            "compression_selected": selected.get("compression_mode", "none") != "none",
            "speculation_selected": selected.get("speculation_provider", "none") != "none",
            "serial_wait_reduction": (
                1.0 - selected_waits / baseline_waits if baseline_waits else 0.0
            ),
            "message_reduction": (
                1.0 - selected_messages / baseline_messages if baseline_messages else 0.0
            ),
            "payload_reduction": (
                1.0 - selected_payload / baseline_payload if baseline_payload else 0.0
            ),
        }
        summary.append(row)
        inferential_notes.append(
            {
                "profile_name": profile,
                "baseline_independent_rows": len(baseline_values),
                "stage_independent_rows": len(stage_values),
                "classification": classification,
                "reason": reason,
                "point_estimate_difference_tps": difference,
                "bootstrap_difference_ci": list(difference_ci),
                "bootstrap_ci_interpretation": (
                    "degenerate descriptive interval from the frozen single repetition"
                    if not independent_replication_sufficient
                    else "inferential"
                ),
            }
        )
    write_csv(output_directory / "network_profile_summary.csv", summary, SUMMARY_COLUMNS)
    stage_throughputs = [float(row["stage_exact_median_tps"]) for row in summary]
    baseline_throughputs = [float(row["same_run_baseline_median_tps"]) for row in summary]
    archived_throughputs = [float(row["archived_010_tps"]) for row in summary]
    multiples = [float(row["throughput_multiple"]) for row in summary if row["throughput_multiple"]]

    def geometric_mean(values: Sequence[float]) -> float:
        positive = [value for value in values if value > 0]
        return math.exp(statistics.mean(math.log(value) for value in positive)) if positive else 0.0

    stage_rows_selected = []
    for summary_row in summary:
        profile = str(summary_row["profile_name"])
        candidates = [
            row
            for row in valid
            if row["profile_name"] == profile
            and str(row["strategy"]).startswith("stage_ring_exact")
            and int(row.get("stage_count") or 0) == int(summary_row["selected_stage_count"] or 0)
            and str(row.get("partition_method")) == str(summary_row["selected_method"])
            and (str(row.get("compression_mode")) != "none")
            == bool(summary_row["compression_selected"])
        ]
        if candidates:
            stage_rows_selected.extend(candidates)
    whole_curve = {
        "geometric_mean_stage_throughput_tps": geometric_mean(stage_throughputs),
        "geometric_mean_same_run_baseline_throughput_tps": geometric_mean(baseline_throughputs),
        "geometric_mean_archived_010_throughput_tps": geometric_mean(archived_throughputs),
        "geometric_mean_improvement_multiple": geometric_mean(multiples),
        "global_wan_to_loopback_ratio": (
            stage_throughputs[-1] / stage_throughputs[0] if stage_throughputs[0] else 0.0
        ),
        "regional_wan_to_loopback_ratio": (
            stage_throughputs[-2] / stage_throughputs[0] if stage_throughputs[0] else 0.0
        ),
        "worst_profile_throughput_tps": min(stage_throughputs, default=0.0),
        "median_serial_waits_per_token": statistics.median(
            _float(row, "serial_waits_per_token") for row in stage_rows_selected
        )
        if stage_rows_selected
        else 0.0,
        "median_messages_per_token": statistics.median(
            _float(row, "messages_per_token") for row in stage_rows_selected
        )
        if stage_rows_selected
        else 0.0,
        "median_wire_bytes_per_token": statistics.median(
            _float(row, "wire_bytes_per_token") for row in stage_rows_selected
        )
        if stage_rows_selected
        else 0.0,
        "improved_profiles": sum(row["classification"] == "IMPROVED" for row in summary),
        "inconclusive_profiles": sum(row["classification"] == "INCONCLUSIVE" for row in summary),
        "regressed_profiles": sum(row["classification"] == "REGRESSED" for row in summary),
    }
    wan = {
        row["profile_name"]: {
            "throughput_multiple": row["throughput_multiple"],
            "stage_exact_median_tps": row["stage_exact_median_tps"],
            "classification": row["classification"],
            "significantly_improved": row["classification"] == "IMPROVED",
        }
        for row in summary
        if row["profile_name"] in {"wifi", "regional_wan", "global_wan"}
    }
    analysis = {
        "whole_curve": whole_curve,
        "wan_metrics": wan,
        "inferential_notes": inferential_notes,
        "statistics": {
            f"{profile}:{strategy}": descriptive_statistics(
                [_float(row, "throughput_tps") for row in grouped_rows]
            )
            for (profile, strategy), grouped_rows in sorted(grouped.items())
        },
        "fixed_thresholds_changed_after_results": False,
    }
    (output_directory / "network_analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary, analysis
