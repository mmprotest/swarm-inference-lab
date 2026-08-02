"""Ablation comparison and acceptance-gate evaluation for Experiment 008."""

from __future__ import annotations

import math
from typing import Any

from swarm_inference.config.experiment_008 import Experiment008AcceptanceConfig
from swarm_inference.experiments.experiment_008.schemas import (
    EvidenceClass,
    GateResult,
    GateStatus,
)


def _metric(
    observations: list[dict[str, Any]], configuration: str, workload: str, metric: str
) -> float | None:
    for observation in observations:
        if (
            observation.get("configuration") == configuration
            and observation.get("workload") == workload
            and observation.get("status") == "COMPLETED"
        ):
            metrics = observation.get("metrics", {})
            value = metrics.get(metric) if isinstance(metrics, dict) else None
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return float(value)
    return None


def _change(
    current: float | None, previous: float | None, *, lower_is_better: bool = False
) -> float | None:
    if current is None or previous is None or previous == 0:
        return None
    return (1 - current / previous) if lower_is_better else (current / previous - 1)


def build_ablation_rows(
    observations: list[dict[str, Any]],
    *,
    token_identity_by_configuration: dict[str, float | None],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stock_decode = _metric(observations, "A", "decode", "decode_tokens_per_second")
    stock_ttft_32k = _metric(observations, "A", "prefill_32k", "time_to_first_token_ms")
    previous_decode: float | None = None
    for configuration in "ABCDEFG":
        decode = _metric(observations, configuration, "decode", "decode_tokens_per_second")
        ttft_8k = _metric(observations, configuration, "prefill_8k", "time_to_first_token_ms")
        ttft_32k = _metric(observations, configuration, "prefill_32k", "time_to_first_token_ms")
        mixed = _metric(observations, configuration, "mixed", "mixed_verified_tokens_per_second")
        interactive_p95 = _metric(
            observations, configuration, "mixed", "interactive_p95_latency_ms"
        )
        peak_vram_values = [
            value
            for workload in ("decode", "prefill_8k", "prefill_32k", "mixed")
            if (value := _metric(observations, configuration, workload, "peak_vram_bytes"))
            is not None
        ]
        peak_ram_values = [
            value
            for workload in ("decode", "prefill_8k", "prefill_32k", "mixed")
            if (value := _metric(observations, configuration, workload, "peak_system_ram_bytes"))
            is not None
        ]
        status_rows = [row for row in observations if row.get("configuration") == configuration]
        completed = sum(row.get("status") == "COMPLETED" for row in status_rows)
        status = (
            "COMPLETED"
            if completed == 4
            else "INCOMPLETE"
            if completed
            else (str(status_rows[0].get("status")) if status_rows else "NOT_RUN")
        )
        reasons = sorted(
            {
                str(row.get("unavailable_reason"))
                for row in status_rows
                if row.get("unavailable_reason")
            }
        )
        rows.append(
            {
                "configuration": configuration,
                "status": status,
                "unavailable_reason": "; ".join(reasons) or None,
                "classification": ("MEASURED" if completed else None),
                "decode_tokens_per_second": decode,
                "decode_change_vs_previous": _change(decode, previous_decode),
                "decode_change_vs_stock": _change(decode, stock_decode),
                "ttft_8k": ttft_8k,
                "ttft_32k": ttft_32k,
                "ttft_change_vs_stock": _change(ttft_32k, stock_ttft_32k, lower_is_better=True),
                "mixed_verified_tokens_per_second": mixed,
                "interactive_p95": interactive_p95,
                "peak_vram": max(peak_vram_values) if peak_vram_values else None,
                "peak_ram": max(peak_ram_values) if peak_ram_values else None,
                "pcie_bytes_per_token": _metric(
                    observations, configuration, "decode", "pcie_bytes_per_output_token"
                ),
                "cpu_gpu_overlap_percent": _metric(
                    observations, configuration, "decode", "cpu_gpu_overlap_percent"
                ),
                "expert_cache_hit_rate": _metric(
                    observations, configuration, "decode", "expert_cache_hit_rate"
                ),
                "useful_prefetch_rate": _metric(
                    observations, configuration, "decode", "useful_prefetch_rate"
                ),
                "token_identity_rate": token_identity_by_configuration.get(configuration),
            }
        )
        previous_decode = decode
    return rows


def _gate(
    gate_id: int,
    name: str,
    passed: bool | None,
    reasons: list[str],
    metrics: dict[str, Any],
    *,
    measured: bool,
) -> GateResult:
    return GateResult(
        gate_id=gate_id,
        name=name,
        status=(
            GateStatus.NOT_EVALUATED
            if passed is None
            else GateStatus.PASS
            if passed
            else GateStatus.FAIL
        ),
        evidence_class=EvidenceClass.MEASURED if measured else None,
        reasons=reasons,
        metrics=metrics,
    )


def evaluate_gates(
    *,
    official_full_run: bool,
    preflight: dict[str, Any] | None,
    observations: list[dict[str, Any]],
    ablations: list[dict[str, Any]],
    correctness: dict[str, Any],
    residency: dict[str, Any],
    planner_quality: dict[str, Any],
    architecture_audit: dict[str, Any],
    acceptance: Experiment008AcceptanceConfig,
) -> list[GateResult]:
    if not official_full_run:
        capacity_passed: bool | None = None
        capacity_reasons = ["official capacity gate is evaluated only by a --full run"]
    elif preflight is None:
        capacity_passed = False
        capacity_reasons = ["no eligible real-model preflight was produced"]
    else:
        peak_gpu = max(
            (
                float(value)
                for row in ablations
                if isinstance((value := row.get("peak_vram")), (int, float))
            ),
            default=float("inf"),
        )
        checks = {
            "weights_exceed_32_gib": bool(preflight.get("genuinely_exceeds_32gb")),
            "weights_exceed_physical_vram": bool(preflight.get("genuinely_exceeds_physical_vram")),
            "generation_succeeded": any(row.get("status") == "COMPLETED" for row in observations),
            "peak_gpu_within_vram": peak_gpu <= float(preflight.get("physical_vram_bytes", 0)),
            "system_ram_contributes": bool(residency.get("system_ram_contributes")),
            "no_full_gpu_duplicate": bool(residency.get("no_complete_gpu_duplicate")),
            "bytes_reconciled": bool(residency.get("reconciled")),
        }
        capacity_passed = all(checks.values())
        capacity_reasons = [
            f"{name}: {'pass' if passed else 'fail'}" for name, passed in checks.items()
        ]
    gate1 = _gate(
        1,
        "model capacity",
        capacity_passed,
        capacity_reasons,
        {
            "total_tensor_bytes": preflight.get("total_tensor_bytes") if preflight else None,
            "physical_vram_bytes": preflight.get("physical_vram_bytes") if preflight else None,
        },
        measured=official_full_run and preflight is not None,
    )

    execution_count = int(correctness.get("deterministic_execution_count", 0))
    token_identity = correctness.get("token_identity_rate")
    correctness_passed = (
        official_full_run
        and execution_count >= acceptance.minimum_correctness_executions
        and token_identity == 1.0
        and bool(correctness.get("fixture_checks_passed"))
    )
    gate2 = _gate(
        2,
        "correctness",
        correctness_passed if official_full_run else None,
        [
            f"deterministic executions: {execution_count} (required {acceptance.minimum_correctness_executions})",
            f"token identity rate: {token_identity if token_identity is not None else 'unavailable'}",
            f"fixture equivalence checks: {correctness.get('fixture_checks_passed')}",
        ],
        {"execution_count": execution_count, "token_identity_rate": token_identity},
        measured=official_full_run and execution_count > 0,
    )

    by_config = {str(row["configuration"]): row for row in ablations}
    stock = by_config.get("A", {})
    adaptive = by_config.get("G", {})

    def gain(key: str, *, lower: bool = False) -> float | None:
        before, after = stock.get(key), adaptive.get(key)
        if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
            return None
        if before == 0 or (lower and after == 0):
            return None
        return 1 - after / before if lower else after / before - 1

    decode_gain = gain("decode_tokens_per_second")
    ttft_gain = gain("ttft_32k", lower=True)
    mixed_gain = gain("mixed_verified_tokens_per_second")
    stock_p95 = stock.get("interactive_p95")
    adaptive_p95 = adaptive.get("interactive_p95")
    mixed_latency_ok = (
        isinstance(stock_p95, (int, float))
        and isinstance(adaptive_p95, (int, float))
        and adaptive_p95 <= stock_p95 * (1 + acceptance.maximum_interactive_p95_increase_fraction)
    )
    performance_path = (
        (decode_gain is not None and decode_gain >= acceptance.minimum_decode_gain_fraction)
        or (ttft_gain is not None and ttft_gain >= acceptance.minimum_ttft_32k_reduction_fraction)
        or (
            mixed_gain is not None
            and mixed_gain >= acceptance.minimum_mixed_gain_fraction
            and mixed_latency_ok
        )
    )
    primary = {
        "decode": (
            stock.get("decode_tokens_per_second"),
            adaptive.get("decode_tokens_per_second"),
            False,
        ),
        "prefill": (stock.get("ttft_32k"), adaptive.get("ttft_32k"), True),
        "mixed": (
            stock.get("mixed_verified_tokens_per_second"),
            adaptive.get("mixed_verified_tokens_per_second"),
            False,
        ),
    }
    non_regression = True
    non_regression_reasons: list[str] = []
    for name, (before, after, lower) in primary.items():
        if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
            non_regression = False
            non_regression_reasons.append(f"{name}: unavailable")
            continue
        ratio = (after / before) if not lower else (1 - (after - before) / before)
        passed = (
            after <= before * (1 + acceptance.maximum_other_workload_regression_fraction)
            if lower
            else after >= before * (1 - acceptance.maximum_other_workload_regression_fraction)
        )
        non_regression &= passed
        non_regression_reasons.append(f"{name}: {'pass' if passed else 'fail'} ({ratio:.6f}x)")
    performance_passed = (
        official_full_run and performance_path and non_regression and token_identity == 1.0
    )
    gate3 = _gate(
        3,
        "adaptive performance",
        performance_passed if official_full_run else None,
        [
            f"decode gain: {decode_gain}",
            f"32K TTFT reduction: {ttft_gain}",
            f"mixed gain: {mixed_gain}; latency constraint: {mixed_latency_ok}",
            *non_regression_reasons,
        ],
        {
            "decode_gain_fraction": decode_gain,
            "ttft_32k_reduction_fraction": ttft_gain,
            "mixed_gain_fraction": mixed_gain,
            "mixed_latency_constraint_passed": mixed_latency_ok,
            "other_workloads_non_regression": non_regression,
        },
        measured=official_full_run and bool(observations),
    )

    regret = planner_quality.get("regret_fraction")
    ranking = planner_quality.get("ranking_agreement_fraction")
    explanations = bool(planner_quality.get("all_selected_placements_explained"))
    rejects = bool(planner_quality.get("can_reject_harmful_techniques"))
    gate4_pass = (
        official_full_run
        and isinstance(regret, (int, float))
        and regret <= acceptance.maximum_planner_regret_fraction
        and isinstance(ranking, (int, float))
        and ranking >= 0.5
        and explanations
        and rejects
    )
    gate4 = _gate(
        4,
        "planner quality",
        gate4_pass if official_full_run else None,
        [
            f"planner regret: {regret}",
            f"pairwise ranking agreement: {ranking}",
            f"placement explanations complete: {explanations}",
            f"harmful-technique rejection demonstrated: {rejects}",
        ],
        planner_quality,
        measured=official_full_run and isinstance(regret, (int, float)),
    )

    capacity_utility = gate1.status == GateStatus.PASS
    performance_cpu_utility = bool(residency.get("positive_cpu_performance_utility"))
    gate5_pass = capacity_utility or performance_cpu_utility
    gate5 = _gate(
        5,
        "positive CPU utility",
        gate5_pass if official_full_run else None,
        [
            "system RAM supplied otherwise-impossible model capacity"
            if capacity_utility
            else "capacity utility was not established",
            "CPU compute improved a measured workload"
            if performance_cpu_utility
            else "no measured CPU-compute performance gain was established",
        ],
        {
            "capacity_utility": capacity_utility,
            "performance_utility": performance_cpu_utility,
        },
        measured=official_full_run and (capacity_utility or bool(observations)),
    )

    architecture_pass = bool(architecture_audit.get("complete"))
    gate6 = _gate(
        6,
        "reusable architecture",
        architecture_pass,
        list(architecture_audit.get("reasons", [])),
        {
            "implemented_feature_count": sum(
                bool(value) for value in architecture_audit.get("features", {}).values()
            ),
            "required_feature_count": len(architecture_audit.get("features", {})),
            "required_artifacts_complete_at_gate_time": bool(
                architecture_audit.get("artifact_audit", {}).get("complete")
            ),
        },
        measured=False,
    )
    return [gate1, gate2, gate3, gate4, gate5, gate6]
