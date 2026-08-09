"""Quantitative scaling fits and raw-row summaries for Experiment 012."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

_CANDIDATES: dict[str, tuple[int, Callable[[float, int], float]]] = {
    "constant": (1, lambda worker_count, branch_factor: 0.0),
    "log2_n": (2, lambda worker_count, branch_factor: math.log2(worker_count)),
    "log_b_n": (2, lambda worker_count, branch_factor: math.log(worker_count, branch_factor)),
    "linear_n": (2, lambda worker_count, branch_factor: worker_count),
    "n_log2_n": (
        2,
        lambda worker_count, branch_factor: worker_count * math.log2(worker_count),
    ),
}


def _linear_fit(x_values: list[float], y_values: list[float]) -> tuple[float, float]:
    if len(x_values) != len(y_values) or not x_values:
        raise ValueError("fit inputs must be non-empty and equal length")
    if all(value == x_values[0] for value in x_values):
        return statistics.mean(y_values), 0.0
    x_mean = statistics.mean(x_values)
    y_mean = statistics.mean(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    slope = (
        sum(
            (x_value - x_mean) * (y_value - y_mean)
            for x_value, y_value in zip(x_values, y_values, strict=True)
        )
        / denominator
    )
    return y_mean - slope * x_mean, slope


def _fit_candidate(
    *,
    name: str,
    observations: list[tuple[int, float]],
    branch_factor: int,
) -> dict[str, Any]:
    parameter_count, transform = _CANDIDATES[name]
    worker_counts = [item[0] for item in observations]
    y_values = [item[1] for item in observations]
    x_values = [transform(worker_count, branch_factor) for worker_count in worker_counts]
    intercept, slope = _linear_fit(x_values, y_values)
    predictions = [intercept + slope * value for value in x_values]
    residuals = [
        observed - predicted for observed, predicted in zip(y_values, predictions, strict=True)
    ]
    rss = sum(value * value for value in residuals)
    rmse = math.sqrt(rss / len(y_values))
    total_sum_squares = sum((value - statistics.mean(y_values)) ** 2 for value in y_values)
    r_squared = 1.0 - rss / total_sum_squares if total_sum_squares else (1.0 if rss == 0 else 0.0)
    effective_rss = max(rss, 1e-24)
    aic = len(y_values) * math.log(effective_rss / len(y_values)) + 2 * parameter_count
    aicc = (
        aic + (2 * parameter_count * (parameter_count + 1)) / (len(y_values) - parameter_count - 1)
        if len(y_values) > parameter_count + 1
        else None
    )
    loo_errors: list[float] = []
    if len(observations) >= 3:
        for excluded in range(len(observations)):
            train_x = [value for index, value in enumerate(x_values) if index != excluded]
            train_y = [value for index, value in enumerate(y_values) if index != excluded]
            loo_intercept, loo_slope = _linear_fit(train_x, train_y)
            predicted = loo_intercept + loo_slope * x_values[excluded]
            loo_errors.append(y_values[excluded] - predicted)
    loo_rmse = (
        math.sqrt(sum(value * value for value in loo_errors) / len(loo_errors))
        if loo_errors
        else None
    )
    return {
        "relationship": name,
        "parameter_count": parameter_count,
        "intercept": intercept,
        "coefficient": slope,
        "predictions": predictions,
        "residuals": residuals,
        "rss": rss,
        "rmse": rmse,
        "r_squared": r_squared,
        "aic": aic,
        "aicc": aicc,
        "leave_one_out_rmse": loo_rmse,
    }


def fit_scaling(observations: list[tuple[int, float]], *, branch_factor: int) -> dict[str, Any]:
    ordered = sorted(observations)
    if len({worker_count for worker_count, _ in ordered}) < 3:
        raise ValueError("at least three distinct worker counts are required for scaling fits")
    fits = [
        _fit_candidate(name=name, observations=ordered, branch_factor=branch_factor)
        for name in _CANDIDATES
    ]
    # AICc is undefined when n <= k + 1. Do not accidentally crown the
    # one-parameter constant model merely because competing two-parameter
    # fits have no AICc at a three-scale discriminating stage.
    ranking_criterion = (
        "aicc" if all(item["aicc"] is not None for item in fits) else "leave_one_out_rmse"
    )
    ranked = sorted(
        fits,
        key=lambda item: (
            float("inf") if item[ranking_criterion] is None else item[ranking_criterion],
            item["rss"],
        ),
    )
    return {
        "observations": [
            {"worker_count": worker_count, "value": value} for worker_count, value in ordered
        ],
        "fits": fits,
        "best_relationship": ranked[0]["relationship"],
        "ranking_criterion": ranking_criterion,
        "best_aicc": ranked[0]["aicc"],
        "largest_n": ordered[-1][0],
        "largest_n_value": ordered[-1][1],
    }


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def analyze_trials(
    rows: list[dict[str, Any]],
    *,
    branch_factor: int,
    metrics: tuple[str, ...],
) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if not row.get("warmup") and row.get("status") == "ok" and row.get("correctness")
    ]
    modes = sorted({str(row["mode"]) for row in eligible})
    results: dict[str, Any] = {}
    for mode in modes:
        mode_rows = [row for row in eligible if row["mode"] == mode]
        metric_results: dict[str, Any] = {}
        for metric in metrics:
            observations = []
            for worker_count in sorted({int(row["worker_count"]) for row in mode_rows}):
                selected = [
                    float(row[metric])
                    for row in mode_rows
                    if int(row["worker_count"]) == worker_count and row.get(metric) is not None
                ]
                if selected:
                    observations.append((worker_count, statistics.median(selected)))
            if len(observations) >= 3:
                metric_results[metric] = fit_scaling(observations, branch_factor=branch_factor)
        results[mode] = metric_results
    return {
        "schema_version": "1.0",
        "branch_factor": branch_factor,
        "eligible_trial_count": len(eligible),
        "excluded_trial_count": len(rows) - len(eligible),
        "modes": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit Experiment 012 scaling relationships")
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch-factor", type=int, default=8)
    parser.add_argument(
        "--metrics",
        default=(
            "root_messages_total,root_bytes_total,root_serial_waits,root_direct_degree,"
            "root_cpu_ns,end_to_end_latency_ms,total_messages,total_bytes,hierarchy_depth"
        ),
    )
    args = parser.parse_args(argv)
    analysis = analyze_trials(
        _load_rows(args.trials),
        branch_factor=args.branch_factor,
        metrics=tuple(item.strip() for item in args.metrics.split(",") if item.strip()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(analysis, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["analyze_trials", "fit_scaling"]
