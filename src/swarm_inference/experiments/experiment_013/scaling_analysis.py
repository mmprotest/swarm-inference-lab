"""Fit preregistered warm-latency scaling models with bootstrap uncertainty."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_012.baseline_harness import _write_json

Feature = Callable[[int], float]
FEATURES: dict[str, Feature | None] = {
    "constant": None,
    "log_n": lambda n: math.log2(n),
    "n": float,
    "n_log_n": lambda n: n * math.log2(n),
}


def _fit(name: str, counts: list[int], values: list[float]) -> dict[str, Any]:
    feature = FEATURES[name]
    sample_count = len(values)
    mean_y = statistics.fmean(values)
    if feature is None:
        intercept = mean_y
        slope = 0.0
        predictions = [intercept] * sample_count
        parameter_count = 1
    else:
        x = [feature(count) for count in counts]
        mean_x = statistics.fmean(x)
        denominator = sum((item - mean_x) ** 2 for item in x)
        slope = sum(
            (item - mean_x) * (value - mean_y) for item, value in zip(x, values, strict=True)
        )
        slope /= denominator
        intercept = mean_y - slope * mean_x
        predictions = [intercept + slope * item for item in x]
        parameter_count = 2
    residuals = [value - predicted for value, predicted in zip(values, predictions, strict=True)]
    rss = sum(residual * residual for residual in residuals)
    tss = sum((value - mean_y) ** 2 for value in values)
    safe_rss = max(rss, 1e-300)
    aic = sample_count * math.log(safe_rss / sample_count) + 2 * parameter_count
    correction = 2 * parameter_count * (parameter_count + 1) / (sample_count - parameter_count - 1)
    return {
        "model": name,
        "sample_count": sample_count,
        "parameter_count": parameter_count,
        "intercept_ms": intercept,
        "slope": slope,
        "rss": rss,
        "rmse_ms": math.sqrt(rss / sample_count),
        "r_squared": 1.0 - rss / tss if tss else 1.0,
        "aic": aic,
        "aicc": aic + correction,
        "predictions_ms": dict(zip((str(item) for item in counts), predictions, strict=True)),
        "residuals_ms": dict(zip((str(item) for item in counts), residuals, strict=True)),
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def analyze(
    *,
    benchmark_summary: Path,
    operations_jsonl: Path,
    output: Path,
    bootstrap_repetitions: int,
    seed: int,
) -> dict[str, Any]:
    summary = json.loads(benchmark_summary.read_text(encoding="utf-8"))
    counts = [int(row["worker_count"]) for row in summary["summaries"]]
    values = [float(row["warm_latency_p50_ms"]) for row in summary["summaries"]]
    fits = [_fit(name, counts, values) for name in FEATURES]
    fits.sort(key=lambda row: float(row["aicc"]))
    minimum_aicc = float(fits[0]["aicc"])
    for fit in fits:
        fit["delta_aicc"] = float(fit["aicc"]) - minimum_aicc
    weight_denominator = sum(math.exp(-0.5 * float(fit["delta_aicc"])) for fit in fits)
    for fit in fits:
        fit["akaike_weight"] = math.exp(-0.5 * float(fit["delta_aicc"])) / weight_denominator

    trials: dict[int, list[float]] = defaultdict(list)
    with operations_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not row.get("warmup") and row.get("status") == "ok":
                trials[int(row["worker_count"])].append(float(row["end_to_end_latency_ms"]))
    if any(not trials[count] for count in counts):
        raise ValueError("every scale must have successful warm trials")

    rng = random.Random(seed)
    winner_counts = {name: 0 for name in FEATURES}
    slopes: dict[str, list[float]] = {name: [] for name in FEATURES}
    intercepts: dict[str, list[float]] = {name: [] for name in FEATURES}
    for _ in range(bootstrap_repetitions):
        boot_values = [
            statistics.median(rng.choices(trials[count], k=len(trials[count]))) for count in counts
        ]
        boot_fits = [_fit(name, counts, boot_values) for name in FEATURES]
        winner = min(boot_fits, key=lambda row: float(row["aicc"]))
        winner_counts[str(winner["model"])] += 1
        for fit in boot_fits:
            name = str(fit["model"])
            slopes[name].append(float(fit["slope"]))
            intercepts[name].append(float(fit["intercept_ms"]))

    bootstrap = {
        name: {
            "winner_count": winner_counts[name],
            "winner_fraction": winner_counts[name] / bootstrap_repetitions,
            "slope_ci95": [
                _percentile(slopes[name], 0.025),
                _percentile(slopes[name], 0.975),
            ],
            "intercept_ms_ci95": [
                _percentile(intercepts[name], 0.025),
                _percentile(intercepts[name], 0.975),
            ],
        }
        for name in FEATURES
    }
    result = {
        "experiment_id": "013",
        "hypothesis_id": "H013-005",
        "architecture": summary["architecture"],
        "worker_counts": counts,
        "warm_p50_ms": values,
        "criterion": "minimum AICc over OLS models with Gaussian residual variance",
        "best_model": fits[0]["model"],
        "fits": fits,
        "bootstrap": {
            "method": "within-scale trial resampling followed by median and model refit",
            "repetitions": bootstrap_repetitions,
            "seed": seed,
            "models": bootstrap,
        },
        "trial_counts": {str(count): len(trials[count]) for count in counts},
        "conclusion": (
            "materially_improved_over_n_log_n"
            if fits[0]["model"] != "n_log_n"
            else "n_log_n_remains_preferred"
        ),
    }
    _write_json(output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-summary", type=Path, required=True)
    parser.add_argument("--operations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13005)
    args = parser.parse_args(argv)
    result = analyze(
        benchmark_summary=args.benchmark_summary.resolve(),
        operations_jsonl=args.operations.resolve(),
        output=args.output.resolve(),
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
