"""Calibration/held-out planner evidence for corrected Experiment 007 metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from swarm_inference.experiments.experiment_007_moe_correction import (
    CANONICAL_EXECUTOR_ID,
)

CorrectedRole = Literal["moe_expert", "background_inference", "idle"]


@dataclass(frozen=True, slots=True)
class PlannerPoint:
    point_id: str
    role: CorrectedRole
    features: tuple[float, ...]
    measured_utility: float
    metadata: dict[str, Any]


def _moe_utility(row: dict[str, Any], minimum_retained: float) -> float:
    if not bool(row.get("output_correctness_passed")):
        return -1.0
    if int(row.get("cpu_expert_calls", 0)) <= 0:
        return -1.0
    retained_margin = float(row["throughput_retained_fraction"]) - minimum_retained
    memory_fraction = float(row["gpu_memory_saved_bytes"]) / max(
        float(row["baseline_gpu_expert_weight_bytes"]), 1.0
    )
    return retained_margin + 0.10 * memory_fraction


def _background_utility(row: dict[str, Any]) -> float:
    gain = float(row["combined_gain_fraction"])
    throughput_penalty = max(0.0, -float(row["gpu_throughput_change_fraction"]))
    latency_penalty = max(0.0, float(row["gpu_p95_latency_change_fraction"]))
    return gain - throughput_penalty - latency_penalty


def corrected_planner_points(
    moe_rows: list[dict[str, Any]],
    background_rows: list[dict[str, Any]],
    *,
    minimum_expert_retained_fraction: float,
) -> list[PlannerPoint]:
    points: list[PlannerPoint] = []
    for row in moe_rows:
        if row.get("arm") != "hybrid_gpu_cpu" or row.get("weight_format") != "bfloat16":
            continue
        if row.get("benchmark_mode") != "natural_routing":
            continue
        if row.get("executor_id") != CANONICAL_EXECUTOR_ID or not bool(
            row.get("matched_baseline_used")
        ):
            raise ValueError("superseded or unmatched MoE metrics cannot calibrate the planner")
        count = int(row["cpu_expert_count"])
        dispatch = float(row["cpu_dispatch_fraction"])
        memory_fraction = float(row["gpu_memory_saved_bytes"]) / max(
            float(row["baseline_gpu_expert_weight_bytes"]), 1.0
        )
        point_id = f"moe:{row['placement_policy']}:{count}"
        points.append(
            PlannerPoint(
                point_id=point_id,
                role="moe_expert",
                features=(1.0, count / 16.0, dispatch, memory_fraction),
                measured_utility=_moe_utility(row, minimum_expert_retained_fraction),
                metadata={
                    "placement_policy": row["placement_policy"],
                    "cpu_expert_count": count,
                    "cpu_dispatch_fraction": dispatch,
                    "source_metric_version": "matched_moe_v1",
                },
            )
        )
    for row in background_rows:
        if row.get("traffic_mode") != "closed_loop":
            continue
        if not bool(row.get("fixed_window_formula_used")):
            raise ValueError("superseded fixed-job background metrics cannot calibrate the planner")
        gpu_concurrency = int(row["gpu_concurrency"])
        cpu_concurrency = int(row["cpu_concurrency"])
        point_id = f"background:g{gpu_concurrency}:c{cpu_concurrency}"
        points.append(
            PlannerPoint(
                point_id=point_id,
                role="background_inference",
                features=(
                    1.0,
                    gpu_concurrency / 16.0,
                    cpu_concurrency / 4.0,
                    gpu_concurrency * cpu_concurrency / 64.0,
                ),
                measured_utility=_background_utility(row),
                metadata={
                    "gpu_concurrency": gpu_concurrency,
                    "cpu_concurrency": cpu_concurrency,
                    "source_metric_version": "fixed_window_token_accounting_v1",
                },
            )
        )
    return points


def split_calibration_and_held_out(
    points: list[PlannerPoint],
) -> tuple[list[PlannerPoint], list[PlannerPoint]]:
    calibration: list[PlannerPoint] = []
    held_out: list[PlannerPoint] = []
    for point in points:
        if point.role == "moe_expert":
            count = int(point.metadata["cpu_expert_count"])
            target = held_out if count in {2, 8} else calibration
        elif point.role == "background_inference":
            gpu_concurrency = int(point.metadata["gpu_concurrency"])
            target = held_out if gpu_concurrency == 4 else calibration
        else:
            target = held_out
        target.append(point)
    calibration_ids = {item.point_id for item in calibration}
    held_out_ids = {item.point_id for item in held_out}
    if calibration_ids & held_out_ids:
        raise RuntimeError("planner calibration and held-out points overlap")
    if not calibration or not held_out:
        raise RuntimeError("planner requires non-empty calibration and held-out partitions")
    return calibration, held_out


def _fit_role(points: list[PlannerPoint]) -> np.ndarray:
    if not points:
        raise ValueError("cannot fit a planner role without calibration points")
    matrix = np.asarray([item.features for item in points], dtype=np.float64)
    target = np.asarray([item.measured_utility for item in points], dtype=np.float64)
    coefficients, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    return coefficients


def evaluate_corrected_planner(
    points: list[PlannerPoint],
    *,
    maximum_regret_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    calibration, held_out = split_calibration_and_held_out(points)
    coefficients = {
        role: _fit_role([item for item in calibration if item.role == role])
        for role in ("moe_expert", "background_inference")
    }
    calibration_rows = [
        {
            "classification": "measured_mixed_backend",
            "point_id": item.point_id,
            "split": "calibration",
            "role": item.role,
            "features": list(item.features),
            "measured_utility": item.measured_utility,
            "observed_before_model_fit": True,
            **item.metadata,
        }
        for item in calibration
    ]
    held_out_rows: list[dict[str, Any]] = []
    for item in held_out:
        predicted = float(np.dot(coefficients[item.role], np.asarray(item.features)))
        held_out_rows.append(
            {
                "classification": "measured_mixed_backend",
                "point_id": item.point_id,
                "split": "held_out",
                "role": item.role,
                "features": list(item.features),
                "predicted_utility": predicted,
                "measured_utility": item.measured_utility,
                "prediction_error": item.measured_utility - predicted,
                "observed_before_prediction": False,
                **item.metadata,
            }
        )
    candidates = [
        *held_out_rows,
        {
            "classification": "measured_mixed_backend",
            "point_id": "idle",
            "split": "held_out_control",
            "role": "idle",
            "predicted_utility": 0.0,
            "measured_utility": 0.0,
            "prediction_error": 0.0,
            "observed_before_prediction": False,
        },
    ]
    selected = max(candidates, key=lambda item: float(item["predicted_utility"]))
    best = max(candidates, key=lambda item: float(item["measured_utility"]))
    regret = float(best["measured_utility"]) - float(selected["measured_utility"])
    regret_fraction = regret / max(float(best["measured_utility"]), 1e-12)
    if float(best["measured_utility"]) <= 0:
        regret_fraction = 0.0 if selected["role"] == "idle" else math.inf
    regret_row = {
        "classification": "measured_mixed_backend",
        "predicted_best_role": selected["role"],
        "predicted_best_point_id": selected["point_id"],
        "measured_best_role": best["role"],
        "measured_best_point_id": best["point_id"],
        "planner_selected_role": selected["role"],
        "planner_selected_point_id": selected["point_id"],
        "planner_selected_predicted_utility": selected["predicted_utility"],
        "planner_selected_measured_utility": selected["measured_utility"],
        "best_measured_utility": best["measured_utility"],
        "planner_regret": regret,
        "planner_regret_fraction": regret_fraction,
        "maximum_regret_fraction": maximum_regret_fraction,
        "passes": regret_fraction <= maximum_regret_fraction,
        "calibration_and_held_out_disjoint": True,
        "held_out_observations_used_for_selection": False,
    }
    model = {
        "model_type": "role_specific_ordinary_least_squares",
        "coefficients": {key: value.tolist() for key, value in coefficients.items()},
        "utility_definitions": {
            "moe_expert": "retained_throughput_margin_above_0.70 + 0.10 * memory_fraction",
            "background_inference": "combined_gain - throughput_interference - p95_interference",
            "idle": "0",
        },
        "calibration_point_count": len(calibration_rows),
        "held_out_point_count": len(held_out_rows),
    }
    return calibration_rows, held_out_rows, [regret_row], model
