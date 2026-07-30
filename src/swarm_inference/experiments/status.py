"""Truthful Experiment 001 integrity and hypothesis status evaluation."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Literal

from swarm_inference.config.models import DataPlaneMode, ExperimentConfig

Status = Literal["PASS", "FAIL"]


@dataclass(frozen=True, slots=True)
class StatusCriterion:
    name: str
    status: Status
    observed: Any
    required: Any
    reason: str
    mandatory_for: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "observed": self.observed,
            "required": self.required,
            "reason": self.reason,
            "mandatory_for": list(self.mandatory_for),
        }


def _criterion(
    name: str,
    passed: bool,
    *,
    observed: Any,
    required: Any,
    reason: str,
    mandatory_for: tuple[str, ...],
) -> StatusCriterion:
    return StatusCriterion(
        name=name,
        status="PASS" if passed else "FAIL",
        observed=observed,
        required=required,
        reason=reason,
        mandatory_for=mandatory_for,
    )


def _cv(values: list[float]) -> float:
    if not values:
        return float("inf")
    mean = statistics.fmean(values)
    if mean == 0:
        return 0.0 if all(value == 0 for value in values) else float("inf")
    return (statistics.stdev(values) / mean) if len(values) > 1 else 0.0


def evaluate_matrix_statuses(
    *,
    config: ExperimentConfig,
    point_rows: list[dict[str, Any]],
    worker_counts: list[int],
    concurrency_counts: list[int],
    repeats: int,
    measured_duration_s: float,
    child_validation_errors: list[str],
    dependency_mutation_failure: bool = False,
) -> tuple[dict[str, Status], list[dict[str, Any]], dict[str, Any]]:
    """Evaluate completeness and the scaling claim as separate propositions."""

    expected_keys = {
        (workers, concurrency, repeat)
        for workers in worker_counts
        for concurrency in concurrency_counts
        for repeat in range(1, repeats + 1)
    }
    observed_keys = {
        (
            int(row["node_count"]),
            int(row["concurrent_request_count"]),
            int(row["repeat"]),
        )
        for row in point_rows
    }
    missing = sorted(expected_keys - observed_keys)
    duplicates = len(point_rows) - len(observed_keys)
    minimum_duration = min(
        (float(row["measured_duration_s"]) for row in point_rows),
        default=0.0,
    )
    all_complete = all(float(row.get("completion_fraction", 0.0)) == 1.0 for row in point_rows)
    all_correct = all(
        float(row.get("committed_token_correctness", row.get("completion_fraction", 0.0))) == 1.0
        for row in point_rows
    )
    all_direct_labels = all(
        row.get("data_plane_mode") == config.data_plane.value for row in point_rows
    )
    coordinator_activation_bytes = sum(
        int(row.get("coordinator_activation_bytes", 0)) for row in point_rows
    )
    stream_contract_violations = [
        str(row.get("point_id", "unknown"))
        for row in point_rows
        if int(row.get("peer_streams_created", 0))
        > int(row.get("active_peer_pairs", 0)) + int(row.get("peer_stream_reconnects", 0))
    ]

    primary_concurrency = max(concurrency_counts) if concurrency_counts else 0
    primary_rows = [
        row for row in point_rows if int(row["concurrent_request_count"]) == primary_concurrency
    ]
    throughput_by_workers: dict[int, list[float]] = {
        workers: [
            float(row["aggregate_verified_output_tokens_s"])
            for row in primary_rows
            if int(row["node_count"]) == workers
        ]
        for workers in worker_counts
    }
    medians = {
        workers: (float(statistics.median(values)) if values else 0.0)
        for workers, values in throughput_by_workers.items()
    }
    primary_cvs = {workers: _cv(values) for workers, values in throughput_by_workers.items()}
    throughput_2 = medians.get(2, 0.0)
    throughput_4 = medians.get(4, 0.0)
    throughput_8 = medians.get(8, 0.0)
    ratios = {
        "2_to_4": throughput_4 / throughput_2 if throughput_2 > 0 else 0.0,
        "4_to_8": throughput_8 / throughput_4 if throughput_4 > 0 else 0.0,
        "2_to_8": throughput_8 / throughput_2 if throughput_2 > 0 else 0.0,
    }
    minimum_meaningful_fraction = min(
        (float(row.get("meaningful_replica_fraction", 0.0)) for row in primary_rows),
        default=0.0,
    )
    maximum_replica_imbalance = max(
        (float(row.get("replica_imbalance_ratio", float("inf"))) for row in primary_rows),
        default=float("inf"),
    )
    prediction_errors = [
        float(row.get("prediction_error_fraction", float("inf"))) for row in point_rows
    ]
    median_prediction_error = (
        float(statistics.median(prediction_errors)) if prediction_errors else float("inf")
    )
    criteria = [
        _criterion(
            "integrity:complete_matrix",
            not missing and duplicates == 0 and len(point_rows) == len(expected_keys),
            observed={
                "observed_points": len(point_rows),
                "expected_points": len(expected_keys),
                "missing": missing,
                "duplicates": duplicates,
            },
            required="every configured worker/concurrency/repeat point exactly once",
            reason="Incomplete or duplicated points invalidate the matrix.",
            mandatory_for=("experiment_integrity", "scaling_hypothesis"),
        ),
        _criterion(
            "integrity:measurement_duration",
            minimum_duration >= measured_duration_s * 0.95,
            observed=minimum_duration,
            required=f">= {measured_duration_s * 0.95:.3f}s at every point",
            reason="Every primary measurement must cover the configured interval.",
            mandatory_for=("experiment_integrity", "scaling_hypothesis"),
        ),
        _criterion(
            "integrity:child_artifacts",
            not child_validation_errors,
            observed=child_validation_errors,
            required="no child artifact validation errors",
            reason="Every child result must retain its evidence bundle.",
            mandatory_for=("experiment_integrity", "scaling_hypothesis"),
        ),
        _criterion(
            "integrity:dependency_preservation",
            not dependency_mutation_failure,
            observed=dependency_mutation_failure,
            required=False,
            reason="The experiment may not remove unrelated optional dependencies.",
            mandatory_for=("experiment_integrity", "scaling_hypothesis"),
        ),
        _criterion(
            "correctness:committed_tokens",
            all_correct,
            observed=min(
                (
                    float(
                        row.get(
                            "committed_token_correctness",
                            row.get("completion_fraction", 0.0),
                        )
                    )
                    for row in point_rows
                ),
                default=0.0,
            ),
            required=1.0,
            reason="Only correctly committed tokens support the hypothesis.",
            mandatory_for=("correctness", "scaling_hypothesis"),
        ),
        _criterion(
            "correctness:completion",
            all_complete,
            observed=min(
                (float(row.get("completion_fraction", 0.0)) for row in point_rows),
                default=0.0,
            ),
            required=1.0,
            reason="Failure-free runs require every admitted request to complete.",
            mandatory_for=("correctness", "scaling_hypothesis"),
        ),
        _criterion(
            "direct:explicit_mode",
            all_direct_labels,
            observed=sorted({str(row.get("data_plane_mode")) for row in point_rows}),
            required=config.data_plane.value,
            reason="The selected data plane must be explicit in every child.",
            mandatory_for=("direct_data_plane", "scaling_hypothesis"),
        ),
        _criterion(
            "direct:no_coordinator_activation_relay",
            (config.data_plane != DataPlaneMode.DIRECT or coordinator_activation_bytes == 0),
            observed=coordinator_activation_bytes,
            required=(0 if config.data_plane == DataPlaneMode.DIRECT else "explicit relay mode"),
            reason="Direct mode may not relay intermediate activation payloads.",
            mandatory_for=("direct_data_plane", "scaling_hypothesis"),
        ),
        _criterion(
            "direct:persistent_stream_reuse",
            not stream_contract_violations,
            observed=stream_contract_violations,
            required="streams <= active peer pairs + reconnects at every point",
            reason="Stream creation must scale with peer pairs, not token operations.",
            mandatory_for=("direct_data_plane", "scaling_hypothesis"),
        ),
        _criterion(
            "utilisation:meaningful_replicas",
            minimum_meaningful_fraction >= config.acceptance.min_meaningful_replica_fraction,
            observed=minimum_meaningful_fraction,
            required=config.acceptance.min_meaningful_replica_fraction,
            reason="Assigned replicas must perform at least 5% of their stage work.",
            mandatory_for=("replica_utilisation", "scaling_hypothesis"),
        ),
        _criterion(
            "utilisation:replica_balance",
            maximum_replica_imbalance <= config.acceptance.max_replica_imbalance_ratio,
            observed=maximum_replica_imbalance,
            required=config.acceptance.max_replica_imbalance_ratio,
            reason="The busiest meaningful replica may not starve its peers.",
            mandatory_for=("replica_utilisation", "scaling_hypothesis"),
        ),
        _criterion(
            "capacity:median_prediction_error",
            median_prediction_error <= config.acceptance.max_capacity_prediction_error,
            observed=median_prediction_error,
            required=config.acceptance.max_capacity_prediction_error,
            reason="Capacity estimates must reflect the measured critical path.",
            mandatory_for=("capacity_prediction", "scaling_hypothesis"),
        ),
        _criterion(
            "scaling:ratio_2_to_4",
            ratios["2_to_4"] >= config.acceptance.min_ratio_2_to_4,
            observed=ratios["2_to_4"],
            required=config.acceptance.min_ratio_2_to_4,
            reason="Two replicas per stage must materially beat one.",
            mandatory_for=("scaling_hypothesis",),
        ),
        _criterion(
            "scaling:ratio_4_to_8",
            ratios["4_to_8"] >= config.acceptance.min_ratio_4_to_8,
            observed=ratios["4_to_8"],
            required=config.acceptance.min_ratio_4_to_8,
            reason="Four replicas per stage must materially beat two.",
            mandatory_for=("scaling_hypothesis",),
        ),
        _criterion(
            "scaling:ratio_2_to_8",
            ratios["2_to_8"] >= config.acceptance.min_ratio_2_to_8,
            observed=ratios["2_to_8"],
            required=config.acceptance.min_ratio_2_to_8,
            reason="The end-to-end gain must exceed the configured floor.",
            mandatory_for=("scaling_hypothesis",),
        ),
        _criterion(
            "scaling:repeatability",
            all(value <= config.acceptance.max_primary_cv for value in primary_cvs.values()),
            observed=primary_cvs,
            required=config.acceptance.max_primary_cv,
            reason="A single unusually fast repeat may not determine PASS.",
            mandatory_for=("scaling_hypothesis",),
        ),
        _criterion(
            "scaling:minimum_milestone",
            all(
                value >= config.acceptance.minimum_aggregate_verified_tokens_s
                for value in medians.values()
            ),
            observed=medians,
            required=config.acceptance.minimum_aggregate_verified_tokens_s,
            reason="The legacy milestone remains necessary but is not sufficient.",
            mandatory_for=("scaling_hypothesis",),
        ),
    ]

    def component(name: str) -> Status:
        relevant = [item for item in criteria if name in item.mandatory_for]
        return "PASS" if relevant and all(item.status == "PASS" for item in relevant) else "FAIL"

    statuses: dict[str, Status] = {
        "experiment_integrity_status": component("experiment_integrity"),
        "correctness_status": component("correctness"),
        "direct_data_plane_status": component("direct_data_plane"),
        "replica_utilisation_status": component("replica_utilisation"),
        "capacity_prediction_status": component("capacity_prediction"),
        "scaling_hypothesis_status": component("scaling_hypothesis"),
        "overall_status": "FAIL",
    }
    mandatory = [
        "experiment_integrity_status",
        "correctness_status",
        "direct_data_plane_status",
        "replica_utilisation_status",
        "capacity_prediction_status",
        "scaling_hypothesis_status",
    ]
    statuses["overall_status"] = (
        "PASS" if all(statuses[name] == "PASS" for name in mandatory) else "FAIL"
    )
    evidence = {
        "primary_concurrency": primary_concurrency,
        "median_throughput_by_worker_count": medians,
        "primary_cv_by_worker_count": primary_cvs,
        "scaling_ratios": ratios,
        "minimum_meaningful_replica_fraction": minimum_meaningful_fraction,
        "maximum_replica_imbalance_ratio": maximum_replica_imbalance,
        "coordinator_activation_bytes": coordinator_activation_bytes,
        "median_capacity_prediction_error": median_prediction_error,
    }
    return statuses, [item.to_dict() for item in criteria], evidence
