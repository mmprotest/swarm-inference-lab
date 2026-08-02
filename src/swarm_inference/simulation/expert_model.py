"""Measured-cost expert simulator with leakage-safe held-out validation."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

EXPERT_COST_FEATURES = (
    "worker_compute_ns",
    "worker_queue_ns",
    "cpu_affinity_cost_ns",
    "gpu_compute_ns",
    "pcie_transfer_ns",
    "cache_cost_ns",
    "storage_read_ns",
    "serialisation_ns",
    "tcp_transport_ns",
    "shared_memory_ns",
    "compression_ns",
    "microshard_compute_ns",
    "reduction_ns",
    "prefill_batching_ns",
    "routing_overlap_savings_ns",
    "failure_recovery_ns",
    "hedging_ns",
    "verification_ns",
    "background_queue_ns",
)


@dataclass(frozen=True, slots=True)
class ExpertCalibrationModel:
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    feature_scales: tuple[float, ...]
    intercept_ns: float
    p95_multiplier: float
    calibration_configuration_ids: tuple[str, ...]
    validation_configuration_ids: tuple[str, ...]
    validation: dict[str, Any]

    @property
    def validated(self) -> bool:
        return bool(self.validation.get("all_gates_pass"))

    def payload(self) -> dict[str, Any]:
        return asdict(self)

    def predict_total_ns(self, row: dict[str, Any]) -> float:
        values = np.asarray([float(row.get(name, 0.0) or 0.0) for name in self.feature_names])
        scales = np.asarray(self.feature_scales, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        prediction = self.intercept_ns + float(np.dot(values / scales, coefficients))
        return max(prediction, 1.0)

    def predict(self, row: dict[str, Any]) -> dict[str, Any]:
        total_ns = self.predict_total_ns(row)
        verified_tokens = float(row.get("verified_tokens", 1.0) or 1.0)
        return {
            "configuration_id": row.get("configuration_id"),
            "predicted_total_ns": total_ns,
            "predicted_throughput": verified_tokens * 1e9 / total_ns,
            "predicted_p95_latency_ms": total_ns / 1e6 * self.p95_multiplier,
            "category": ("SIMULATED_CALIBRATED" if self.validated else "SIMULATED_UNCALIBRATED"),
        }


def deterministic_calibration_split(
    rows: list[dict[str, Any]],
    *,
    validation_fraction: float = 0.25,
    seed: int = 1010,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation fraction must be between zero and one")
    configuration_ids = sorted({str(row["configuration_id"]) for row in rows})
    if len(configuration_ids) < 4:
        raise ValueError("held-out calibration requires at least four measured configurations")
    scored = sorted(
        configuration_ids,
        key=lambda item: hashlib.sha256(f"{seed}|{item}".encode()).digest(),
    )
    validation_count = max(1, round(len(scored) * validation_fraction))
    validation_ids = set(scored[:validation_count])
    calibration = [row for row in rows if str(row["configuration_id"]) not in validation_ids]
    validation = [row for row in rows if str(row["configuration_id"]) in validation_ids]
    if not calibration or not validation:
        raise ValueError("calibration split produced an empty partition")
    return calibration, validation


def _aggregate_configurations(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["configuration_id"]), []).append(row)
    aggregated = []
    for configuration_id, samples in sorted(grouped.items()):
        result: dict[str, Any] = {
            "configuration_id": configuration_id,
            "workload_id": str(samples[0].get("workload_id", "default")),
            "verified_tokens": float(np.median([row.get("verified_tokens", 1) for row in samples])),
        }
        for field in (*EXPERT_COST_FEATURES, "measured_total_ns", "measured_p95_latency_ms"):
            values = [float(row.get(field, 0.0) or 0.0) for row in samples]
            result[field] = float(np.median(values))
        measured_throughputs = [
            float(row["measured_throughput"])
            if row.get("measured_throughput") is not None
            else float(row.get("verified_tokens", 1.0)) * 1e9 / float(row["measured_total_ns"])
            for row in samples
        ]
        result["measured_throughput"] = float(np.median(measured_throughputs))
        aggregated.append(result)
    return aggregated


def _percentage_error(predicted: float, measured: float) -> float:
    return abs(predicted - measured) / max(abs(measured), 1e-12)


def _ranking_agreement(rows: list[dict[str, Any]]) -> float:
    pairs = 0
    agreements = 0
    by_workload: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_workload.setdefault(str(row.get("workload_id", "default")), []).append(row)
    for group in by_workload.values():
        for left_index, left in enumerate(group):
            for right in group[left_index + 1 :]:
                measured_delta = left["measured_throughput"] - right["measured_throughput"]
                predicted_delta = left["predicted_throughput"] - right["predicted_throughput"]
                if measured_delta == 0:
                    continue
                pairs += 1
                if predicted_delta * measured_delta > 0:
                    agreements += 1
    return agreements / pairs if pairs else 1.0


def _planner_regret(rows: list[dict[str, Any]]) -> float:
    by_workload: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_workload.setdefault(str(row.get("workload_id", "default")), []).append(row)
    regrets = []
    for group in by_workload.values():
        predicted_best = max(group, key=lambda item: item["predicted_throughput"])
        measured_best = max(group, key=lambda item: item["measured_throughput"])
        denominator = max(float(measured_best["measured_throughput"]), 1e-12)
        regrets.append(
            max(
                0.0,
                (measured_best["measured_throughput"] - predicted_best["measured_throughput"])
                / denominator,
            )
        )
    return max(regrets, default=0.0)


def calibrate_expert_simulator(
    measured_rows: list[dict[str, Any]],
    *,
    validation_fraction: float = 0.25,
    seed: int = 1010,
    ridge: float = 1e-6,
) -> tuple[ExpertCalibrationModel, list[dict[str, Any]]]:
    """Fit only calibration topologies, then score untouched configuration IDs."""

    rows = _aggregate_configurations(measured_rows)
    calibration, validation = deterministic_calibration_split(
        rows, validation_fraction=validation_fraction, seed=seed
    )
    active_features = tuple(
        name
        for name in EXPERT_COST_FEATURES
        if any(abs(float(row.get(name, 0.0) or 0.0)) > 0 for row in calibration)
    )
    if not active_features:
        raise ValueError("calibration data contains no measured cost components")
    matrix = np.asarray(
        [[float(row.get(name, 0.0) or 0.0) for name in active_features] for row in calibration],
        dtype=np.float64,
    )
    target = np.asarray([float(row["measured_total_ns"]) for row in calibration])
    scales = np.maximum(np.median(np.abs(matrix), axis=0), 1.0)
    normalized = matrix / scales
    design = np.column_stack([np.ones(len(normalized)), normalized])
    regularizer = np.eye(design.shape[1]) * ridge
    regularizer[0, 0] = 0.0
    fitted = np.linalg.solve(design.T @ design + regularizer, design.T @ target)
    # Cost components cannot make a topology faster, except the explicit
    # routing-overlap savings feature whose input is recorded as negative.
    coefficients = np.maximum(fitted[1:], 0.0)
    intercept = max(float(fitted[0]), 0.0)
    calibration_predictions = np.maximum(intercept + normalized @ coefficients, 1.0)
    p95_ratios = [
        float(row["measured_p95_latency_ms"]) / max(prediction / 1e6, 1e-12)
        for row, prediction in zip(calibration, calibration_predictions, strict=True)
        if float(row["measured_p95_latency_ms"]) > 0
    ]
    p95_multiplier = float(np.median(p95_ratios)) if p95_ratios else 1.0
    provisional = ExpertCalibrationModel(
        feature_names=active_features,
        coefficients=tuple(float(item) for item in coefficients),
        feature_scales=tuple(float(item) for item in scales),
        intercept_ns=intercept,
        p95_multiplier=p95_multiplier,
        calibration_configuration_ids=tuple(
            sorted(str(row["configuration_id"]) for row in calibration)
        ),
        validation_configuration_ids=tuple(
            sorted(str(row["configuration_id"]) for row in validation)
        ),
        validation={},
    )
    scored = []
    for row in validation:
        prediction = provisional.predict(row)
        scored.append(
            {
                **row,
                **prediction,
                "throughput_error_fraction": _percentage_error(
                    prediction["predicted_throughput"], row["measured_throughput"]
                ),
                "p95_error_fraction": _percentage_error(
                    prediction["predicted_p95_latency_ms"],
                    row["measured_p95_latency_ms"],
                ),
            }
        )
    median_throughput_error = float(np.median([row["throughput_error_fraction"] for row in scored]))
    p95_latency_error = float(np.percentile([row["p95_error_fraction"] for row in scored], 95))
    ranking = _ranking_agreement(scored)
    regret = _planner_regret(scored)
    gates = {
        "median_throughput_error": median_throughput_error,
        "median_throughput_error_pass": median_throughput_error <= 0.10,
        "p95_latency_error": p95_latency_error,
        "p95_latency_error_pass": p95_latency_error <= 0.15,
        "plan_ranking_agreement": ranking,
        "plan_ranking_agreement_pass": ranking >= 0.80,
        "planner_regret": regret,
        "planner_regret_pass": regret <= 0.05,
    }
    gates["all_gates_pass"] = all(
        bool(value) for key, value in gates.items() if key.endswith("_pass")
    )
    model = ExpertCalibrationModel(
        feature_names=provisional.feature_names,
        coefficients=provisional.coefficients,
        feature_scales=provisional.feature_scales,
        intercept_ns=provisional.intercept_ns,
        p95_multiplier=provisional.p95_multiplier,
        calibration_configuration_ids=provisional.calibration_configuration_ids,
        validation_configuration_ids=provisional.validation_configuration_ids,
        validation=gates,
    )
    for row in scored:
        row["category"] = "SIMULATED_CALIBRATED" if model.validated else "SIMULATED_UNCALIBRATED"
    return model, scored


def project_virtual_topologies(
    model: ExpertCalibrationModel,
    base_row: dict[str, Any],
    *,
    node_counts: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128),
) -> list[dict[str, Any]]:
    results = []
    for nodes in node_counts:
        row = dict(base_row)
        row["configuration_id"] = f"virtual-{nodes}"
        row["worker_compute_ns"] = float(base_row.get("worker_compute_ns", 0.0)) / nodes
        row["microshard_compute_ns"] = float(base_row.get("microshard_compute_ns", 0.0)) / nodes
        row["tcp_transport_ns"] = float(base_row.get("tcp_transport_ns", 0.0)) * nodes
        row["reduction_ns"] = float(base_row.get("reduction_ns", 0.0)) * np.log2(nodes)
        results.append({"node_count": nodes, **model.predict(row)})
    return results


def remote_break_even_surface(
    *,
    worker_compute_speeds: Iterable[float],
    bandwidths_bps: Iterable[float],
    latencies_ms: Iterable[float],
    expert_bytes: int,
    activation_bytes: int,
    selected_experts: int,
    batch_size: int,
    local_compute_ns: float,
    cache_hit_rate: float,
    storage_bandwidth_bytes_s: float | None = None,
    shard_counts: Iterable[int] = (1, 2, 4, 8),
    serialisation_ns: float = 0.0,
    reduction_ns_per_worker: float = 0.0,
    compression_ratio: float = 1.0,
    codec_ns: float = 0.0,
    straggler_probability: float = 0.0,
) -> list[dict[str, Any]]:
    rows = []
    payload = activation_bytes * batch_size * compression_ratio
    for speed in worker_compute_speeds:
        if speed <= 0:
            raise ValueError("worker compute speed must be positive")
        for bandwidth in bandwidths_bps:
            if bandwidth <= 0:
                raise ValueError("break-even bandwidth must be positive")
            for latency in latencies_ms:
                candidates = []
                for shards in shard_counts:
                    transfer_ns = (2 * payload * 8 / bandwidth * 1e9) * shards
                    latency_ns = 2 * latency * 1e6
                    compute_ns = local_compute_ns / speed / shards
                    storage_pressure_bytes = (1 - cache_hit_rate) * expert_bytes * selected_experts
                    storage_ns = (
                        storage_pressure_bytes / storage_bandwidth_bytes_s * 1e9
                        if storage_bandwidth_bytes_s is not None
                        else 0.0
                    )
                    straggler_ns = straggler_probability * compute_ns
                    total_ns = (
                        compute_ns
                        + transfer_ns
                        + latency_ns
                        + serialisation_ns
                        + codec_ns
                        + storage_ns
                        + reduction_ns_per_worker * max(shards - 1, 0)
                        + straggler_ns
                    )
                    candidates.append(
                        (shards, total_ns, transfer_ns, storage_ns, storage_pressure_bytes)
                    )
                best_shards, best_ns, transfer_ns, storage_ns, storage_pressure_bytes = min(
                    candidates, key=lambda item: item[1]
                )
                whole = next(item for item in candidates if item[0] == 1)
                rows.append(
                    {
                        "worker_compute_speed_ratio": speed,
                        "bandwidth_bps": bandwidth,
                        "one_way_latency_ms": latency,
                        "expert_bytes": expert_bytes,
                        "activation_bytes": activation_bytes,
                        "selected_experts": selected_experts,
                        "batch_size": batch_size,
                        "cache_hit_rate": cache_hit_rate,
                        "storage_bandwidth_bytes_s": storage_bandwidth_bytes_s,
                        "storage_pressure_bytes": storage_pressure_bytes,
                        "remote_whole_expert_beneficial": whole[1] < local_compute_ns,
                        "microsharding_beneficial": best_shards > 1
                        and best_ns < min(local_compute_ns, whole[1]),
                        "best_shard_count": best_shards,
                        "expected_utility": local_compute_ns - best_ns,
                        "confidence_interval": None,
                        "dominant_bottleneck": max(
                            {
                                "worker_compute": local_compute_ns / speed / best_shards,
                                "network_transfer": transfer_ns,
                                "network_latency": 2 * latency * 1e6,
                                "storage_read": storage_ns,
                                "reduction": reduction_ns_per_worker * max(best_shards - 1, 0),
                            },
                            key=lambda key: {
                                "worker_compute": local_compute_ns / speed / best_shards,
                                "network_transfer": transfer_ns,
                                "network_latency": 2 * latency * 1e6,
                                "storage_read": storage_ns,
                                "reduction": reduction_ns_per_worker * max(best_shards - 1, 0),
                            }[key],
                        ),
                    }
                )
    return rows
