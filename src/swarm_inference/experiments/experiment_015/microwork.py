"""Evidence-preserving expert-microwork redesign analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.experiments.experiment_015.evidence import (
    atomic_json,
    file_identity,
    read_json,
)


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _round_bfloat16(values: np.ndarray) -> np.ndarray:
    """Round FP32 to BF16 and return an FP32 view of the represented values."""
    source = np.asarray(values, dtype=np.float32)
    bits = source.view(np.uint32).copy()
    rounding = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded = bits + rounding
    rounded &= np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def _quantization_metrics(reference: np.ndarray, restored: np.ndarray) -> dict[str, Any]:
    expected = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(restored, dtype=np.float64)
    delta = actual - expected
    denominator = float(np.linalg.norm(expected))
    return {
        "finite": bool(np.isfinite(actual).all()),
        "maximum_absolute_error": float(np.max(np.abs(delta))),
        "mean_absolute_error": float(np.mean(np.abs(delta))),
        "relative_l2_error": float(np.linalg.norm(delta) / max(denominator, 1e-30)),
        "p99_absolute_error": _percentile(np.abs(delta), 99.0),
    }


def _network_service_ms(
    payload_bytes: float,
    *,
    rtt_ms: float,
    bandwidth_gbps: float,
    software_base_ms: float,
) -> float:
    return software_base_ms + rtt_ms + payload_bytes * 8.0 / (
        bandwidth_gbps * 1_000_000.0
    )


def analyze_microwork(
    repository_root: Path,
    output_path: Path,
    network_surface_path: Path,
    *,
    internal_rtts_ms: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0),
    bandwidths_gbps: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 100.0),
) -> dict[str, Any]:
    """Reconcile measured EP results and calculate explicitly shaped redesign bounds.

    The direct-buffer result is an optimistic subtraction of individually measured
    overheads.  It is deliberately *not* called a benchmark or admitted into the
    architecture search.
    """
    root = repository_root.expanduser().resolve()
    exp014 = root / "artifacts" / "experiment-014"
    scaling_path = exp014 / "sub-layer" / "h014-sub-005-scaling-network.json"
    promoted_path = (
        exp014 / "sub-layer" / "h014-sub-011-promoted-four-worker-real-expert.json"
    )
    trace_path = exp014 / "oracle-full-93" / "hidden-trace.f32"
    scaling = read_json(scaling_path)
    promoted = read_json(promoted_path)

    rows: list[dict[str, Any]] = []
    for source in scaling["scaling_curve"]:
        workers = int(source["workers"])
        layer_p50 = float(source["distributed_layer_p50_ms"])
        reference_p50 = float(source["reference_layer_p50_ms"])
        if workers == 4:
            layer_p50 = float(promoted["performance"]["distributed_complete_layer_wall"]["p50_ms"])
            reference_p50 = float(promoted["performance"]["single_gpu_complete_layer_wall"]["p50_ms"])
        rows.append(
            {
                "workers": workers,
                "evidence_class": "MEASURED",
                "worker_tracked_bytes": (
                    int(promoted["memory"]["largest_worker_tracked_bytes"])
                    if workers == 4
                    else int(source["worker_bytes"])
                ),
                "worker_gib": (
                    int(promoted["memory"]["largest_worker_tracked_bytes"])
                    if workers == 4
                    else int(source["worker_bytes"])
                )
                / (1024.0**3),
                "complete_layer_p50_ms": layer_p50,
                "complete_layer_p95_ms": float(
                    promoted["performance"]["distributed_complete_layer_wall"]["p95_ms"]
                    if workers == 4
                    else source["distributed_layer_p95_ms"]
                ),
                "complete_layer_p99_ms": float(
                    promoted["performance"]["distributed_complete_layer_wall"]["p99_ms"]
                    if workers == 4
                    else source["distributed_layer_p99_ms"]
                ),
                "reference_layer_p50_ms": reference_p50,
                "relative_complete_layer_throughput": reference_p50 / layer_p50,
                "mean_transport_bytes_per_token": float(source["mean_total_transport_bytes"]),
                "critical_path_payload_bytes": int(source["critical_path_payload_bytes"]),
                "synchronization_count_per_layer": int(source["synchronization_points"]),
                "economic_efficiency_if_workers_are_distinct_paid_compute": (
                    reference_p50 / layer_p50 / workers
                ),
            }
        )

    performance = promoted["performance"]
    decomposition = performance["latency_decomposition"]
    measured_four = float(performance["distributed_complete_layer_wall"]["p50_ms"])
    known_removable = (
        float(decomposition["network_transport_exposed"]["p50_ms"])
        + float(decomposition["parent_latent_d2h"]["p50_ms"])
        + float(decomposition["parent_expert_rows_h2d_enqueue"]["p50_ms"])
        + float(decomposition["collection_deserialization"]["p50_ms"])
        + float(decomposition["expert_compute_critical_device"]["p50_ms"])
        - float(performance["ideal_parallel_expert_service_ms"])
    )
    optimistic_direct_ms = measured_four - known_removable
    reference_four = float(performance["single_gpu_complete_layer_wall"]["p50_ms"])

    trace = np.memmap(trace_path, dtype=np.float32, mode="r")
    if trace.size % 7168:
        raise ValueError("retained hidden trace is not composed of 7168-wide Kimi rows")
    sample = np.asarray(trace).reshape(-1, 7168)
    precision_rows = [
        {
            "format": "FP32",
            "bytes_per_scalar": 4,
            "payload_ratio": 1.0,
            "component_error": _quantization_metrics(sample, sample),
            "full_layer_correctness_gate": "PASS_FROM_EXISTING_BASELINE_ONLY",
        },
        {
            "format": "BF16",
            "bytes_per_scalar": 2,
            "payload_ratio": 0.5,
            "component_error": _quantization_metrics(sample, _round_bfloat16(sample)),
            "full_layer_correctness_gate": "NOT_RUN",
        },
        {
            "format": "FP16",
            "bytes_per_scalar": 2,
            "payload_ratio": 0.5,
            "component_error": _quantization_metrics(
                sample, sample.astype(np.float16).astype(np.float32)
            ),
            "full_layer_correctness_gate": "NOT_RUN",
        },
        {
            "format": "FP8_OR_MXFP8",
            "bytes_per_scalar": 1,
            "payload_ratio": 0.25,
            "component_error": None,
            "full_layer_correctness_gate": "UNSUPPORTED_BY_RETAINED_KIMI_KERNEL",
        },
    ]

    four = next(row for row in rows if row["workers"] == 4)
    surface: list[dict[str, Any]] = []
    for precision in precision_rows:
        for rtt in internal_rtts_ms:
            for bandwidth in bandwidths_gbps:
                payload = float(four["mean_transport_bytes_per_token"]) * float(
                    precision["payload_ratio"]
                )
                shaped_network = _network_service_ms(
                    payload,
                    rtt_ms=rtt,
                    bandwidth_gbps=bandwidth,
                    software_base_ms=0.0,
                )
                projected = optimistic_direct_ms + shaped_network
                surface.append(
                    {
                        "evidence_class": "SHAPED",
                        "workers": 4,
                        "format": precision["format"],
                        "rtt_ms": rtt,
                        "bandwidth_gbps": bandwidth,
                        "payload_bytes": payload,
                        "network_ms": shaped_network,
                        "optimistic_complete_layer_ms": projected,
                        "optimistic_relative_throughput": reference_four / projected,
                        "admitted": False,
                    }
                )

    network_receipt: dict[str, Any] = {
        "schema_version": "experiment-015-microwork-network-surface-v1",
        "cycle_id": "H015-004B-H015-005A",
        "status": "PASS",
        "evidence_class": "SHAPED",
        "scope": "direct-device zero-software-base sensitivity, not an implementation result",
        "rows": surface,
    }
    atomic_json(network_surface_path, network_receipt)

    receipt: dict[str, Any] = {
        "schema_version": "experiment-015-microwork-results-v1",
        "cycle_id": "H015-004A-H015-005A",
        "status": "PASS",
        "measured_scaling": rows,
        "best_measured": next(row for row in rows if row["workers"] == 4),
        "direct_resident_redesign_upper_bound": {
            "evidence_class": "PROJECTED",
            "measured_start_ms": measured_four,
            "known_overhead_subtracted_ms": known_removable,
            "optimistic_layer_p50_ms": optimistic_direct_ms,
            "optimistic_relative_throughput": reference_four / optimistic_direct_ms,
            "assumption": "every individually timed host/serialization cost vanishes without replacement cost",
            "admitted": False,
        },
        "transport_precision": {
            "evidence_class": None,
            "scientific_result": False,
            "scope": (
                "CPU round-trip diagnostic on retained real Kimi hidden rows; outside "
                "the four Experiment 015 evidence classes; no full-layer CUDA transport run"
            ),
            "rows": precision_rows,
            "retained_formats": ["FP32"],
        },
        "overlap": {
            "status": "NOT_ESTABLISHED",
            "reason": "no asynchronous device-resident send/receive implementation was executed",
            "same_gpu_contention_from_experiment_014": "did not improve complete-layer economics",
        },
        "decision": "RETAIN_EXISTING_FOUR_WORKER_BASELINE_ONLY",
        "strong_90_percent_gate": {
            "measured_pass": False,
            "projected_upper_bound_pass": reference_four / optimistic_direct_ms >= 0.90,
        },
        "sources": {
            "scaling": file_identity(scaling_path, relative_to=root),
            "promoted_four_worker": file_identity(promoted_path, relative_to=root),
            "hidden_trace": file_identity(trace_path, relative_to=root),
        },
    }
    atomic_json(output_path, receipt)
    return receipt


__all__ = ["analyze_microwork"]
